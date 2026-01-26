import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models
from image_cross_attention import ImageCrossAttention
from crossview_hybrid_attention import CrossViewHybridAttention


class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

class TriPlaneModel(nn.Module):
    """
    [0126 Simple Version].

    For its forward method,  output ray-based Tri-Plane features for a specialized camera position input.

    Core functionality:
    1. Initialize Tri-Plane using satellite image (XY plane uses satellite features, XZ/YZ initialized to 0)
    2. Optimize Tri-Plane features by Cross-View Hybrid Attention(CVHA)
    3. Update Tri-Plane planes (XY/XZ/YZ) with new images and camera parameters by ICA
    4. Output pure Tri-Plane features, shape [B, 3, 32, H, W]
    """
    def __init__(
        self, 
        satellite_img: torch.Tensor,  # input satellite image [B, 3, H, W]
        plane_resolution: tuple = 224,      # Tri-Plane resolution (H, W)
        plane_channel: int = 32,      # channels per plane
        num_of_points_in_pillar: int = 4,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ):
        super().__init__()
        self.satellite_img = satellite_img
        self.device = device
        self.plane_channel = plane_channel
        self.num_planes = 3                   # 3 plane：XY/XZ/YZ
        self.plane_h, self.plane_w = plane_resolution  # Tri-Plane resolution

        # initialize ResNet50 for satellite image feature extraction
        self.satellite_feature_extractor = self._build_satellite_feature_extractor()
        
        
        # 3. 相机参数编码器：将相机参数→三个平面的更新权重（无渲染逻辑，仅编码权重）
        # 相机参数维度可自定义（如输入12维外参+4维内参=16维）
        self.cam_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_planes),
            nn.Softmax(dim=-1)  # 权重和为1，控制各平面更新幅度
        ).to(self.device)
        
        self.cross_view_hybrid_attention = CrossViewHybridAttention(
            embed_dim=self.plane_channel,
            num_heads=8,
            num_levels=1,
            num_points=4,
            num_tpv_queue=1,
        ).to(self.device)


    @torch.no_grad()
    def _build_satellite_feature_extractor(self):
        resnet = models.resnet50(pretrained=True)
        # get the output of layer4
        feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        # frozen parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        return feature_extractor.to(self.device)

    def _init_tri_plane(self, satellite_img):
        """
        Get Tri-plane from a single satellite image.
        """
        satellite_img_feature_extractor = self._build_satellite_feature_extractor()
        satellite_img_feature = satellite_img_feature_extractor(self.satellite_img)

        tri_plane_hw = satellite_img_feature
        tri_plane_hz = torch.zeros_like(satellite_img_feature)
        tri_plane_wz = torch.zeros_like(satellite_img_feature)

        cross-View_hybrid_attention = CrossViewHybridAttention(
            embed_dim=tri_plane_hw.shape[1],
            num_heads=8,
            num_levels=1,
            num_points=4,
            num_tpv_queue=1,
        ).to(self.device)

        reference_points_cvha = cross-View_hybrid_attention.get_reference_points()


    def forward(self, satellite_img: torch.Tensor):
        """
        Get Tri-plane features from a satellite image.
        :param cam_param_condition: camera parameters [B, 4, 4]
        """

        satellite_img_feature_extractor = self._build_satellite_feature_extractor()
        satellite_img_feature = satellite_img_feature_extractor(self.satellite_img)

        tri_plane_hw = satellite_img_feature
        tri_plane_hz = torch.zeros_like(satellite_img_feature)
        tri_plane_wz = torch.zeros_like(satellite_img_feature)

        reference_points_cvha = self.cross_view_hybrid_attention.get_cross_view_ref_points(tri_plane_hw, tri_plane_hz, tri_plane_wz)

        self.cross_view_hybrid_attention(tri_plane_hw, tri_plane_hz, tri_plane_wz, reference_points_cvha)
        tri_plane = torch.stack([tri_plane_hw, tri_plane_hz, tri_plane_wz], dim=1)  # [B, 3, 32, H, W]


        return tri_plane

class RayBasedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_levels, num_points):
        super(RayBasedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        # Define layers here (e.g., multi-head attention layers)
        # This is a placeholder for the actual implementation

    def forward(self, query, key, value, reference_points):
        # Implement the attention mechanism here
        # This is a placeholder for the actual implementation
        attn_output = query  # Replace with actual attention output
        return attn_output

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}