import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models
from tri_plane.image_cross_attention import ImageCrossAttention
from tri_plane.crossview_hybrid_attention import CrossViewHybridAttention



def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize input for ImageNet pretrained ResNet. Accepts [0,1] or [0,255]."""
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        x = x.float()
    if x.max() > 2.0:
        x = x / 255.0
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std



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
        plane_hw=(224, 224),          # (H, W) for XY plane
        plane_z=64,                   # Z resolution for XZ/YZ planes
        plane_channel=32,
        num_points_in_pillar=(16, 16, 16),  # (for wz, zh, hw) as in your ref-points generator
        num_heads=8,
        dropout=0.1,
        device=None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.H, self.W = plane_hw
        self.Z = int(plane_z)
        self.C = int(plane_channel)
        self.num_points_in_pillar = num_points_in_pillar

        # --- Backbone: frozen ResNet50 up to layer4 ---
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B,2048,h',w']
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Project backbone channels to plane channels
        self.proj = nn.Conv2d(2048, self.C, kernel_size=1, bias=False)

        # --- CVHA: must match your implementation constraints ---
        # IMPORTANT:
        #   - num_levels=3 because we pass XY/XZ/YZ as 3 levels
        #   - num_tpv_queue=2 because your CVHA forward asserts it and fuses 2 queues
        self.cvha = CrossViewHybridAttention(
            embed_dims=self.C,
            num_heads=num_heads,
            num_levels=3,
            num_points=4,
            dropout=dropout,
            num_tpv_queue=2,
            batch_first=True,
        ).to(self.device)

        self.to(self.device)

        self.plane_channel = plane_channel


    def _init_planes(self, satellite_img: torch.Tensor):
        """Initialize XY from satellite image feature; XZ/YZ as zeros."""

        x = imagenet_normalize(satellite_img)
        # backbone is frozen, so explicitly no_grad here
        with torch.no_grad():
            feat = self.backbone(x)    # [B,2048,h',w']
        # proj MUST have gradients
        feat = self.proj(feat)         # [B,C,h',w']
        xy = F.interpolate(
            feat, size=(self.H, self.W),
            mode="bilinear", align_corners=False
        )                               # [B,C,H,W]
        xz = torch.zeros(
            (xy.shape[0], self.C, self.Z, self.H),
            device=xy.device, dtype=xy.dtype
        )
        yz = torch.zeros(
            (xy.shape[0], self.C, self.W, self.Z),
            device=xy.device, dtype=xy.dtype
        )
        return xy, xz, yz

    def _build_deform_aux(self):
        """Build spatial_shapes and level_start_index for 3 levels with different shapes."""
        # Level0: XY -> (H, W)
        # Level1: XZ -> (Z, H)
        # Level2: YZ -> (W, Z)
        spatial_shapes = torch.tensor(
            [[self.H, self.W], [self.Z, self.H], [self.W, self.Z]],
            dtype=torch.long, device=self.device
        )  # [3,2]

        level_start_index = torch.zeros((3,), dtype=torch.long, device=self.device)
        level_start_index[1] = self.H * self.W
        level_start_index[2] = self.H * self.W + self.Z * self.H
        return spatial_shapes, level_start_index

    def _build_reference_points(self, B: int):
        """Build reference_points [B, N, 3, 2] using your TPVFormer-style generator.

        Your generator returns [N, 3, P, 2]. We reduce P by mean -> [N, 3, 2],
        then repeat for batch -> [B, N, 3, 2].
        """
        ref = CrossViewHybridAttention.get_cross_view_ref_points(
            tpv_h=self.H,
            tpv_w=self.W,
            tpv_z=self.Z,
            num_points_in_pillar=self.num_points_in_pillar
        )  # [N,3,P,2]

        ref = ref.to(self.device)
        ref = ref.mean(dim=2)                   # [N,3,2]
        ref = ref.unsqueeze(0).repeat(B, 1, 1, 1).contiguous()  # [B,N,3,2]
        return ref

    def forward(self, satellite_img: torch.Tensor):
        """
        Args:
            satellite_img: [B,3,H_in,W_in]
        Returns:
            dict with:
              - xy: [B,C,H,W]
              - xz: [B,C,Z,H]
              - yz: [B,C,W,Z]
        """
        satellite_img = satellite_img.to(self.device)

        # 1) Initialize planes
        xy, xz, yz = self._init_planes(satellite_img)
        B = xy.shape[0]

        # 2) Flatten each plane and concat as query/value tokens
        # XY tokens: [B, H*W, C]
        xy_tok = xy.flatten(2).transpose(1, 2).contiguous()
        # XZ tokens: [B, Z*H, C]
        xz_tok = xz.flatten(2).transpose(1, 2).contiguous()
        # YZ tokens: [B, W*Z, C]
        yz_tok = yz.flatten(2).transpose(1, 2).contiguous()

        query = torch.cat([xy_tok, xz_tok, yz_tok], dim=1)  # [B, N, C]
        # For num_tpv_queue=2, value must be [2B, N, C]
        value = torch.cat([query, query], dim=0)            # [2B, N, C]

        # 3) Build deformable attention auxiliary tensors
        spatial_shapes, level_start_index = self._build_deform_aux()
        reference_points = self._build_reference_points(B)  # [B, N, 3, 2]

        # 4) CVHA fusion
        out = self.cvha(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )  # [B, N, C]

        # 5) Split back and reshape to planes
        n_xy = self.H * self.W
        n_xz = self.Z * self.H
        n_yz = self.W * self.Z

        out_xy = out[:, :n_xy, :].transpose(1, 2).contiguous().view(B, self.C, self.H, self.W)
        out_xz = out[:, n_xy:n_xy + n_xz, :].transpose(1, 2).contiguous().view(B, self.C, self.Z, self.H)
        out_yz = out[:, n_xy + n_xz:n_xy + n_xz + n_yz, :].transpose(1, 2).contiguous().view(B, self.C, self.W, self.Z)

        return {"xy": out_xy, "xz": out_xz, "yz": out_yz}


    # @torch.no_grad()
    # def _build_satellite_feature_extractor(self):
    #     resnet = models.resnet50(pretrained=True)
    #     # get the output of layer4
    #     feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
    #     # frozen parameters
    #     for param in feature_extractor.parameters():
    #         param.requires_grad = False
    #     feature_extractor.eval()
    #     return feature_extractor.to(self.device)

    # def _init_tri_plane(self, satellite_img):
    #     """
    #     Get Tri-plane from a single satellite image.
    #     """
    #     satellite_img_feature_extractor = self._build_satellite_feature_extractor()
    #     satellite_img_feature = satellite_img_feature_extractor(self.satellite_img)

    #     tri_plane_hw = satellite_img_feature
    #     tri_plane_hz = torch.zeros_like(satellite_img_feature)
    #     tri_plane_wz = torch.zeros_like(satellite_img_feature)

    #     cross-View_hybrid_attention = CrossViewHybridAttention(
    #         embed_dim=tri_plane_hw.shape[1],
    #         num_heads=8,
    #         num_levels=1,
    #         num_points=4,
    #         num_tpv_queue=1,
    #     ).to(self.device)

    #     reference_points_cvha = cross-View_hybrid_attention.get_reference_points()


    # def forward(self, satellite_img: torch.Tensor):
    #     """
    #     Get Tri-plane features from a satellite image.
    #     :param cam_param_condition: camera parameters [B, 4, 4]
    #     """


    #     satellite_img_feature = self.satellite_img_feature_extractor(satellite_img)

    #     tri_plane_hw = satellite_img_feature
    #     tri_plane_hz = torch.zeros_like(satellite_img_feature)
    #     tri_plane_wz = torch.zeros_like(satellite_img_feature)

    #     reference_points_cvha = self.cross_view_hybrid_attention.get_cross_view_ref_points(tri_plane_hw, tri_plane_hz, tri_plane_wz)

    #     self.cross_view_hybrid_attention(tri_plane_hw, tri_plane_hz, tri_plane_wz, reference_points_cvha)
    #     tri_plane = torch.stack([tri_plane_hw, tri_plane_hz, tri_plane_wz], dim=1)  # [B, 3, 32, H, W]


    #     return tri_plane


    
##############################################################################
# Below is a simple test script for the TriPlaneModel
##############################################################################

import os
import torch
from PIL import Image
from torchvision import transforms



def load_image_as_tensor(path: str, device: torch.device):
    """Load an RGB image and convert to tensor [1,3,H,W] in float."""
    img = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()  # outputs float in [0,1]
    x = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W]
    return x


def save_plane_channel_as_image(t: torch.Tensor, out_path: str):
    """Save plane's channel-0 as a grayscale PNG for quick inspection.
    t: [C,H,W] or [1,C,H,W]
    """
    if t.dim() == 4:
        t = t[0]
    c0 = t[0]  # [H,W]
    # Normalize to [0,1] for saving
    c0 = c0.detach().float()
    c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)
    img = transforms.ToPILImage()(c0.cpu())
    img.save(out_path)


if __name__ == "__main__":
    # ---- Config ----
    satellite_img_path = "/data/tlxd/Cross-View/SCWM/tmp_test/-27.4603,153.021201.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build model ----
    # Adjust these if you want
    model = TriPlaneModel(
        plane_hw=(224, 224),     # XY plane resolution
        plane_z=64,              # Z resolution for XZ/YZ
        plane_channel=256,
        num_points_in_pillar=(16, 16, 16),
        device=device,
    ).to(device)
    model.eval()

    # ---- Load image ----
    assert os.path.exists(satellite_img_path), f"Image not found: {satellite_img_path}"
    sat = load_image_as_tensor(satellite_img_path, device=device)

    # ---- Forward ----
    with torch.no_grad():
        out = model(sat)

    # ---- Print results ----
    xy, xz, yz = out["xy"], out["xz"], out["yz"]
    print("XY:", xy.shape, "min/max:", float(xy.min()), float(xy.max()))
    print("XZ:", xz.shape, "min/max:", float(xz.min()), float(xz.max()))
    print("YZ:", yz.shape, "min/max:", float(yz.min()), float(yz.max()))

    # ---- Save quick visualization ----
    os.makedirs("./tri_plane_debug", exist_ok=True)
    save_plane_channel_as_image(xy, "./tri_plane_debug/xy_c0.png")
    # For XZ/YZ, save a slice-like view (they're 2D maps already):
    # XZ is [B,C,Z,H] -> treat as image [Z,H]
    save_plane_channel_as_image(xz, "./tri_plane_debug/xz_c0.png")
    # YZ is [B,C,W,Z] -> treat as image [W,Z]
    save_plane_channel_as_image(yz, "./tri_plane_debug/yz_c0.png")

    print("Saved debug images to ./tri_plane_debug/")