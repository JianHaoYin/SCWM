import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Ray sampler (supports HxW)
# -----------------------------
class RaySamplerHW(nn.Module):
    """Create ray origins and directions for an arbitrary HxW image."""
    def __init__(self):
        super().__init__()

    def forward(self, cam2world: torch.Tensor, intrinsics: torch.Tensor, out_h: int, out_w: int):
        """
        Args:
            cam2world: [B,4,4]
            intrinsics: [B,3,3]
        Returns:
            ray_origins: [B, HW, 3]
            ray_dirs:    [B, HW, 3] normalized
        """
        device = cam2world.device
        B = cam2world.shape[0]
        HW = out_h * out_w

        cam_locs_world = cam2world[:, :3, 3]  # [B,3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        # uv in [0,1] image coordinates (center-of-pixel convention)
        yy, xx = torch.meshgrid(
            torch.arange(out_h, dtype=torch.float32, device=device),
            torch.arange(out_w, dtype=torch.float32, device=device),
            indexing="ij"
        )
        u = (xx + 0.5) / out_w
        v = (yy + 0.5) / out_h
        uv = torch.stack([u, v], dim=-1).view(1, HW, 2).repeat(B, 1, 1)  # [B,HW,2]

        x_cam = uv[:, :, 0]  # [B,HW]
        y_cam = uv[:, :, 1]
        z_cam = torch.ones((B, HW), device=device)

        # OpenCV pinhole (same as your RaySampler)
        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
                  - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)  # [B,HW,4]
        world_rel_points = torch.bmm(cam2world, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]  # [B,HW,3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = F.normalize(ray_dirs, dim=-1)
        ray_origins = cam_locs_world[:, None, :].repeat(1, HW, 1)

        return ray_origins, ray_dirs


# ----------------------------------------------------
# 2) Sample from heterogeneous tri-planes (xy/xz/yz)
# ----------------------------------------------------
def sample_from_triplane_hetero(triplane: dict, coords: torch.Tensor, box_warp: float):
    """
    Sample features from tri-planes with different resolutions.

    Args:
        triplane:
            xy: [B,C,Hxy,Wxy]
            xz: [B,C,Z,Hxy]     (height=Z, width=Hxy)
            yz: [B,C,Wxy,Z]     (height=Wxy, width=Z)
        coords: [B,M,3] world coordinates in box space (assumed in [-box_warp/2, box_warp/2])
        box_warp: scene box side length
    Returns:
        feat: [B,M,C] fused features (sum of 3 planes)
    """
    xy = triplane["xy"]
    xz = triplane["xz"]
    yz = triplane["yz"]
    B, C, Hxy, Wxy = xy.shape
    _, _, Z, HxzW = xz.shape
    _, _, HyzH, Wyz = yz.shape
    assert HxzW == Hxy, "xz width must match Hxy in this layout"
    assert HyzH == Wxy and Wyz == Z, "yz layout must be [B,C,Wxy,Z]"

    # Normalize coords to [-1,1] for grid_sample
    # coords assumed in [-box_warp/2, box_warp/2]
    scale = 2.0 / box_warp
    x = coords[..., 0] * scale  # [B,M]
    y = coords[..., 1] * scale
    z = coords[..., 2] * scale

    # grid_sample grid: [B, M, 1, 2] (x is width axis, y is height axis)
    # XY plane: width->x, height->y
    grid_xy = torch.stack([x, y], dim=-1).view(B, -1, 1, 2)

    # XZ plane is stored as [B,C,Z,Hxy] => height->z, width->x(we use Hxy axis as "x axis")
    grid_xz = torch.stack([x, z], dim=-1).view(B, -1, 1, 2)

    # YZ plane is stored as [B,C,Wxy,Z] => height->y, width->z
    grid_yz = torch.stack([z, y], dim=-1).view(B, -1, 1, 2)

    # Sample each plane
    feat_xy = F.grid_sample(xy, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B,C,M,1]
    feat_xz = F.grid_sample(xz, grid_xz, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B,C,M,1]
    feat_yz = F.grid_sample(yz, grid_yz, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B,C,M,1]

    # Fuse: sum (you can also concat)
    feat = feat_xy + feat_xz + feat_yz  # [B,C,M,1]
    feat = feat.squeeze(-1).permute(0, 2, 1).contiguous()  # [B,M,C]
    return feat


# -----------------------------
# 3) A simple decoder (feature + sigma)
# -----------------------------
class FeatureSigmaDecoder(nn.Module):
    """Decode sampled tri-plane features into per-point feature and density."""
    def __init__(self, in_dim: int, out_feat_dim: int):
        super().__init__()
        hidden = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_feat = nn.Linear(hidden, out_feat_dim)
        self.to_sigma = nn.Linear(hidden, 1)

    def forward(self, sampled_feat: torch.Tensor, ray_dirs: torch.Tensor):
        """
        Args:
            sampled_feat: [B, M, C]
            ray_dirs:     [B, M, 3]  (can be unused for now)
        Returns:
            feat:  [B, M, F]
            sigma: [B, M, 1]
        """
        x = self.mlp(sampled_feat)
        feat = self.to_feat(x)
        sigma = self.to_sigma(x)
        return {"feat": feat, "sigma": sigma}


# -----------------------------
# 4) A minimal ray marcher (alpha compositing)
# -----------------------------
def volume_render(feat: torch.Tensor, sigma: torch.Tensor, depths: torch.Tensor):
    """
    Simple NeRF-style volume rendering (no mip, no fancy marcher).
    Args:
        feat:   [B, R, S, F]
        sigma:  [B, R, S, 1]
        depths: [B, R, S, 1]
    Returns:
        feat_img:  [B, R, F]
        depth_img: [B, R, 1]
    """
    # delta between consecutive samples
    deltas = depths[:, :, 1:, :] - depths[:, :, :-1, :]
    delta_last = torch.full_like(deltas[:, :, :1, :], 1e10)
    deltas = torch.cat([deltas, delta_last], dim=2)  # [B,R,S,1]

    # alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-F.relu(sigma) * deltas)  # [B,R,S,1]

    # transmittance T
    # T_i = Π_{j<i} (1-alpha_j)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1, :]), 1.0 - alpha + 1e-10], dim=2), dim=2)[:, :, :-1, :]
    weights = alpha * T  # [B,R,S,1]

    feat_img = (weights * feat).sum(dim=2)  # [B,R,F]
    depth_img = (weights * depths).sum(dim=2)  # [B,R,1]
    return feat_img, depth_img


# -----------------------------
# 5) Renderer: tri-plane + rays -> feature image
# -----------------------------
class TriPlaneImportanceRenderer(nn.Module):
    """A simplified coarse-only renderer (easy to debug)."""
    def __init__(self):
        super().__init__()

    def forward(self, triplane: dict, decoder: nn.Module, ray_origins: torch.Tensor, ray_dirs: torch.Tensor, opts: dict):
        """
        Args:
            triplane: dict {"xy","xz","yz"}
            ray_origins: [B,R,3]
            ray_dirs:    [B,R,3]
        Returns:
            feat_img: [B,R,F]
            depth_img:[B,R,1]
        """
        B, R, _ = ray_origins.shape
        S = int(opts["depth_resolution"])
        ray_start = opts.get("ray_start", 0.0)
        ray_end = opts.get("ray_end", 1.0)
        box_warp = float(opts["box_warp"])

        # stratified depth samples in [ray_start, ray_end]
        t_vals = torch.linspace(ray_start, ray_end, S, device=ray_origins.device).view(1, 1, S, 1)
        t_vals = t_vals.repeat(B, R, 1, 1)  # [B,R,S,1]

        # sample points: p = o + t*d
        sample_coords = ray_origins.unsqueeze(2) + t_vals * ray_dirs.unsqueeze(2)  # [B,R,S,3]
        sample_coords = sample_coords.view(B, R * S, 3)  # [B,RS,3]

        sample_dirs = ray_dirs.unsqueeze(2).expand(B, R, S, 3).reshape(B, R * S, 3)  # [B,RS,3]

        # tri-plane sampling -> [B,RS,C]
        sampled_feat = sample_from_triplane_hetero(triplane, sample_coords, box_warp=box_warp)

        # decode -> feat + sigma
        out = decoder(sampled_feat, sample_dirs)
        feat = out["feat"].view(B, R, S, -1)    # [B,R,S,F]
        sigma = out["sigma"].view(B, R, S, 1)   # [B,R,S,1]

        # volume render
        feat_img, depth_img = volume_render(feat, sigma, t_vals)
        return feat_img, depth_img


class TriPlaneFeatureRenderer(nn.Module):
    """User-facing module: tri-plane + camera -> image feature map."""
    def __init__(self, plane_feat_dim: int = 32, out_feat_dim: int = 32):
        super().__init__()
        self.ray_sampler = RaySamplerHW()
        self.renderer = TriPlaneImportanceRenderer()
        self.decoder = FeatureSigmaDecoder(in_dim=plane_feat_dim, out_feat_dim=out_feat_dim)

    def forward(self, triplane: dict, cam2world: torch.Tensor, intrinsics: torch.Tensor, out_h: int, out_w: int, opts: dict):
        """
        Returns:
            feat_image:  [B, F, out_h, out_w]
            depth_image: [B, 1, out_h, out_w]
        """
        ray_o, ray_d = self.ray_sampler(cam2world, intrinsics, out_h, out_w)  # [B,HW,3]
        feat_rays, depth_rays = self.renderer(triplane, self.decoder, ray_o, ray_d, opts)  # [B,HW,F], [B,HW,1]

        B = cam2world.shape[0]
        Fdim = feat_rays.shape[-1]
        feat_image = feat_rays.transpose(1, 2).contiguous().view(B, Fdim, out_h, out_w)
        depth_image = depth_rays.transpose(1, 2).contiguous().view(B, 1, out_h, out_w)
        return feat_image, depth_image
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake tri-plane for test
    B, C, H, W, Z = 1, 32, 224, 224, 64
    triplane = {
        "xy": torch.randn(B, C, H, W, device=device),
        "xz": torch.randn(B, C, Z, H, device=device),
        "yz": torch.randn(B, C, W, Z, device=device),
    }

    # Dummy camera
    cam2world = torch.eye(4, device=device).view(1, 4, 4)
    intr = torch.tensor([[[500.0, 0.0, 0.5],
                          [0.0, 500.0, 0.5],
                          [0.0, 0.0, 1.0]]], device=device)

    renderer = TriPlaneFeatureRenderer(plane_feat_dim=32, out_feat_dim=32).to(device)

    opts = {
        "box_warp": 2.0,            # coordinates assumed in [-1,1]
        "ray_start": 0.0,
        "ray_end": 2.0,
        "depth_resolution": 32,     # coarse samples per ray
    }

    feat_img, depth_img = renderer(triplane, cam2world, intr, out_h=64, out_w=96, opts=opts)
    print("feat_img:", feat_img.shape, feat_img.min().item(), feat_img.max().item())
    print("depth_img:", depth_img.shape, depth_img.min().item(), depth_img.max().item())