import torch
import torch.nn as nn


from tri_plane import TriPlaneModel
from ray_based_attention import TriPlaneFeatureRenderer


class SatelliteToTargetFeatureModel(nn.Module):
    """
    End-to-end training forward:
      satellite image -> tri-plane -> render target-view feature map.

    Inputs:
      satellite_img: [B,3,Hs,Ws]
      cam2world:     [B,4,4]   (OpenCV convention; consistent with RaySampler)
      intrinsics:    [B,3,3]
      out_h, out_w:  output feature resolution

    Outputs:
      feat_image:  [B,F,out_h,out_w]
      depth_image: [B,1,out_h,out_w]  (optional but useful)
      aux: dict    (debug info)
    """

    def __init__(
        self,
        # --- tri-plane config ---
        tri_plane_hw=(224, 224),
        tri_plane_z=64,
        tri_plane_c=32,
        num_points_in_pillar=(4, 4, 4),

        # --- render config ---
        out_feat_dim=32,

        # --- device ---
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 1) Satellite -> Tri-plane
        self.triplane_model = TriPlaneModel(
            plane_hw=tri_plane_hw,
            plane_z=tri_plane_z,
            plane_channel=tri_plane_c,
            num_points_in_pillar=num_points_in_pillar,
            device=device,
        ).to(device)

        # 2) Tri-plane -> Rendered feature map
        self.renderer = TriPlaneFeatureRenderer(
            plane_feat_dim=tri_plane_c,
            out_feat_dim=out_feat_dim,
        ).to(device)

        self.out_feat_dim = out_feat_dim

    def forward(
        self,
        satellite_img: torch.Tensor,
        cam2world: torch.Tensor,
        intrinsics: torch.Tensor,
        out_h: int,
        out_w: int,
        render_opts: dict,
        return_aux: bool = True,
    ):
        """
        render_opts example:
        {
            "box_warp": 2.0,
            "ray_start": 0.0,
            "ray_end": 2.0,
            "depth_resolution": 32,
            # optional later:
            # "depth_resolution_importance": 0,
            # "density_noise": 0.0,
        }
        """
        satellite_img = satellite_img.to(self.device)
        cam2world = cam2world.to(self.device)
        intrinsics = intrinsics.to(self.device)

        # --- Step A: Build tri-plane ---
        # IMPORTANT: do NOT wrap this in torch.no_grad() because proj + CVHA should be trainable.
        triplane = self.triplane_model(satellite_img)  # dict: {"xy","xz","yz"}

        # --- Step B: Render target-view feature image ---
        feat_img, depth_img = self.renderer(
            triplane=triplane,
            cam2world=cam2world,
            intrinsics=intrinsics,
            out_h=out_h,
            out_w=out_w,
            opts=render_opts,
        )

        if not return_aux:
            return feat_img, depth_img

        aux = {
            "triplane_xy": triplane["xy"],
            "triplane_xz": triplane["xz"],
            "triplane_yz": triplane["yz"],
        }
        return feat_img, depth_img, aux