
import torch
# from torch.cuda.amp import custom_bwd, custom_fwd
# from torch.autograd.function import Function, once_differentiable

import torch
import torch.nn.functional as F


def multi_scale_deform_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch multi-scale deformable attention.

    Args:
        value: (bs, num_value, num_heads, head_dim)
        value_spatial_shapes: (num_levels, 2) each (H_l, W_l)
        value_level_start_index: (num_levels,) start index of each level within num_value
        sampling_locations: (bs, num_query, num_heads, num_levels, num_points, 2)
            locations are normalized in [0,1], last dim is (x, y)
        attention_weights: (bs, num_query, num_heads, num_levels, num_points)

    Returns:
        output: (bs, num_query, num_heads*head_dim)
    """
    assert value.dim() == 4
    bs, num_value, num_heads, head_dim = value.shape
    bs2, num_query, num_heads2, num_levels, num_points, _ = sampling_locations.shape
    assert bs2 == bs and num_heads2 == num_heads
    assert attention_weights.shape == (bs, num_query, num_heads, num_levels, num_points)

    # We will accumulate in fp32 for stability, then cast back
    orig_dtype = value.dtype
    value_f = value.float()
    sampling_locations_f = sampling_locations.float()
    attention_weights_f = attention_weights.float()

    # Convert sampling locations from [0,1] to [-1,1] for grid_sample
    # grid_sample expects (x, y) in [-1,1]
    sampling_grids = sampling_locations_f * 2.0 - 1.0  # (bs, nq, nh, nl, np, 2)

    # Output accumulator: (bs, nq, nh, head_dim)
    output = value_f.new_zeros((bs, num_query, num_heads, head_dim))

    for lvl in range(num_levels):
        H_l = int(value_spatial_shapes[lvl, 0].item())
        W_l = int(value_spatial_shapes[lvl, 1].item())
        start = int(value_level_start_index[lvl].item())
        end = start + H_l * W_l

        # Slice value for this level and reshape to feature map
        # value_l: (bs, H_l*W_l, nh, hd) -> (bs, nh, hd, H_l, W_l)
        value_l = value_f[:, start:end, :, :]  # (bs, Hl*Wl, nh, hd)
        value_l = value_l.permute(0, 2, 3, 1).contiguous()
        value_l = value_l.view(bs, num_heads, head_dim, H_l, W_l)

        # Prepare for grid_sample: merge batch and head dims
        # v: (bs*nh, hd, H_l, W_l)
        v = value_l.view(bs * num_heads, head_dim, H_l, W_l)

        # Grid for this level: (bs, nq, nh, np, 2) -> (bs*nh, nq, np, 2)
        grid_l = sampling_grids[:, :, :, lvl, :, :]  # (bs, nq, nh, np, 2)
        grid_l = grid_l.permute(0, 2, 1, 3, 4).contiguous()  # (bs, nh, nq, np, 2)
        grid_l = grid_l.view(bs * num_heads, num_query, num_points, 2)

        # Sample: output (bs*nh, hd, nq, np)
        sampled = F.grid_sample(
            v,
            grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # sampled -> (bs, nh, hd, nq, np) -> (bs, nq, nh, np, hd)
        sampled = sampled.view(bs, num_heads, head_dim, num_query, num_points)
        sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()

        # Attention weights for this level: (bs, nq, nh, np)
        attn_l = attention_weights_f[:, :, :, lvl, :]  # (bs, nq, nh, np)
        attn_l = attn_l.unsqueeze(-1)  # (bs, nq, nh, np, 1)

        # Weighted sum over points, accumulate
        output = output + (sampled * attn_l).sum(dim=3)  # sum over np -> (bs, nq, nh, hd)

    output = output.reshape(bs, num_query, num_heads * head_dim)
    return output.to(orig_dtype)

# class MultiScaleDeformableAttnFunction_fp16(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, im2col_step):
#         """GPU version of multi-scale deformable attention.

#         Args:
#             value (Tensor): The value has shape
#                 (bs, num_keys, mum_heads, embed_dims//num_heads)
#             value_spatial_shapes (Tensor): Spatial shape of
#                 each feature map, has shape (num_levels, 2),
#                 last dimension 2 represent (h, w)
#             sampling_locations (Tensor): The location of sampling points,
#                 has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points, 2),
#                 the last dimension 2 represent (x, y).
#             attention_weights (Tensor): The weight of sampling points used
#                 when calculate the attention, has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points),
#             im2col_step (Tensor): The step used in image to column.

#         Returns:
#             Tensor: has shape (bs, num_queries, embed_dims)
#         """
#         ctx.im2col_step = im2col_step
#         output = ext_module.ms_deform_attn_forward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             im2col_step=ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes,
#                               value_level_start_index, sampling_locations,
#                               attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         """GPU version of backward function.

#         Args:
#             grad_output (Tensor): Gradient
#                 of output tensor of forward.

#         Returns:
#              Tuple[Tensor]: Gradient
#                 of input tensors in forward.
#         """
#         value, value_spatial_shapes, value_level_start_index, \
#             sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value = torch.zeros_like(value)
#         grad_sampling_loc = torch.zeros_like(sampling_locations)
#         grad_attn_weight = torch.zeros_like(attention_weights)

#         ext_module.ms_deform_attn_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             grad_output.contiguous(),
#             grad_value,
#             grad_sampling_loc,
#             grad_attn_weight,
#             im2col_step=ctx.im2col_step)

#         return grad_value, None, None, \
#             grad_sampling_loc, grad_attn_weight, None


# class MultiScaleDeformableAttnFunction_fp32(Function):

#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, im2col_step):
#         """GPU version of multi-scale deformable attention.

#         Args:
#             value (Tensor): The value has shape
#                 (bs, num_keys, mum_heads, embed_dims//num_heads)
#             value_spatial_shapes (Tensor): Spatial shape of
#                 each feature map, has shape (num_levels, 2),
#                 last dimension 2 represent (h, w)
#             sampling_locations (Tensor): The location of sampling points,
#                 has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points, 2),
#                 the last dimension 2 represent (x, y).
#             attention_weights (Tensor): The weight of sampling points used
#                 when calculate the attention, has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points),
#             im2col_step (Tensor): The step used in image to column.

#         Returns:
#             Tensor: has shape (bs, num_queries, embed_dims)
#         """

#         ctx.im2col_step = im2col_step
#         output = ext_module.ms_deform_attn_forward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             im2col_step=ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes,
#                               value_level_start_index, sampling_locations,
#                               attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         """GPU version of backward function.

#         Args:
#             grad_output (Tensor): Gradient
#                 of output tensor of forward.

#         Returns:
#              Tuple[Tensor]: Gradient
#                 of input tensors in forward.
#         """
#         value, value_spatial_shapes, value_level_start_index, \
#             sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value = torch.zeros_like(value)
#         grad_sampling_loc = torch.zeros_like(sampling_locations)
#         grad_attn_weight = torch.zeros_like(attention_weights)

#         ext_module.ms_deform_attn_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             grad_output.contiguous(),
#             grad_value,
#             grad_sampling_loc,
#             grad_attn_weight,
#             im2col_step=ctx.im2col_step)

#         return grad_value, None, None, \
#             grad_sampling_loc, grad_attn_weight, None


# if __name__ == "__main__":
#     from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
#     value = torch.randint(10, (1, 16, 1, 1)).float().cuda()
#     v = value.squeeze().reshape(4, 4)
#     spatial_shapes = torch.tensor([[4, 4]]).cuda()
#     level_start_index = torch.tensor([0]).cuda()
#     sampling_locations = torch.tensor([0.375, 0.875]).reshape(1, 1, 1, 1, 1, -1).cuda()
#     attention_weights = torch.tensor([1.0]).reshape(1, 1, 1, 1, 1).cuda()
#     gpuFunc = MultiScaleDeformableAttnFunction_fp32.apply(
#         value, spatial_shapes, level_start_index, sampling_locations, attention_weights, 64
#     )

#     cpuFun = multi_scale_deformable_attn_pytorch(
#         value, spatial_shapes, sampling_locations, attention_weights
#     )
#     import pdb; pdb.set_trace()
#     pass