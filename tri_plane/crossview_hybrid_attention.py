from tri_plane.multi_scale_deformable_attn_function import multi_scale_deform_attn_pytorch
import warnings
import torch
import torch.nn as nn
import math


class CrossViewHybridAttention(nn.Module):
    """Cross view hybrid attention module. Based on deformable attention.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
         (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 deformable_attention=None,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 num_tpv_queue=1):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        if deformable_attention is None:
            deformable_attention = multi_scale_deform_attn_pytorch
        self.deformable_attention = deformable_attention
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_tpv_queue = num_tpv_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * num_tpv_queue, num_tpv_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_tpv_queue,
                                           num_tpv_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()



    def init_weights(self):
        '''Default initialization for Parameters of Module (PyTorch原生实现).'''
        # init sampling_offesets
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        
        # 
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        # (num_heads, 1, 1, 2) -> (num_heads, num_levels*num_tpv_queue, num_points, 2)
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels * self.num_tpv_queue, self.num_points, 1)
        # 
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        # 
        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        # 
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        

        nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)
        nn.init.constant_(self.value_proj.bias, 0.)
        
        # 
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
        nn.init.constant_(self.output_proj.bias, 0.)
        
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if value is None:
            value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        #assert self.num_tpv_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(self.num_tpv_queue*bs,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_tpv_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_tpv_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_tpv_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(3, 0, 1, 2, 4, 5)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_tpv_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        

        # reference_points: (bs, num_query, num_levels, 2/4)
        # expand to match offsets batch: (bs*num_tpv_queue, num_query, num_levels, 2/4)
        if reference_points.size(0) == bs and self.num_tpv_queue > 1:
            reference_points = reference_points.repeat(self.num_tpv_queue, 1, 1, 1)


        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        # if torch.cuda.is_available() and value.is_cuda:

        #     # using fp16 deformable attention is unstable because it performs many sum operations
        #     if value.dtype == torch.float16:
        #         MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        #     else:
        #         MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
        #     output = MultiScaleDeformableAttnFunction.apply(
        #         value, spatial_shapes, level_start_index, sampling_locations,
        #         attention_weights, self.im2col_step)
        # else:
        output = self.deformable_attention(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        # output shape (bs*num_tpv_queue, num_query, embed_dims)
        # (bs*num_tpv_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_tpv_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_tpv_queue)-> (num_query, embed_dims, bs)
        output = (output[..., :bs] + output[..., bs:])/self.num_tpv_queue

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity # （bs, num_query, embed_dims）
    
    @staticmethod
    def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
        
        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ], dim=0) # hw+zh+wz, 3, #p, 2
        
        return reference_points # hw+zh+wz, 3, #p, 2