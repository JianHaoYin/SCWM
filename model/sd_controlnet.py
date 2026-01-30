import torch
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.training_utils import EMAModel
from tpvformer import TPVFormer  # 从TPVFormer仓库导入核心模型
from tpvformer.config import cfg  # TPVFormer配置文件

class TPVControlNet(nn.Module):
    def __init__(self, sd_pretrained_path, controlnet_pretrained_path, tpv_pretrained_path=None):
        super().__init__()
        # 1. 加载预训练ControlNet（替换默认输入编码器为TPV特征适配层）
        self.controlnet = ControlNetModel.from_pretrained(controlnet_pretrained_path)
        # 替换ControlNet的输入层：默认接收1/3通道图像，改为接收TPV融合特征（64通道）
        self.controlnet.controlnet_input_blocks[0] = nn.Conv2d(64, 320, kernel_size=3, padding=1)
        
        # 2. 加载预训练SD（冻结主干，只训练ControlNet和TPVFormer）
        self.sd_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_pretrained_path,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        # 冻结SD的U-Net和文本编码器（仅训练ControlNet和TPVFormer）
        for param in self.sd_pipeline.unet.parameters():
            param.requires_grad = False
        for param in self.sd_pipeline.text_encoder.parameters():
            param.requires_grad = False
        
        # 3. 加载TPVFormer（三平面特征提取）
        self.tpvformer = TPVFormer(cfg)
        if tpv_pretrained_path:
            self.tpvformer.load_state_dict(torch.load(tpv_pretrained_path)["model"])
        # TPV特征融合模块：将Top/Side/Front三平面特征（各200×200×64）融合为64通道特征图
        self.tpv_fusion = nn.Sequential(
            nn.Conv2d(64*3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((512, 512))  # 缩放到SD输入尺寸（512×512）
        )
        
        # 4. 损失函数（生成损失+TPV特征一致性损失）
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()  # 可复用VGG16提取特征计算感知损失

    def forward(self, batch):
        # 1. TPVFormer提取三平面特征
        top_plane, side_plane, front_plane = batch["tpv_inputs"]
        # TPVFormer输出：top_feat (200×200×64), side_feat (200×200×64), front_feat (200×200×64)
        top_feat, side_feat, front_feat = self.tpvformer.extract_tpv_features(top_plane, side_plane, front_plane)
        
        # 2. TPV特征融合（拼接→降维→适配ControlNet输入）
        tpv_feat = torch.cat([top_feat, side_feat, front_feat], dim=1)  # (B, 192, 200, 200)
        control_feat = self.tpv_fusion(tpv_feat)  # (B, 64, 512, 512)
        
        # 3. SD+ControlNet生成下一帧图像
        prompt = "drone view of terrain, buildings, consistent with 3D structure"  # 固定提示词（可根据场景修改）
        generated_img = self.sd_pipeline(
            image=batch["sd_input"],  # 无人机当前帧（img2img输入）
            control_image=control_feat,  # TPV融合特征（ControlNet约束）
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            output_type="pt"
        ).images
        
        # 4. 计算损失
        l2_loss = self.l2_loss(generated_img, batch["target"])
        perceptual_loss = self.perceptual_loss(generated_img, batch["target"])
        # TPV特征一致性损失（可选：确保生成图像的3D特征与TPV特征匹配）
        tpv_consistency_loss = self.calc_tpv_consistency(generated_img, control_feat)
        
        total_loss = l2_loss + 0.1 * perceptual_loss + 0.05 * tpv_consistency_loss
        return {
            "generated_img": generated_img,
            "total_loss": total_loss,
            "l2_loss": l2_loss,
            "perceptual_loss": perceptual_loss
        }

    def calc_tpv_consistency(self, generated_img, control_feat):
        """计算生成图像与TPV特征的一致性损失（简化：用CNN提取生成图特征与control_feat对比）"""
        feat_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((512, 512))
        ).to(generated_img.device)
        generated_feat = feat_extractor(generated_img)
        return self.l2_loss(generated_feat, control_feat)