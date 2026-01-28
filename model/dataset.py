import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class DroneDataset(Dataset):
    def __init__(self, data_root, img_size=512):
        self.data_root = data_root
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.tpv_transform = transforms.Compose([  # TPVFormer输入尺寸（论文默认）
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        # 1. 加载数据
        sat_img = Image.open(f"{self.data_root}/satellite/{idx}.png").convert("RGB")  # 卫星图（BEV俯瞰）
        drone_img = Image.open(f"{self.data_root}/drone/{idx}.png").convert("RGB")  # 无人机当前帧
        action = np.load(f"{self.data_root}/action/{idx}.npy")  # [轨迹坐标(x,y,z), 相机姿态(roll,pitch,yaw), 内参]
        
        # 2. 生成TPVFormer所需的三个平面输入（Top/Side/Front）
        top_plane = self.tpv_transform(sat_img)  # 卫星图作为Top平面（BEV）
        side_plane, front_plane = self.generate_side_front_plane(drone_img, action)  # 从无人机图+姿态生成侧/正视图
        
        # 3. SD+ControlNet输入（无人机当前帧作为img2img输入）
        sd_input = self.transform(drone_img)
        # 4. 标签：无人机下一帧真实图像（用于计算生成损失）
        target_img = self.transform(Image.open(f"{self.data_root}/target/{idx}.png").convert("RGB"))
        
        return {
            "tpv_inputs": (top_plane, side_plane, front_plane),
            "sd_input": sd_input,
            "action": torch.tensor(action, dtype=torch.float32),
            "target": target_img
        }

    def generate_side_front_plane(self, drone_img, action):
        """根据无人机当前帧和姿态，生成TPV所需的Side（侧视图）和Front（正视图）"""
        # 简化逻辑：通过相机姿态旋转当前帧，模拟侧/正视图（实际可用3D渲染或深度估计优化）
        pitch, yaw = action[3], action[4]  # 提取姿态角
        h, w = drone_img.size[1], drone_img.size[0]
        
        # 旋转生成正视图（yaw=0）和侧视图（yaw=90°）
        front_plane = self.rotate_image(drone_img, yaw)
        side_plane = self.rotate_image(drone_img, yaw + 90)
        
        return self.tpv_transform(front_plane), self.tpv_transform(side_plane)

    def rotate_image(self, img, angle):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        M = cv2.getRotationMatrix2D((img_cv.shape[1]//2, img_cv.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img_cv, M, (img_cv.shape[1], img_cv.shape[0]))
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    def __len__(self):
        return len(os.listdir(f"{self.data_root}/satellite"))