def infer(model, init_drone_img, sat_img, action_sequence, device):
    """
    输入：无人机初始帧 + 卫星图 + 动作序列（多帧轨迹+姿态）
    输出：多帧无人机视角生成图像
    """
    model.eval()
    generated_frames = [init_drone_img]
    current_drone_img = init_drone_img  # 初始帧作为当前帧
    
    for action in action_sequence:
        # 数据预处理
        tpv_inputs = model.preprocess_infer(sat_img, current_drone_img, action)  # 复用训练时的预处理逻辑
        sd_input = model.transform(current_drone_img).unsqueeze(0).to(device)
        
        # 前向推理
        with torch.no_grad():
            batch = {
                "tpv_inputs": tpv_inputs,
                "sd_input": sd_input,
                "action": torch.tensor(action).unsqueeze(0).to(device)
            }
            outputs = model(batch)
            next_frame = outputs["generated_img"][0].permute(1, 2, 0).cpu().numpy()
            next_frame = (next_frame + 1) / 2 * 255  # 反归一化
            next_frame = next_frame.astype(np.uint8)
        
        generated_frames.append(next_frame)
        current_drone_img = Image.fromarray(next_frame)  # 下一帧作为当前帧继续生成
    
    # 保存多帧图像
    for i, frame in enumerate(generated_frames):
        Image.fromarray(frame).save(f"./generated_frame_{i}.png")
    return generated_frames

# 调用示例
if __name__ == "__main__":
    # 加载训练好的模型
    model = TPVControlNet(
        sd_pretrained_path=sd_pretrained_path,
        controlnet_pretrained_path=controlnet_pretrained_path,
        tpv_pretrained_path="./best_sd_tpv_controlnet.pth"
    ).to(device)
    model.eval()
    
    # 输入数据
    init_drone_img = Image.open("./init_drone_img.png").convert("RGB")
    sat_img = Image.open("./satellite_img.png").convert("RGB")
    action_sequence = np.load("./action_sequence.npy")  # 多帧动作序列（N×(3+3+4)）
    
    # 生成多帧图像
    generated_frames = infer(model, init_drone_img, sat_img, action_sequence, device)