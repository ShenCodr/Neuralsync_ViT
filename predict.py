import os
import json
import glob

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from vit_model import vit_base_patch16_224_in21k as create_model

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义预处理操作
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 设置输入和输出文件夹路径
    input_folder = r"C:\\Users\\沈超\\Desktop\\couple\\connectivity"
    output_folder = r"C:\\Users\\沈超\Desktop\\ans\\couple_ans"  
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取类别映射文件 class_indices.json
    json_path = r"D:/VsCode/ViT_model/class_indices.json"
    assert os.path.exists(json_path), f"文件: '{json_path}' 不存在."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    # 创建模型（num_classes 与训练时一致）
    model = create_model(num_classes=2, has_logits=False).to(device)
    # 设置预训练模型权重路径
    model_weight_path = r"D:/VsCode/ViT_model/checkpoints/model-19.pth"
    assert os.path.exists(model_weight_path), f"权重文件: '{model_weight_path}' 不存在."
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()

    # 找到所有匹配模式的图片
    pattern = os.path.join(input_folder, "Couple15_HW_connectivity_seg*.jpg")
    img_paths = sorted(glob.glob(pattern), key=lambda x: int(x.split("seg")[1].split(".")[0]))
    
    # 检查是否找到图片
    if not img_paths:
        print(f"在 {input_folder} 中没有找到匹配的图片")
        return
    
    print(f"找到 {len(img_paths)} 张待预测图片")
    
    # 创建图表展示所有结果
    fig, axs = plt.subplots(4, 5, figsize=(20, 16))  # 4行5列显示20张图片
    fig.suptitle("连通性矩阵分类结果", fontsize=16)
    
    # 存储每个类别的概率
    class_probs = {class_name: [] for class_name in class_indict.values()}
    segment_nums = []
    
    # 处理每张图片
    for i, img_path in enumerate(img_paths):
        # 提取段号
        segment_num = int(os.path.basename(img_path).split("seg")[1].split(".")[0])
        segment_nums.append(segment_num)
        
        # 加载和预处理图片
        img = Image.open(img_path)
        img_tensor = data_transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        
        # 进行预测
        with torch.no_grad():
            output = torch.squeeze(model(img_tensor.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).item()
        
        # 存储每个类别的概率
        for idx, prob in enumerate(predict):
            class_probs[class_indict[str(idx)]].append(prob.item())
        
        # 计算图表位置
        row, col = i // 5, i % 5
        
        # 在相应位置显示图片
        axs[row, col].imshow(np.array(img))
        axs[row, col].set_title(f"Seg {segment_num}: {class_indict[str(predict_cla)]} ({predict[predict_cla]:.2f})")
        axs[row, col].axis('off')
        
        # 打印预测结果
        print(f"图片 {os.path.basename(img_path)}:")
        print(f"  预测类别: {class_indict[str(predict_cla)]}, 概率: {predict[predict_cla]:.3f}")
        for j in range(len(predict)):
            print(f"  类别 {class_indict[str(j)]}: {predict[j]:.3f}")
        print("")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为顶部标题留出空间
    plt.savefig(os.path.join(output_folder, "prediction_grid.png"), dpi=150)
    
    # 绘制概率变化折线图
    plt.figure(figsize=(10, 6))
    for class_name, probs in class_probs.items():
        plt.plot(segment_nums, probs, marker='o', label=class_name)
    
    plt.xlabel('时间段编号')
    plt.ylabel('预测概率')
    plt.title('各类别预测概率随时间段变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "probability_trends.png"), dpi=150)
    
    # 保存结果表格为CSV
    import csv
    with open(os.path.join(output_folder, "prediction_results.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        header = ["段号"] + list(class_probs.keys()) + ["预测类别"]
        writer.writerow(header)
        
        # 写入数据
        for i, seg_num in enumerate(segment_nums):
            row_data = [seg_num]
            for class_name in class_probs.keys():
                row_data.append(class_probs[class_name][i])
            
            # 添加预测类别
            max_prob_class = max(class_probs.keys(), key=lambda x: class_probs[x][i])
            row_data.append(max_prob_class)
            
            writer.writerow(row_data)
    
    print(f"\n所有结果已保存到: {output_folder}")
    
    # 可选: 显示图表
    plt.show()

if __name__ == '__main__':
    main()