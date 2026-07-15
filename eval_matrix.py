# eval_matrix.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 导入模型
from src.recognition.models.sar_yolo import SARYoloCls
from src.recognition.models.sar_vit import SARViT

def get_test_loader(data_dir):
    """只加载原始验证集用于测试"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val_enhanced'), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return loader, dataset.classes

def draw_confusion_matrix(model, loader, classes, model_name, device):
    """生成并绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f" 正在使用 {model_name} 权重进行推理...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 收集所有的预测值和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 将其转换为百分比 (归一化)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 使用 seaborn 画一张非常学术的高清热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name.upper()} (Val Enhanced)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图片
    save_path = f"Confusion_Matrix_{model_name}_enhanced.png"
    plt.savefig(save_path, dpi=300)
    print(f" {model_name} 的混淆矩阵已生成: {save_path}")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/SAR_Cls_Dataset"
    
    # 获取测试数据和类别名字
    test_loader, class_names = get_test_loader(data_dir)
    num_classes = len(class_names)
    
    # ==========================
    # 评估 YOLO
    # ==========================
    yolo_weight_path = "data/weights/yolo_best.pth"
    if os.path.exists(yolo_weight_path):
        model_yolo = SARYoloCls(num_classes=num_classes).to(device)
        # 加载最佳权重
        model_yolo.load_state_dict(torch.load(yolo_weight_path, map_location=device))
        draw_confusion_matrix(model_yolo, test_loader, class_names, "yolo", device)
    else:
        print(f"找不到权重文件 {yolo_weight_path}")

    # ==========================
    # 评估 ViT
    # ==========================
    vit_weight_path = "data/weights/vit_best.pth"
    if os.path.exists(vit_weight_path):
        model_vit = SARViT(num_classes=num_classes).to(device)
        # 加载最佳权重
        model_vit.load_state_dict(torch.load(vit_weight_path, map_location=device))
        draw_confusion_matrix(model_vit, test_loader, class_names, "vit", device)
    else:
        print(f"找不到权重文件 {vit_weight_path}")