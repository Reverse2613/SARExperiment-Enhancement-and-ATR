# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 导入手写的两个模型
from src.recognition.models.sar_yolo import SARYoloCls
from src.recognition.models.sar_vit import SARViT

def get_dataloaders(data_dir, batch_size=64):
    """
    构建数据加载器。
    把数据分成了 train, val_original, val_enhanced 三个文件夹。
    """
    # 定义预处理流水线
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 强制转为单通道(SAR的特性)
        transforms.Resize((128, 128)),               # 确保统一尺寸
        transforms.RandomHorizontalFlip(p=0.5), # 50%概率水平翻转
        transforms.RandomVerticalFlip(p=0.5),   # 50%概率垂直翻转 (SAR是俯视，垂直翻转合理)
        transforms.ToTensor(),                       # 转为 Tensor，并将像素值从 0-255 缩放到 0-1
    ])

    # 验证集【绝对不能】做数据增强，必须保持原样
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 使用 ImageFolder 自动根据文件夹名打标签 (完全去掉了 XML 的依赖)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_orig_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val_original'), transform=val_transform)
    val_enh_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val_enhanced'), transform=val_transform)

    # 构建 DataLoader (自动批次化、打乱、多线程加载)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # 验证集不需要打乱
    val_orig_loader = DataLoader(val_orig_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_enh_loader = DataLoader(val_enh_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_orig_loader, val_enh_loader, train_dataset.classes

def evaluate(model, dataloader, criterion, device):
    """
    验证函数：计算在一个数据集上的 Loss 和 准确率 (Accuracy)
    """
    model.eval() # 开启评估模式 (关闭 Dropout 等)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # 验证时不计算梯度，省显存提速度
        for images, labels in dataloader:
            # 将数据推送到 GPU/CPU
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            # 找到概率最大的类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train_model(model_name="yolo", num_epochs=20):
    """
    主训练引擎
    """
    data_dir = "data/SAR_Cls_Dataset"
    
    # ==========================================
    # 1. 硬件探测与自适应 
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 训练启动！当前使用计算硬件: {device.type.upper()}")

    # ==========================================
    # 2. 准备数据
    # ==========================================
    train_loader, val_orig_loader, val_enh_loader, class_names = get_dataloaders(data_dir, batch_size=64)
    num_classes = len(class_names)
    print(f" 数据加载完毕。共有 {num_classes} 个类别。")

    # ==========================================
    # 3. 实例化模型并推送到设备
    # ==========================================
    if model_name.lower() == "yolo":
        print(" 正在构建 SAR-YOLOv11 (基于CBAM) 网络...")
        model = SARYoloCls(num_classes=num_classes)
    elif model_name.lower() == "vit":
        print(" 正在构建 SAR-ViT (ViT) 网络...")
        model = SARViT(num_classes=num_classes)
    else:
        raise ValueError("不支持的模型名称，请选择 'yolo' 或 'vit'")
    # model.load_state_dict(torch.load("data/weights/yolo_best.pth", map_location=device),strict=True)
    
    #将整个模型架构和权重推送到 GPU/CPU
    model = model.to(device)

    # ==========================================
    # 4. 定义损失函数与优化器
    # ==========================================
    criterion = nn.CrossEntropyLoss() # 多分类任务标准 Loss
    # 使用 AdamW 优化器，带权重衰减防过拟合
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # ==========================================
    # 5. 核心训练与双轨验证循环
    # ==========================================
    print("\n" + "="*50)
    print(" 开始训练循环...")

    #  核心升级 2：追踪并保存最优权重
    best_orig_acc = 0.0
    os.makedirs('data/weights', exist_ok=True)
    best_model_path = f"data/weights/{model_name}_best.pth"

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train() # 开启训练模式
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 遍历训练集批次
        for images, labels in train_loader:
            # 同样，数据必须去到设备上
            images, labels = images.to(device), labels.to(device)
            
            # [标准 PyTorch 训练三步曲]
            optimizer.zero_grad()         # 1. 清空上一轮的旧梯度
            outputs = model(images)       # 2. 前向传播
            loss = criterion(outputs, labels) # 3. 计算 Loss
            
            loss.backward()               # 4. 反向传播求梯度
            optimizer.step()              # 5. 更新网络权重
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()  

        avg_train_loss = train_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        
        # ==========================================
        #  双轨验证：见证奇迹的时刻
        # ==========================================
        # 考卷 1：原始降质图
        orig_loss, orig_acc = evaluate(model, val_orig_loader, criterion, device)
        # 考卷 2：提质后的图
        enh_loss, enh_acc = evaluate(model, val_enh_loader, criterion, device)

        # 学习率调度
        scheduler.step(orig_acc)
        epoch_time = time.time() - start_time   

        #  核心升级 2：发现更好的模型，立刻保存！
        if orig_acc > best_orig_acc:
            best_orig_acc = orig_acc
            torch.save(model.state_dict(), best_model_path)

        # 打印本轮汇报
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] ({epoch_time:.1f}s)"
              f"| Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.1f}%)"
              f"| Val(原图) 准确率: {orig_acc:.2f}% "
              f"| Val(提质图) 准确率: {enh_acc:.2f}% "
              f"| 提质增益: {(enh_acc - orig_acc):+.2f}%"
              )

    print("="*50)
    print(f" 训练完成！历史最高 Val_Orig 准确率: {best_orig_acc:.2f}%")
    print(f"最终最佳权重已保存在: {best_model_path}")    

if __name__ == "__main__":
    # 在这里切换训练哪个模型！
    
    # 第一步：先训练 SAR-YOLO
    #train_model(model_name="yolo", num_epochs=50)
    
    # 当你跑完 YOLO 之后，可以把上面那行注释掉，跑 ViT：
    train_model(model_name="vit", num_epochs=50)