# src/utils/data_selector.py
import os
import shutil
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_xml_for_subclass(xml_path):
    """
    解析单个 XML 文件，提取 <subclass> 标签的值。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 寻找 object 节点下的 subclass 节点
        subclass_node = root.find('.//object/subclass')
        if subclass_node is not None:
            return subclass_node.text.strip()
    except Exception as e:
        print(f"解析 XML 失败 {xml_path}: {e}")
    return None

def normalize_name(name: str) -> str:
    """
    字符串归一化。
    将名字全部转为小写，并剔除所有空格和下划线。
    例如："Medium_Bus" -> "mediumbus", "shovel _loader" -> "shovelloader"
    """
    return name.lower().replace(" ", "").replace("_", "")


def build_dataset(raw_train_dir, low_quality_dir, enhanced_dir, output_dir):
    """
    核心清洗逻辑：映射、抽样并构建最终数据集
    """
    print("="*50)
    print(" 开始构建实验二标准数据集...")
    
    # 定义输出子目录
    out_train = os.path.join(output_dir, 'train')
    out_val_orig = os.path.join(output_dir, 'val_original')
    out_val_enh = os.path.join(output_dir, 'val_enhanced')
    
    # 1. 扫描测试集 (ATR_low_quality)，确定我们需要哪 10 个子类，以及它们的数量
    target_classes = [d for d in os.listdir(low_quality_dir) if os.path.isdir(os.path.join(low_quality_dir, d))]
    test_counts = {}
    # norm_to_orig 负责把归一化后的名字映射回你原本带下划线的文件夹名
    # 例如：{'mediumbus': 'medium_bus', 'shovelloader': 'shovel _loader'}
    norm_to_orig = {}

    for cls in target_classes:
        num_files = len([f for f in os.listdir(os.path.join(low_quality_dir, cls)) if f.endswith(('.tif', '.png', '.jpg'))])
        test_counts[cls] = num_files
        norm_name = normalize_name(cls)
        norm_to_orig[norm_name] = cls
               
    print(f" 发现目标类别数: {len(target_classes)}")
    for cls, count in test_counts.items():
        print(f"   - {cls}: {count} 张测试图")

    # 2. 扫描官方 train 文件夹，建立 图像 -> subclass 的映射候选池
    # 结构: {'Forklift': ['path/to/img1.tif', 'path/to/img2.tif'], ...}
    train_candidates = defaultdict(list)
    
    print("\n 正在深度扫描官方 train 文件夹的 XML 标注...")
    for root_dir, _, files in os.walk(raw_train_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root_dir, file)
                tif_path = xml_path.replace('.xml', '.tif') # 因为对应的图片文件与标签文件同名
                
                if not os.path.exists(tif_path):
                    continue
                    
                subclass_name = parse_xml_for_subclass(xml_path)
                
                # 如果这个 XML 里写的类别，正好是我们需要的那 10 个类别之一，则加入候选
                if subclass_name:
                    # 将 XML 读出来的标签也归一化
                    norm_subclass = normalize_name(subclass_name)
                    # 如果在我们的映射字典里找到了，说明是目标类别
                    if norm_subclass in norm_to_orig:
                        # 获取原始的文件夹名
                        orig_cls_name = norm_to_orig[norm_subclass]
                        train_candidates[orig_cls_name].append(tif_path)

    # 3. 抽样与复制 (执行 训练样本量 <= 2 * 测试样本量 的规则)
    print("\n 开始抽样并复制文件...")
    for cls in target_classes:
        candidates = train_candidates.get(cls, [])
        max_allowed = 2 * test_counts[cls]
        
        # 随机抽样
        if len(candidates) > max_allowed:
            selected_train = random.sample(candidates, max_allowed)
        else:
            selected_train = candidates
            
        print(f"   => [{cls}]: 官方找到 {len(candidates)} 张，最终抽取 {len(selected_train)} 张 (上限 {max_allowed})")
        
        # 创建分类文件夹
        os.makedirs(os.path.join(out_train, cls), exist_ok=True)
        os.makedirs(os.path.join(out_val_orig, cls), exist_ok=True)
        os.makedirs(os.path.join(out_val_enh, cls), exist_ok=True)
        
        # 复制 Train
        for src_tif in selected_train:
            dst_tif = os.path.join(out_train, cls, os.path.basename(src_tif))
            shutil.copy(src_tif, dst_tif)
            
        # 复制 Val Original
        src_orig_dir = os.path.join(low_quality_dir, cls)
        for f in os.listdir(src_orig_dir):
            if f.endswith(('.tif', '.png', '.jpg')):
                shutil.copy(os.path.join(src_orig_dir, f), os.path.join(out_val_orig, cls, f))
                
        # 复制 Val Enhanced
        src_enh_dir = os.path.join(enhanced_dir, cls)
        for f in os.listdir(src_enh_dir):
            if f.endswith(('.tif', '.png', '.jpg')):
                shutil.copy(os.path.join(src_enh_dir, f), os.path.join(out_val_enh, cls, f))

    print("\n 数据集重构完成！全部保存在:", output_dir)
    print("="*50)

if __name__ == "__main__":
    # 配置路径（确保相对路径与目录一致）
    RAW_TRAIN_DIR = "train"                        # 解压后的 50 个文件夹的官方数据集
    LOW_QUALITY_DIR = "data/ATR_low_quality"       # 实验一的原图
    ENHANCED_DIR = "data/ATR_enhanced"             # 实验一提质后的图
    OUTPUT_DIR = "data/SAR_Cls_Dataset"            # 洗好的数据集输出地
    
    # 设置随机种子，保证每次抽样结果一致，方便实验可复现
    random.seed(2026) 
    
    build_dataset(RAW_TRAIN_DIR, LOW_QUALITY_DIR, ENHANCED_DIR, OUTPUT_DIR)