# main_enhance.py
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 导入模块化写好的组件
from src.utils.data_loader import DataLoader
from src.enhancement.filters import LeeFilter
from src.enhancement.enhancer import ImageEnhancer
from src.metrics.evaluator import MetricsEvaluator

def process_single_image(img_path: str, save_dir: str):
    """
    处理单张图像的完整流水线,并返回三个指标以便统计。
    """

    # 1. 加载数据
    original_img = DataLoader.load_image(img_path)

    # 2. 核心算法处理流水线
    filtered_img = LeeFilter.apply(original_img, window_size=5)
    final_img = ImageEnhancer.apply_clahe(filtered_img, clip_limit=2.5)

    # 3. 客观指标评估 (假定左上角 20x20 像素区域为纯背景，用于计算 ENL)
    bg_roi = (0, 0, 20, 20) 
    
    enl_orig = MetricsEvaluator.calculate_enl(original_img, bg_roi)
    ent_orig = MetricsEvaluator.calculate_entropy(original_img)
    enl_final = MetricsEvaluator.calculate_enl(final_img, bg_roi)
    ent_final = MetricsEvaluator.calculate_entropy(final_img)
    epi = MetricsEvaluator.calculate_epi(original_img, filtered_img)

    # 4. 保存结果    
    img_name = os.path.basename(img_path)
    # 为了避免终端输出太长，我们可以把打印精简一下
    # print(f"正在处理: {img_name}...")
    save_path = os.path.join(save_dir, img_name) # 去掉前缀，保持原名，方便后续读取
    DataLoader.save_image(final_img, save_path)

    return enl_orig, enl_final, ent_orig, ent_final, epi


def plot_category_metrics(metrics_dict):
    """
    [可视化出图]：根据统计字典生成各类别指标对比柱状图
    """
    categories = list(metrics_dict.keys())
    x = np.arange(len(categories))
    width = 0.35  # 柱子的宽度

    # 提取平均数据
    enl_orig = [np.mean(metrics_dict[c]['enl_orig']) for c in categories]
    enl_final = [np.mean(metrics_dict[c]['enl_final']) for c in categories]
    ent_orig = [np.mean(metrics_dict[c]['ent_orig']) for c in categories]
    ent_final = [np.mean(metrics_dict[c]['ent_final']) for c in categories]
    epi = [np.mean(metrics_dict[c]['epi']) for c in categories]

    # 创建一个 1行3列 的大画布
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # --- 1. ENL (等效视数) 对比图 ---
    axs[0].bar(x - width/2, enl_orig, width, label='Original')
    axs[0].bar(x + width/2, enl_final, width, label='Enhanced')
    axs[0].set_title('ENL Comparison per Category (Higher is Better)')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(categories, rotation=45, ha='right')
    axs[0].legend()

    # --- 2. Entropy (信息熵) 对比图 ---
    axs[1].bar(x - width/2, ent_orig, width, label='Original')
    axs[1].bar(x + width/2, ent_final, width, label='Enhanced')
    axs[1].set_title('Entropy Comparison per Category')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(categories, rotation=45, ha='right')
    axs[1].legend()

    # --- 3. EPI (边缘保持指数) 柱状图 ---
    axs[2].bar(x, epi, width, color='green', alpha=0.7)
    axs[2].set_title('Edge Preservation Index (EPI) per Category')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(categories, rotation=45, ha='right')
    # EPI 理想值是 1，画一条虚线作为参考
    axs[2].axhline(y=1.0, color='r', linestyle='--', label='Ideal EPI (1.0)')
    axs[2].legend()

    plt.tight_layout()
    # 自动保存图表到根目录，方便你写进报告
    plt.savefig("Experiment1_Metrics_Report.png", dpi=300)
    plt.show()


def visualize_results(img1, img2, img3, img_title):
    """
    画图工具：将处理前后的三张图并排显示
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('1. Original (Degraded)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('2. Lee Filtered (Denoised)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img3, cmap='gray')
    plt.title('3. Final (CLAHE Enhanced)')
    plt.axis('off')

    plt.suptitle(f"Processing Result: {img_title}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- 工程的入口配置区 ---
    
    input_folder = "data/ATR_low_quality"
    output_folder = "data/ATR_enhanced"
    
    # [数据统计]：用于分类别保存指标
    # 数据结构长这样: {'Forklift': {'enl_orig': [...], 'enl_final': [...]}, ...}
    category_metrics = defaultdict(lambda: defaultdict(list))

    if not os.path.exists(input_folder):
        print(f"【错误】找不到输入文件夹: {input_folder}")
    else:
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        processed_count = 0
        
        print("开始扫描文件夹，准备执行批量图像提质...")
        
    # [核心更新]: 使用 os.walk 递归遍历所有子文件夹
    # root: 当前正在遍历的文件夹路径 (例如: data/ATR_low_quality/Forklift)
    # dirs: 当前文件夹里的子文件夹名列表
    # files: 当前文件夹里的文件名列表
    for root, dirs, files in os.walk(input_folder):
        # 提取当前所在的类别名称 (即文件夹名，例如 "Forklift")
        category_name = os.path.basename(root)

        for file_name in files:
            # 检查文件后缀是不是图片
            if file_name.lower().endswith(valid_extensions):
                # 1. 拼出原始图像的完整绝对/相对路径
                file_path = os.path.join(root, file_name)
                    
                # 2. 计算相对路径，以保持输出文件夹的目录结构一致 (镜像重建)
                # 例如 root 是 "data/ATR_low_quality/Forklift"，相对路径就是 "Forklift"
                relative_path = os.path.relpath(root, input_folder)
                    
                # 3. 拼出对应的保存目录并创建
                # 保存目录变成: data/ATR_enhanced/Forklift
                save_dir = os.path.join(output_folder, relative_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                        
                # 4. 调用处理函数
                # 只在处理第一张图的时候弹窗和打印详细报告 (processed_count == 0)
                # show_figure = (processed_count == 0) 
                    
                # 进度提示
                if processed_count % 50 == 0:
                    print(f"已处理 {processed_count} 张图像...")
                        
                # 处理图像并拿到返回的指标
                e_orig, e_final, ent_o, ent_f, epi_val = process_single_image(file_path, save_dir)
                # 记录到当前类别的字典中
                category_metrics[category_name]['enl_orig'].append(e_orig)
                category_metrics[category_name]['enl_final'].append(e_final)
                category_metrics[category_name]['ent_orig'].append(ent_o)
                category_metrics[category_name]['ent_final'].append(ent_f)
                category_metrics[category_name]['epi'].append(epi_val)
                processed_count += 1
                    
    if processed_count > 0:
        print(f"\n【任务完成】共成功提质 {processed_count} 张图像！")
        print("正在生成各类别指标对比图表...")
        plot_category_metrics(category_metrics)
        print("图表已保存为 Experiment1_Metrics_Report.png，可直接用于实验报告！")