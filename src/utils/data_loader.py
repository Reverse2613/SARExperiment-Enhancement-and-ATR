# src/utils/data_loader.py
import cv2
import os
import numpy as np

class DataLoader:
    """
    数据加载器：负责安全地读取和保存SAR图像。
    为了防止路径错误或格式不支持，加入了安全检查机制。
    """
    
    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        读取单张SAR图像（以灰度图模式读取）
        
        参数:
            file_path: 图像的绝对或相对路径
            
        返回:
            image: numpy二维数组格式的灰度图像
        """
        # 安全性检查 1：路径是否存在？
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"【错误】找不到图像文件，请检查路径: {file_path}")
            
        # 使用 OpenCV 读取，强制转换为灰度图 (SAR图像本身就是单通道强度图)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # 安全性检查 2：文件是否损坏或不是图像？
        if image is None:
            raise ValueError(f"【错误】无法解析图像文件，可能已损坏: {file_path}")
            
        return image

    @staticmethod
    def save_image(image: np.ndarray, save_path: str):
        """
        将处理后的图像保存到本地
        """
        # 安全性检查：如果保存目录不存在，就自动创建它
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        success = cv2.imwrite(save_path, image)
        if not success:
            raise IOError(f"【错误】图像保存失败，请检查写入权限或路径: {save_path}")
        print(f"【成功】图像已保存至: {save_path}")