# src/enhancement/enhancer.py
import cv2
import numpy as np

class ImageEnhancer:
    """
    图像增强类。
    用于提升去噪后 SAR 图像的目标与背景对比度。
    """
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
        应用 CLAHE (限制对比度自适应直方图均衡化)
        
        参数:
            image: 输入的灰度图像
            clip_limit: 限制对比度的阈值。值越大，对比度越强，但也越容易放大残余噪声。
                        安全建议：SAR 图像一般设置在 2.0 到 3.0 之间。
            tile_grid_size: 局部网格划分大小。默认 8x8 是通用标准。
            
        返回:
            对比度增强后的图像
        """
        # 安全检查：确保输入是单通道灰度图
        if len(image.shape) != 2:
            raise ValueError("【错误】CLAHE 只能处理单通道灰度图像！")
            
        # OpenCV 已经内置了高度优化的 CLAHE 算法，直接实例化并应用
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_img = clahe_obj.apply(image)
        
        return enhanced_img