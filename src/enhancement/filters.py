# src/enhancement/filters.py
import numpy as np
from scipy.ndimage import uniform_filter

class LeeFilter:
    """
    Lee 滤波器实现类。
    用于抑制 SAR 图像的相干斑噪声，同时保持边缘。
    """
    
    @staticmethod
    def apply(image: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        对图像应用 Lee 滤波。
        
        参数:
            image: 输入的灰度图像 (numpy array)
            window_size: 滑动窗口的大小。通常取 3, 5, 7。窗口越大越平滑，但细节损失越多。默认 5 适合 SAR 车辆。
            
        返回:
            去噪后的图像 (uint8 类型)
        """
        # 安全转换：将 uint8 (0-255) 转为浮点数，防止计算过程中数值溢出
        img_float = image.astype(np.float64)
        
        # 1. 计算局部均值
        # uniform_filter 是 scipy 提供的高度优化的均值滤波函数，速度极快
        local_mean = uniform_filter(img_float, window_size)
        
        # 2. 计算局部方差
        # 根据数学公式：方差 = 平方的均值 - 均值的平方  ( D(X) = E(X^2) - [E(X)]^2 )
        local_sqr_mean = uniform_filter(img_float ** 2, window_size)
        local_var = local_sqr_mean - local_mean ** 2
        
        # 3. 估计全局噪声方差
        # 在标准的简化 Lee 滤波中，通常用图像局部方差的均值或最小值来近似全局噪声方差。
        # 这里为了稳健，取局部方差的均值
        noise_var = np.mean(local_var)
        
        # 4. 计算权重系数 K
        # 加上 1e-8 是为了安全机制，防止出现分母为 0 导致程序崩溃报错 (Divide by zero)
        K = local_var / (local_var + noise_var + 1e-8)
        
        # 5. 套用核心公式：输出 = 局部均值 + K * (当前像素 - 局部均值)
        filtered_img = local_mean + K * (img_float - local_mean)
        
        # 6. 安全恢复：限制数值在 0-255 范围内，并转回图像格式
        filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
        
        return filtered_img