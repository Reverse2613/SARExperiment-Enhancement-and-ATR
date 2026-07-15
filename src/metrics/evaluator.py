# src/metrics/evaluator.py
import numpy as np
import cv2
from scipy.stats import entropy

class MetricsEvaluator:
    """
    客观评估指标计算器：计算ENL、EPI和信息熵。
    全部采用静态方法，方便随时调用。
    """
    
    @staticmethod
    def calculate_enl(image: np.ndarray, bg_roi: tuple = None) -> float:
        """
        计算等效视数 (Equivalent Number of Looks, ENL)
        原理：ENL = (均值^2) / 方差。ENL越大，相干斑抑制越好。
        
        参数:
            image: 传入的图像
            bg_roi: 背景区域坐标 (x_min, y_min, x_max, y_max)。
                    如果不传，默认计算整图（不推荐，因为目标会影响方差）。
        """
        img_float = image.astype(np.float64)
        
        # 裁剪出背景区域
        if bg_roi is not None:
            xmin, ymin, xmax, ymax = bg_roi
            region = img_float[ymin:ymax, xmin:xmax]
        else:
            region = img_float
            
        mean_val = np.mean(region)
        var_val = np.var(region)
        
        # 安全性检查：防止除以 0 的情况导致程序崩溃
        if var_val == 0:
            return float('inf') # 如果方差为0（纯色图），ENL无穷大
            
        enl = (mean_val ** 2) / var_val
        return enl

    @staticmethod
    def calculate_epi(img_original: np.ndarray, img_filtered: np.ndarray) -> float:
        """
        计算边缘保持指数 (Edge Preservation Index, EPI)
        原理：比较滤波前后图像的高频信息（使用拉普拉斯算子提取边缘）。
        EPI越接近1，说明边缘保持得越好。
        """
        # 转换为浮点数防止溢出
        img_orig_f = img_original.astype(np.float64)
        img_filt_f = img_filtered.astype(np.float64)
        
        # 使用拉普拉斯算子计算边缘图
        laplacian_orig = cv2.Laplacian(img_orig_f, cv2.CV_64F)
        laplacian_filt = cv2.Laplacian(img_filt_f, cv2.CV_64F)
        
        # 计算高频信息的绝对值总和
        sum_orig = np.sum(np.abs(laplacian_orig))
        sum_filt = np.sum(np.abs(laplacian_filt))
        
        if sum_orig == 0:
            return 1.0 # 如果原图没有边缘，保持率设为1
            
        epi = sum_filt / sum_orig
        return epi

    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        计算一维信息熵 (Information Entropy)
        原理：统计每个灰度级(0-255)出现的概率，计算香农熵。
        """
        # 统计灰度直方图
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        # 将频次转换为概率分布
        prob = hist.ravel() / hist.sum()
        # 过滤掉概率为0的项，避免 log(0) 报错
        prob = prob[prob > 0]
        # 计算香农熵 (以2为底)
        ent = entropy(prob, base=2)
        return ent