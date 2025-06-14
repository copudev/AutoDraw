from collections import deque

import cv2
import numpy as np


class StrokeBuilder:
    def __init__(self):
        self.stroke = []
        self.tmp = []

    def commit(self):
        if len(self.tmp) != 0:
            self.stroke.append(self.tmp)
            self.tmp = []

    def add(self, point):
        self.tmp.append(point)

    def build(self):
        return self.stroke

    def from_image(self, img, preprocess=True):
        """无回溯的深度优先"""
        if preprocess:
            # 高斯滤波 + Candy 算子
            img_bin = line_draft(img)
        else:
            img_bin = img
        # cv2.imwrite("result.jpg", img_bin)
        # exit()
        area_to_scan = []
        for x in range(img_bin.shape[0]):
            for y in range(img_bin.shape[1]):
                if img_bin[x][y] == 0:
                    img_bin[x][y] = 255
                    self.add([y, x])
                    area_to_scan = area8(x, y, img_bin.shape)

                while len(area_to_scan) != 0:
                    ix, iy = area_to_scan.pop(0)
                    if img_bin[ix][iy] == 0:
                        img_bin[ix][iy] = 255
                        self.add([iy, ix])
                        for jx, jy in area_to_scan:
                            if img_bin[jx][jy] == 0:
                                img_bin[jx][jy] = 255
                        area_to_scan = area8(ix, iy, img_bin.shape)
                self.commit()
        return self.build()

    def from_image_df(self, img, preprocess=True):
        """有回溯的深度优先"""
        if preprocess:
            # 高斯滤波 + Candy 算子
            img_bin = line_draft(img)
        else:
            img_bin = img
        # cv2.imwrite("result.jpg", img_bin)
        # exit()
        area_to_scan = deque()
        for x in range(img_bin.shape[0]):
            for y in range(img_bin.shape[1]):
                if img_bin[x][y] == 0:
                    img_bin[x][y] = 255
                    self.add([y, x])
                    [area_to_scan.appendleft(i) for i in area8(x, y, img_bin.shape)]

                while len(area_to_scan) != 0:
                    ix, iy = area_to_scan.popleft()
                    if img_bin[ix][iy] == 0:
                        img_bin[ix][iy] = 255
                        self.add([iy, ix])
                        [area_to_scan.appendleft(i) for i in area8(ix, iy, img_bin.shape)]
                self.commit()
        return self.build()

    def from_image_m(self, img, preprocess=True, threshold1=150, threshold2=200):
        """m通路搜索"""
        if preprocess:
            # 高斯滤波 + Candy 算子
            img_bin = line_draft(img, threshold1, threshold2)
        else:
            img_bin = img
        cv2.imwrite("result.jpg", img_bin)
        # exit()
        area_to_extend = []
        for x in range(img_bin.shape[0]):
            for y in range(img_bin.shape[1]):
                if img_bin[x][y] == 0:
                    img_bin[x][y] = 255
                    self.add([y, x])

                    for ix, iy in area4(x, y, img_bin.shape):
                        if img_bin[ix][iy] == 0:
                            area_to_extend.append([ix, iy])

                    if len(area_to_extend) == 0:
                        for ix, iy in area8(x, y, img_bin.shape):
                            if img_bin[ix][iy] == 0:
                                area_to_extend.append([ix, iy])

                while len(area_to_extend) != 0:
                    ix, iy = area_to_extend.pop(0)
                    area_to_extend = []
                    img_bin[ix][iy] = 255
                    self.add([iy, ix])
                    for jx, jy in area4(ix, iy, img_bin.shape):
                        if img_bin[jx][jy] == 0:
                            area_to_extend.append([jx, jy])

                    if len(area_to_extend) == 0:
                        for jx, jy in area8(ix, iy, img_bin.shape):
                            if img_bin[jx][jy] == 0:
                                area_to_extend.append([jx, jy])

                self.commit()
        return self.build()


# Legacy
def image_linearize(img, threshold=200, size=(3, 3), is_gray=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if not is_gray else img
    img_invert = 255 - img_gray

    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    min_image = cv2.erode(img_invert, kernel)

    linear_reduction = img_gray + min_image
    img_bin = np.where(linear_reduction > threshold, 255, 0).astype("uint8")
    return img_bin


def line_draft(img, threshold1=30, threshold2=100, blur_kernel=5, use_adaptive=False):
    """改进的线稿处理函数"""
    if img is None:
        raise ValueError("输入图像为空")
    
    if len(img.shape) == 3:
        # 如果是彩色图像，先转换为灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    if img_gray.size == 0:
        raise ValueError("图像尺寸为空")
    
    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)
    
    # 自适应高斯模糊
    im_deno = cv2.GaussianBlur(img_enhanced, (blur_kernel, blur_kernel), 0)
    
    if use_adaptive:
        # 使用自适应阈值预处理
        adaptive_thresh = cv2.adaptiveThreshold(im_deno, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # 组合Canny边缘检测
        img_edge = cv2.Canny(adaptive_thresh, threshold1, threshold2)
    else:
        # 标准Canny边缘检测
        img_edge = cv2.Canny(im_deno, threshold1, threshold2)
    
    # 形态学操作来连接断开的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    
    # 反转颜色，让边缘为黑色
    img_edge = 255 - img_edge
    
    return img_edge


def area8(px, py, im_shape):
    area = [(px - 1, py - 1), (px, py - 1), (px + 1, py - 1),
            (px - 1, py), (px + 1, py),
            (px - 1, py + 1), (px, py + 1), (px + 1, py + 1)]
    i = 0
    while i < len(area):
        p = area[i]
        if p[0] < 0 or p[0] >= im_shape[0] or p[1] < 0 or p[1] >= im_shape[1]:
            area.pop(i)
            i -= 1
        i += 1
    return area


def area4(px, py, im_shape):
    area = [(px, py - 1),
            (px - 1, py), (px + 1, py),
            (px, py + 1)]
    i = 0
    while i < len(area):
        p = area[i]
        if p[0] < 0 or p[0] >= im_shape[0] or p[1] < 0 or p[1] >= im_shape[1]:
            area.pop(i)
            i -= 1
        i += 1
    return area


def pil2cv(img):
    """将PIL图像转换为OpenCV格式"""
    if img is None:
        print("错误: 输入图像为 None")
        return None
    
    # 确保图像是PIL Image对象
    if not hasattr(img, 'mode'):
        print("错误: 输入不是有效的PIL图像对象")
        return None
    
    try:
        print(f"图像模式: {img.mode}, 尺寸: {img.size}")
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 检查数组是否为空
        if img_array.size == 0:
            print("错误: 图像数组为空")
            return None
        
        print(f"数组形状: {img_array.shape}, 数据类型: {img_array.dtype}")
        
        # 根据图像模式进行相应转换
        if img.mode == 'RGB':
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img.mode == 'RGBA':
            # 处理透明通道，转换为RGB后再转BGR
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img.mode == 'L':  # 灰度图
            return img_array
        elif img.mode == 'P':  # 调色板模式
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # 对于其他模式，先转换为RGB
            print(f"不支持的图像模式 {img.mode}，尝试转换为RGB")
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"图像转换错误: {e}")
        return None