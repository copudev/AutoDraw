import cv2
import numpy as np

def extract_contours(img):
    """从二值化图像中提取轮廓并转换为绘制路径"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img.max() > 1:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
    
    stroke_paths = []
    
    for contour in contours:
        # 降低面积阈值，保留更多小细节
        if cv2.contourArea(contour) < 5:
            continue
        
        # 将轮廓点转换为路径格式
        path = []
        for point in contour:
            x, y = point[0]
            path.append([int(x), int(y)])
        
        # 降低点数要求，保留更多细节路径
        if len(path) > 1:
            stroke_paths.append(path)
    
    # 按轮廓长度排序，优先绘制长线条
    stroke_paths.sort(key=lambda path: len(path), reverse=True)
    
    return stroke_paths

def image2stroke(img, preprocess=True, threshold1=50, threshold2=100):
    try:
        # 如果是彩色图像，转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 预处理：使用更温和的模糊
        if preprocess:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # 使用更小的形态学核，避免连接太多线条
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 提取轮廓
        stroke_paths = extract_contours(edges)
        
        print(f"检测到 {len(stroke_paths)} 条线条路径")
        
        return stroke_paths
        
    except Exception as e:
        print(f"图像处理出错: {e}")
        return []

def adaptive_smooth_stroke(stroke, base_tolerance=2.0, min_distance=3):
    if not stroke or len(stroke) == 0:
        return stroke
    
    smoothed_stroke = []
    
    for line in stroke:
        if len(line) < 3:
            smoothed_stroke.append(line)
            continue
        
        # 计算线条的复杂度
        line_length = calculate_line_length(line)
        curvature = calculate_curvature(line)
        
        # 根据线条特征调整平滑参数
        if line_length < 50:  # 短线条（可能是细节）
            tolerance = base_tolerance * 0.5  # 更保守的平滑
            distance = max(1, min_distance - 1)
        elif curvature > 0.1:  # 高曲率线条（可能是重要轮廓）
            tolerance = base_tolerance * 0.7
            distance = min_distance
        else:  # 长直线条
            tolerance = base_tolerance * 1.2
            distance = min_distance + 1
        
        # 应用自适应平滑
        smoothed_line = smart_smooth_line(line, tolerance, distance)
        smoothed_stroke.append(smoothed_line)
    
    return smoothed_stroke

def calculate_line_length(line):
    """计算线条总长度"""
    total_length = 0
    for i in range(1, len(line)):
        dx = line[i][0] - line[i-1][0]
        dy = line[i][1] - line[i-1][1]
        total_length += np.sqrt(dx*dx + dy*dy)
    return total_length

def calculate_curvature(line):
    """计算线条的平均曲率"""
    if len(line) < 3:
        return 0
    
    curvatures = []
    for i in range(1, len(line) - 1):
        p1 = np.array(line[i-1])
        p2 = np.array(line[i])
        p3 = np.array(line[i+1])
        
        # 计算角度变化
        v1 = p2 - p1
        v2 = p3 - p2
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0

def smart_smooth_line(line, tolerance, min_distance):
    """智能平滑单条线条"""
    if len(line) < 3:
        return line
    
    # 1. 移除距离太近的点，但保留关键点
    filtered_points = [line[0]]  # 保留第一个点
    
    for i in range(1, len(line)):
        dist = np.sqrt((line[i][0] - filtered_points[-1][0])**2 + 
                      (line[i][1] - filtered_points[-1][1])**2)
        
        # 检查是否为关键点（高曲率点）
        is_key_point = False
        if i > 0 and i < len(line) - 1:
            curvature = calculate_point_curvature(line[i-1], line[i], line[i+1])
            if curvature > 0.5:  # 高曲率阈值
                is_key_point = True
        
        if dist >= min_distance or is_key_point:
            filtered_points.append(line[i])
    
    # 确保保留最后一个点
    if len(filtered_points) > 1 and filtered_points[-1] != line[-1]:
        filtered_points.append(line[-1])
    
    # 2. 使用Douglas-Peucker算法，但保护关键点
    if len(filtered_points) > 2:
        simplified = protected_douglas_peucker(filtered_points, tolerance)
    else:
        simplified = filtered_points
    
    return simplified

def calculate_point_curvature(p1, p2, p3):
    """计算单点的曲率"""
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    v1 = p2 - p1
    v2 = p3 - p2
    
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)

def protected_douglas_peucker(points, tolerance):
    """带关键点保护的Douglas-Peucker算法"""
    if len(points) <= 2:
        return points
    
    # 找到距离起点和终点连线最远的点
    max_distance = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        distance = point_to_line_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # 检查最远点是否为关键点
    is_key_point = False
    if max_index > 0 and max_index < len(points) - 1:
        curvature = calculate_point_curvature(points[max_index-1], points[max_index], points[max_index+1])
        if curvature > 0.3:  # 关键点阈值
            is_key_point = True
    
    # 如果最大距离小于容差且不是关键点，简化为直线
    if max_distance < tolerance and not is_key_point:
        return [points[0], points[-1]]
    
    # 递归处理两段
    left_part = protected_douglas_peucker(points[:max_index + 1], tolerance)
    right_part = protected_douglas_peucker(points[max_index:], tolerance)
    
    # 合并结果（去除重复点）
    return left_part[:-1] + right_part

def point_to_line_distance(point, line_start, line_end):
    """计算点到直线的距离"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # 直线长度的平方
    line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
    
    if line_length_sq == 0:
        return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # 计算垂直距离
    t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length_sq))
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    return np.sqrt((x0 - projection_x)**2 + (y0 - projection_y)**2)

# 保留原有的函数作为备选
def smooth_stroke(stroke, tolerance=2.0, min_distance=3):
    """
    原始平滑函数 - 现在调用自适应版本
    """
    return adaptive_smooth_stroke(stroke, tolerance, min_distance)

def douglas_peucker(points, tolerance):
    """Douglas-Peucker算法简化路径"""
    return protected_douglas_peucker(points, tolerance)

def bezier_smooth(points, num_points=None):
    """使用贝塞尔曲线平滑路径 - 仅用于长线条"""
    if len(points) < 4:  # 提高使用贝塞尔的阈值
        return points
    
    # 检查是否为细节线条
    line_length = calculate_line_length(points)
    if line_length < 30:  # 短线条不使用贝塞尔平滑
        return points
    
    if num_points is None:
        num_points = max(len(points), int(len(points) * 0.9))  # 保留更多点
    
    # 转换为numpy数组
    points = np.array(points)
    
    # 生成平滑曲线
    smooth_points = []
    
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        
        # 计算控制点
        if i == 0:
            control1 = start + (end - start) * 0.2  # 减小控制点影响
        else:
            prev_point = points[i - 1]
            control1 = start + (end - prev_point) * 0.15
        
        if i == len(points) - 2:
            control2 = end - (end - start) * 0.2
        else:
            next_point = points[i + 2]
            control2 = end - (next_point - start) * 0.15
        
        # 生成贝塞尔曲线上的点
        segment_points = int(max(2, np.linalg.norm(end - start) / 15))  # 减少插值点
        for t in np.linspace(0, 1, segment_points):
            if t == 0 and len(smooth_points) > 0:
                continue
            
            point = (1-t)**3 * start + 3*(1-t)**2*t * control1 + 3*(1-t)*t**2 * control2 + t**3 * end
            smooth_points.append([int(point[0]), int(point[1])])
    
    return smooth_points