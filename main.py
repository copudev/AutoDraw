import time
import tkinter as tk
from tkinter import filedialog
import os

from prompt_toolkit import *
from prompt_toolkit.completion import *
from prompt_toolkit.document import Document
from prompt_toolkit.validation import *
from PIL import ImageGrab
from lib import pil2cv, line_draft
from imgprocess import image2stroke
from mctl import draw_stroke
import cv2
import pyautogui

app_banner = "<b><yellow bg='black'>Auto</yellow><black bg='yellow'>Draw</black></b>\n" \
             "<gray>Version: 0.1.0</gray>\n" \
             "输入 <b>`help`</b> 获取帮助信息"

# 全局变量定义
threshold1 = 50          
threshold2 = 100         
start_point = [0, 0]
resize_factor = 1.0
max_size = 800
blur_kernel = 5          
use_adaptive = False     
enable_smooth = True
smooth_tolerance = 2.0
min_point_distance = 3

# 函数定义部分
def app_help(_):
    help_text = "AutoDraw 自动画图工具\n" \
                "支持剪切板图片和文件直接处理\n\n" \
                "基础命令\n" \
                "`:threshold1 [数值]` 设置边缘检测下阈值 (0-255，值越小检测越多)\n" \
                "`:threshold2 [数值]` 设置边缘检测上阈值 (0-255)\n" \
                "`:blur [数值]` 设置模糊核大小 (奇数，3-15)\n" \
                "`:adaptive [on/off]` 开启/关闭自适应阈值模式\n" \
                "`:preset [模式]` 应用预设参数 (detailed/smooth/complex/sketch/simple)\n" \
                "`:smooth [on/off]` 开启/关闭线条平滑\n" \
                "`select` 打开文件选择对话框预览图片\n" \
                "`select-draw` 打开文件选择对话框并直接绘制\n" \
                "`browse` 浏览文件夹中的图片\n" \
                "`simple` 快速切换到简笔画模式\n" \
                "`file [路径]` 预览指定文件的线稿效果\n" \
                "`draw-file [路径]` 使用指定文件进行自动绘制\n" \
                "`show` 预览剪切板图片的线稿效果\n" \
                "`draw` 使用剪贴板图片进行自动绘制\n\n" \
                "推荐流程\n" \
                "简笔画模式: `simple` → `select` → `:start-point` → `select-draw`\n" \
                "复杂图像: `:preset complex` → `select` → `:start-point` → `select-draw`\n" \
                "手绘风格: `:preset sketch` → `select` → `:start-point` → `select-draw`\n\n" \
                "`quit` 退出程序"
    print_formatted_text(HTML(help_text))

def app_threshold1(text):
    """设置边缘检测下阈值"""
    global threshold1
    if len(text) == 1:
        print_formatted_text(HTML(f"当前 threshold1: <green>{threshold1}</green>"))
    else:
        try:
            threshold1 = int(text[1])
            print_formatted_text(HTML(f"threshold1 设置为: <green>{threshold1}</green>"))
            print_formatted_text(HTML("<yellow>提示: 较小值检测更多边缘细节</yellow>"))
        except ValueError:
            print_formatted_text(HTML("<red>threshold1 必须是整数</red>"))

def app_threshold2(text):
    """设置边缘检测上阈值"""
    global threshold2
    if len(text) == 1:
        print_formatted_text(HTML(f"当前 threshold2: <green>{threshold2}</green>"))
    else:
        try:
            threshold2 = int(text[1])
            print_formatted_text(HTML(f"threshold2 设置为: <green>{threshold2}</green>"))
            print_formatted_text(HTML("<yellow>提示: 通常设为 threshold1 的 2-3 倍</yellow>"))
        except ValueError:
            print_formatted_text(HTML("<red>threshold2 必须是整数</red>"))

def app_start_point(text):
    """设置绘制起点"""
    global start_point
    if len(text) == 1:
        import pyautogui
        start_point = list(pyautogui.position())
        print_formatted_text(HTML(f"绘制起点设置为: <green>{start_point}</green>"))
        print_formatted_text(HTML("<yellow>提示: 请确保鼠标已移动到画布的合适位置</yellow>"))
    else:
        try:
            x = int(text[1])
            y = int(text[2]) if len(text) > 2 else start_point[1]
            start_point = [x, y]
            print_formatted_text(HTML(f"绘制起点设置为: <green>{start_point}</green>"))
        except (ValueError, IndexError):
            print_formatted_text(HTML("<red>用法: :start-point [x] [y] 或 :start-point (使用当前鼠标位置)</red>"))

def app_resize(text):
    """设置图像缩放因子"""
    global resize_factor
    if len(text) == 1:
        print_formatted_text(HTML(f"当前缩放因子: <green>{resize_factor}</green>"))
    else:
        try:
            resize_factor = float(text[1])
            print_formatted_text(HTML(f"缩放因子设置为: <green>{resize_factor}</green>"))
            print_formatted_text(HTML("<yellow>提示: 1.0=原始大小, 0.5=缩小一半, 2.0=放大一倍</yellow>"))
        except ValueError:
            print_formatted_text(HTML("<red>缩放因子必须是数字</red>"))

def app_maxsize(text):
    """设置最大尺寸限制"""
    global max_size
    if len(text) == 1:
        print_formatted_text(HTML(f"当前最大尺寸: <green>{max_size}</green>像素"))
    else:
        try:
            max_size = int(text[1])
            print_formatted_text(HTML(f"最大尺寸设置为: <green>{max_size}</green>像素"))
            print_formatted_text(HTML("<yellow>提示: 图像会自动缩放到不超过此尺寸</yellow>"))
        except ValueError:
            print_formatted_text(HTML("<red>最大尺寸必须是整数</red>"))

def app_blur(text):
    """设置模糊核大小"""
    global blur_kernel
    if len(text) == 1:
        print_formatted_text(HTML(f"当前模糊核大小: <green>{blur_kernel}</green>"))
    else:
        try:
            blur_kernel = int(text[1])
            if blur_kernel % 2 == 0:
                blur_kernel += 1  # 确保是奇数
            print_formatted_text(HTML(f"模糊核大小设置为: <green>{blur_kernel}</green>"))
            print_formatted_text(HTML("<yellow>提示: 较小值保留更多细节，较大值去除更多噪点</yellow>"))
        except ValueError:
            print_formatted_text(HTML("<red>模糊核大小必须是整数</red>"))

def app_adaptive(text):
    """切换自适应阈值模式"""
    global use_adaptive
    if len(text) == 1:
        status = "开启" if use_adaptive else "关闭"
        print_formatted_text(HTML(f"自适应阈值模式: <green>{status}</green>"))
    else:
        if text[1].lower() in ['on', '1', 'true', '开启']:
            use_adaptive = True
            print_formatted_text(HTML("<green>自适应阈值模式已开启</green>"))
            print_formatted_text(HTML("<yellow>提示: 适合处理复杂光照的图像</yellow>"))
        else:
            use_adaptive = False
            print_formatted_text(HTML("<green>自适应阈值模式已关闭</green>"))

def app_preset(text):
    """应用预设参数"""
    global threshold1, threshold2, blur_kernel, use_adaptive
    
    if len(text) < 2:
        print_formatted_text(HTML("<yellow>可用预设:</yellow>"))
        print_formatted_text(HTML("detailed  - 详细模式 (检测更多细节)"))
        print_formatted_text(HTML("smooth    - 平滑模式 (减少噪点)"))
        print_formatted_text(HTML("complex   - 复杂图像模式"))
        print_formatted_text(HTML("sketch    - 手绘风格模式"))
        print_formatted_text(HTML("simple    - 简笔画模式 (简化线条)"))
        return
    
    preset = text[1].lower()
    
    if preset == "detailed":
        threshold1, threshold2, blur_kernel, use_adaptive = 30, 80, 3, False
        print_formatted_text(HTML("<green>已应用详细模式预设</green>"))
    elif preset == "smooth":
        threshold1, threshold2, blur_kernel, use_adaptive = 100, 200, 7, False
        print_formatted_text(HTML("<green>已应用平滑模式预设</green>"))
    elif preset == "complex":
        threshold1, threshold2, blur_kernel, use_adaptive = 20, 60, 5, True
        print_formatted_text(HTML("<green>已应用复杂图像模式预设</green>"))
    elif preset == "sketch":
        threshold1, threshold2, blur_kernel, use_adaptive = 50, 120, 3, False
        print_formatted_text(HTML("<green>已应用手绘风格模式预设</green>"))
    elif preset == "simple":
        threshold1, threshold2, blur_kernel, use_adaptive = 80, 160, 9, False
        print_formatted_text(HTML("<green>已应用简笔画模式预设</green>"))
        print_formatted_text(HTML("<yellow>提示: 此模式会简化线条，减少细节，适合卡通风格</yellow>"))
    else:
        print_formatted_text(HTML("<red>未知预设，请使用: detailed, smooth, complex, sketch, simple</red>"))
        return
    
    print_formatted_text(HTML(f"<cyan>threshold1={threshold1}, threshold2={threshold2}, blur={blur_kernel}, adaptive={use_adaptive}</cyan>"))

def app_simple_mode(_):
    """快速切换到简笔画模式"""
    global threshold1, threshold2, blur_kernel, use_adaptive
    threshold1, threshold2, blur_kernel, use_adaptive = 80, 160, 9, False
    print_formatted_text(HTML("<green>已切换到简笔画模式</green>"))
    print_formatted_text(HTML("<yellow>特点: 简化线条，减少细节，适合卡通风格绘制</yellow>"))
    print_formatted_text(HTML(f"<cyan>参数: threshold1={threshold1}, threshold2={threshold2}, blur={blur_kernel}</cyan>"))

def simple_drawing_mode(im):
    """简笔画模式的特殊处理"""
    import cv2
    import numpy as np
    
    # 1. 更强的双边滤波，保持边缘的同时大幅平滑区域
    im_smooth = cv2.bilateralFilter(im, 15, 80, 80)
    
    # 2. 形态学操作简化结构
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    im_smooth = cv2.morphologyEx(im_smooth, cv2.MORPH_CLOSE, kernel)
    im_smooth = cv2.morphologyEx(im_smooth, cv2.MORPH_OPEN, kernel)
    
    # 3. 减少灰度级别，创造卡通效果
    im_quantized = im_smooth // 32 * 32
    
    return im_quantized

def preprocess_image(im):
    """预处理图像：缩放和尺寸限制"""
    import cv2
    
    height, width = im.shape[:2]
    print_formatted_text(HTML(f"<cyan>原始图像尺寸: {width}x{height}</cyan>"))
    
    # 应用缩放因子
    if resize_factor != 1.0:
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        im = cv2.resize(im, (new_width, new_height))
        print_formatted_text(HTML(f"<cyan>缩放后尺寸: {new_width}x{new_height}</cyan>"))
        width, height = new_width, new_height
    
    # 应用最大尺寸限制
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        im = cv2.resize(im, (new_width, new_height))
        print_formatted_text(HTML(f"<cyan>限制尺寸后: {new_width}x{new_height}</cyan>"))
    
    return im

def app_file(text):
    """从文件加载图片"""
    if len(text) < 2:
        print_formatted_text(HTML("<red>请提供文件路径。用法: file [文件路径]</red>"))
        return
    
    file_path = " ".join(text[1:])
    
    try:
        from PIL import Image
        im = Image.open(file_path)
        
        print_formatted_text(HTML(f"<green>成功加载图片: {file_path}</green>"))
        print_formatted_text(HTML(f"<cyan>图像模式: {im.mode}, 原始尺寸: {im.size}</cyan>"))
        
        im = pil2cv(im)
        if im is None:
            print_formatted_text(HTML("<red>图像转换失败。</red>"))
            return
        
        # 应用图像预处理
        im = preprocess_image(im)
        
        # 检查是否为简笔画模式
        if threshold1 == 80 and threshold2 == 160 and blur_kernel == 9:
            print_formatted_text(HTML("<cyan>检测到简笔画模式，应用特殊处理...</cyan>"))
            im = simple_drawing_mode(im)
        
        try:
            im_processed = line_draft(im, threshold1=threshold1, threshold2=threshold2,
                                 blur_kernel=blur_kernel, use_adaptive=use_adaptive)
            cv2.imshow("image", im_processed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print_formatted_text(HTML(f"<red>处理图像时出错: {e}</red>"))
            
    except FileNotFoundError:
        print_formatted_text(HTML(f"<red>找不到文件: {file_path}</red>"))
    except Exception as e:
        print_formatted_text(HTML(f"<red>加载文件时出错: {e}</red>"))

def app_draw_file(text):
    """从文件加载图片并绘制"""
    if len(text) < 2:
        print_formatted_text(HTML("<red>请提供文件路径。用法: draw-file [文件路径]</red>"))
        return
    
    file_path = " ".join(text[1:])
    
    try:
        from PIL import Image
        im = Image.open(file_path)
        
        print_formatted_text(HTML(f"<green>成功加载图片: {file_path}</green>"))
        print_formatted_text(HTML(f"<cyan>图像模式: {im.mode}, 原始尺寸: {im.size}</cyan>"))
        
        im = pil2cv(im)
        if im is None:
            print_formatted_text(HTML("<red>图像转换失败。</red>"))
            return
        
        # 应用图像预处理
        im = preprocess_image(im)
        
        # 检查是否为简笔画模式
        if threshold1 == 80 and threshold2 == 160 and blur_kernel == 9:
            print_formatted_text(HTML("<cyan>检测到简笔画模式，应用特殊处理...</cyan>"))
            im = simple_drawing_mode(im)
        
        try:
            stroke = image2stroke(im, preprocess=True, threshold1=threshold1, threshold2=threshold2)
            if not stroke:
                print_formatted_text(HTML("<red>未检测到可绘制的线条。</red>"))
                return
                
            for t in [3, 2, 1]:
                print(f"{t}...", end="", flush=True)
                time.sleep(1)
            print("")
            
            draw_stroke(stroke, start_point, min=1)
        except pyautogui.FailSafeException:
            print_formatted_text(HTML("<red>绘制已停止。</red>"))
        except Exception as e:
            print_formatted_text(HTML(f"<red>绘制过程中出错: {e}</red>"))
            
    except FileNotFoundError:
        print_formatted_text(HTML(f"<red>找不到文件: {file_path}</red>"))
    except Exception as e:
        print_formatted_text(HTML(f"<red>加载文件时出错: {e}</red>"))

def app_show(_):
    im = ImageGrab.grabclipboard()
    
    if im is None:
        print_formatted_text(HTML("<red>剪贴板中没有图片。</red>"))
        return
    
    if not hasattr(im, 'mode'):
        print_formatted_text(HTML(f"<red>剪贴板内容不是图片，而是: {type(im)}</red>"))
        return
    
    im = pil2cv(im)
    if im is None:
        print_formatted_text(HTML("<red>图像转换失败。</red>"))
        return
    
    im = preprocess_image(im)
    
    try:
        im = line_draft(im, threshold1=threshold1, threshold2=threshold2, 
                       blur_kernel=blur_kernel, use_adaptive=use_adaptive)
        cv2.imshow("image", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print_formatted_text(HTML(f"<red>处理图像时出错: {e}</red>"))

def app_draw(_):
    im = ImageGrab.grabclipboard()
    if im is None:
        print_formatted_text(HTML("<red>剪贴板中没有图片。</red>"))
        return
    
    if not hasattr(im, 'mode'):
        print_formatted_text(HTML(f"<red>剪贴板内容不是有效图片，类型: {type(im)}</red>"))
        return
    
    im = pil2cv(im)
    if im is None:
        print_formatted_text(HTML("<red>图像转换失败。</red>"))
        return
    
    im = preprocess_image(im)
    
    try:
        stroke = image2stroke(im, preprocess=True, threshold1=threshold1, threshold2=threshold2)
        if not stroke:
            print_formatted_text(HTML("<red>未检测到可绘制的线条。</red>"))
            return
            
        for t in [3, 2, 1]:
            print(f"{t}...", end="", flush=True)
            time.sleep(1)
        print("")
        
        draw_stroke(stroke, start_point, min=1)
    except pyautogui.FailSafeException:
        print_formatted_text(HTML("<red>绘制已停止。</red>"))
    except Exception as e:
        print_formatted_text(HTML(f"<red>绘制过程中出错: {e}</red>"))

def app_select(_):
    """打开文件选择对话框"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        filetypes = [
            ('图片文件', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp'),
            ('JPEG文件', '*.jpg *.jpeg'),
            ('PNG文件', '*.png'),
            ('WebP文件', '*.webp'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择要预览的图片",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )
        
        root.destroy()
        
        if file_path:
            print_formatted_text(HTML(f"<green>已选择文件: {file_path}</green>"))
            app_file(["file", file_path])
        else:
            print_formatted_text(HTML("<yellow>未选择文件</yellow>"))
            
    except Exception as e:
        print_formatted_text(HTML(f"<red>打开文件选择对话框时出错: {e}</red>"))

def app_select_draw(_):
    """打开文件选择对话框并直接绘制"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        filetypes = [
            ('图片文件', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp'),
            ('JPEG文件', '*.jpg *.jpeg'),
            ('PNG文件', '*.png'),
            ('WebP文件', '*.webp'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择要绘制的图片",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )
        
        root.destroy()
        
        if file_path:
            print_formatted_text(HTML(f"<green>已选择文件: {file_path}</green>"))
            app_draw_file(["draw-file", file_path])
        else:
            print_formatted_text(HTML("<yellow>未选择文件</yellow>"))
            
    except Exception as e:
        print_formatted_text(HTML(f"<red>打开文件选择对话框时出错: {e}</red>"))

def app_browse(_):
    """浏览并选择文件夹中的图片"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory(
            title="选择包含图片的文件夹",
            initialdir=os.path.expanduser("~")
        )
        
        root.destroy()
        
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            image_files = []
            
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
            
            if image_files:
                print_formatted_text(HTML(f"<green>找到 {len(image_files)} 个图片文件:</green>"))
                for i, file in enumerate(image_files[:10]):
                    filename = os.path.basename(file)
                    print_formatted_text(HTML(f"<cyan>{i+1}. {filename}</cyan>"))
                
                if len(image_files) > 10:
                    print_formatted_text(HTML(f"<yellow>... 还有 {len(image_files)-10} 个文件</yellow>"))
                
                print_formatted_text(HTML("<yellow>使用以下命令处理特定文件:</yellow>"))
                example_file = image_files[0].replace('\\', '\\\\')
                print_formatted_text(HTML(f'<cyan>file "{example_file}"</cyan>'))
                print_formatted_text(HTML(f'<cyan>draw-file "{example_file}"</cyan>'))
            else:
                print_formatted_text(HTML("<red>该文件夹中没有找到图片文件</red>"))
        else:
            print_formatted_text(HTML("<yellow>未选择文件夹</yellow>"))
            
    except Exception as e:
        print_formatted_text(HTML(f"<red>浏览文件夹时出错: {e}</red>"))

app_completer = FuzzyCompleter(NestedCompleter({
    'show': None,
    "draw": None,
    "file": None,
    "draw-file": None,
    "select": None,
    "select-draw": None,
    "browse": None,
    "simple": None,
    ":resize": None,
    ":maxsize": None,
    ":blur": None,
    ":adaptive": None,
    ":preset": None,
    "help": None,
    "quit": None,
    ":threshold1": None,
    ":threshold2": None,
    ":start-point": None
}))

app_callback = {
    "help": app_help,
    "quit": lambda _: exit(0),
    ":threshold1": app_threshold1,
    ":threshold2": app_threshold2,
    ":start-point": app_start_point,
    ":resize": app_resize,
    ":maxsize": app_maxsize,
    ":blur": app_blur,
    ":adaptive": app_adaptive,
    ":preset": app_preset,
    "show": app_show,
    "draw": app_draw,
    "file": app_file,
    "draw-file": app_draw_file,
    "select": app_select,
    "select-draw": app_select_draw,
    "browse": app_browse,
    "simple": app_simple_mode,
}

class AppValidator(Validator):
    def validate(self, document: Document) -> None:
        text = document.text.split()
        if len(text) == 0:
            raise ValidationError(message="Powered by Kl1nge5")
        if text[0] not in app_callback:
            raise ValidationError(message="Unknown command")
        if text[0] in [":threshold1", ":threshold2"] and len(text) > 1:
            if not text[1].isdigit():
                raise ValidationError(message="Argument must be digit")
            if int(text[1]) > 255 or int(text[1]) < 0:
                raise ValidationError(message="Argument must be >=0 and <=255")
        if text[0] == ":resize" and len(text) > 1:
            try:
                float(text[1])
            except ValueError:
                raise ValidationError(message="Resize factor must be a number")
        if text[0] == ":maxsize" and len(text) > 1:
            if not text[1].isdigit():
                raise ValidationError(message="Max size must be an integer")

app_session = PromptSession()

def mainloop():
    text = app_session.prompt("> ", completer=app_completer, validator=AppValidator()).strip().split()
    callback = app_callback.get(text[0], None)
    callback(text)

if __name__ == '__main__':
    print_formatted_text(HTML(app_banner))
    while True:
        mainloop()