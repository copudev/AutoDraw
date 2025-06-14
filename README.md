# AutoDraw - 自动画图工具

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://github.com/yourusername/autodraw)

**一款基于计算机视觉的自动绘图工具，能够将图片转换为可绘制的线条路径并自动执行绘制**

</div>

## 📸 演示效果

> 将任意图片转换为线稿并自动绘制

## ✨ 主要特性

- 🎨 **智能线条提取** - 使用 OpenCV 的 Canny 边缘检测算法提取图像轮廓
- 🖱️ **自动绘制** - 通过 PyAutoGUI 控制鼠标自动完成绘制过程
- 🎛️ **灵活参数调节** - 支持多种预设模式和自定义参数
- 📋 **多种输入方式** - 支持剪贴板图片、本地文件、文件夹浏览
- 🎯 **实时预览** - 处理前可预览线稿效果
- 🎪 **多种绘图模式** - 适配"你画我猜"、微软画图等不同应用场景

## 📦 安装要求

### 系统要求
- Windows 操作系统
- Python 3.6 或更高版本

### 依赖包
```bash
pip install opencv-python
pip install pyautogui
pip install numpy
pip install pillow
pip install prompt-toolkit
```

### 一键安装
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/copudev/autodraw.git
cd autodraw
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行程序
```bash
python main.py
```

### 4. 基础使用流程

1. **准备图片** - 复制图片到剪贴板或准备本地图片文件
2. **设置起点** - 使用 `:start-point` 命令设置绘制起始位置
3. **预览效果** - 使用 `show` 或 `select` 命令预览线稿效果
4. **开始绘制** - 使用 `draw` 或 `select-draw` 命令开始自动绘制

## 🎮 命令参考

### 基础命令
| 命令 | 功能 | 示例 |
|------|------|------|
| `help` | 显示帮助信息 | `help` |
| `show` | 预览剪贴板图片线稿 | `show` |
| `draw` | 绘制剪贴板图片 | `draw` |
| `file [路径]` | 预览指定文件 | `file image.jpg` |
| `draw-file [路径]` | 绘制指定文件 | `draw-file image.jpg` |
| `select` | 文件选择对话框预览 | `select` |
| `select-draw` | 文件选择对话框绘制 | `select-draw` |
| `browse` | 浏览文件夹 | `browse` |

### 参数设置
| 命令 | 功能 | 范围 | 示例 |
|------|------|------|------|
| `:threshold1 [值]` | 设置边缘检测下阈值 | 0-255 | `:threshold1 50` |
| `:threshold2 [值]` | 设置边缘检测上阈值 | 0-255 | `:threshold2 100` |
| `:start-point [x] [y]` | 设置绘制起点 | - | `:start-point 100 200` |
| `:resize [倍数]` | 设置缩放因子 | >0 | `:resize 0.5` |
| `:maxsize [像素]` | 设置最大尺寸 | >0 | `:maxsize 800` |
| `:blur [大小]` | 设置模糊核大小 | 奇数 | `:blur 5` |

### 预设模式
| 模式 | 适用场景 | 特点 |
|------|----------|------|
| `simple` | 简笔画、卡通风格 | 简化线条，减少细节 |
| `:preset detailed` | 详细插画 | 保留更多细节 |
| `:preset smooth` | 平滑线条 | 减少噪点 |
| `:preset complex` | 复杂图像 | 自适应处理 |
| `:preset sketch` | 手绘风格 | 模拟手绘效果 |

## 💡 使用技巧

### 推荐工作流程

#### 简笔画模式
```bash
simple                    # 切换到简笔画模式
select                    # 选择并预览图片
:start-point             # 设置鼠标当前位置为起点
select-draw              # 选择图片并开始绘制
```

#### 复杂图像模式
```bash
:preset complex          # 应用复杂图像预设
:maxsize 600            # 限制图像大小
select                   # 预览效果
:start-point            # 设置起点
select-draw             # 开始绘制
```

### 参数调优指南

- **threshold1**: 较小值检测更多边缘细节，较大值减少噪点
- **threshold2**: 通常设为 threshold1 的 2-3 倍
- **blur**: 较小值保留细节，较大值去除噪点
- **resize**: 建议 0.3-2.0 之间，过大可能导致绘制时间过长

## 🔧 高级配置

### 自定义绘制函数

程序支持两种绘制模式：

1. **你画我猜模式** (`draw_trajectory`)
   - 快速绘制，适合在线绘图游戏
   
2. **微软画图模式** (`draw_trajectory_ms_painting`)
   - 拖拽绘制，适合传统绘图软件

### 图像预处理选项

可以在 `line_draft` 函数中调整以下参数：
- 对比度增强
- 自适应阈值
- 形态学操作
- 边缘连接

## 📁 项目结构

```
autodraw/
├── main.py              # 主程序入口
├── mctl.py              # 鼠标控制模块
├── minfo.py             # 鼠标信息获取
├── lib.py               # 核心图像处理库
├── imgprocess.py        # 图像预处理模块
├── requirements.txt     # 依赖列表
└── README.md           # 项目说明
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发设置
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 报告问题
- 使用 GitHub Issues 报告 bug
- 提供详细的重现步骤
- 包含系统信息和错误日志

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [OpenCV](https://opencv.org/) - 计算机视觉库
- [PyAutoGUI](https://pyautogui.readthedocs.io/) - GUI 自动化
- [Prompt Toolkit](https://python-prompt-toolkit.readthedocs.io/) - 命令行界面

## 📞 联系方式

- 项目主页: [https://github.com/copudev/autodraw](https://github.com/copudev/autodraw)
- 问题反馈: [Issues](https://github.com/copudev/autodraw/issues)

---

<div align="center">
⭐ 如果这个项目对你有帮助，请给个 Star！非常感谢！！！
</div>
