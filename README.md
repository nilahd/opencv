<!--
 * @Author: Coisini 3059421373@qq.com
 * @Date: 2025-04-27 11:09:46
 * @LastEditors: Coisini 3059421373@qq.com
 * @LastEditTime: 2025-04-28 10:40:20
 * @FilePath: \git\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 视频目标检测与追踪系统

这是一个基于OpenCV和Flask的视频目标检测与追踪系统，支持多种目标的实时检测和追踪。

## 功能特性

- 支持多种目标检测：
  - 行人检测
  - 车辆检测
  - 小狗检测（柯基犬）
- 多目标追踪能力：
  - 支持同时追踪多个目标
  - 智能目标管理，避免重复追踪
  - 自动处理目标丢失和新增
- 实时视频处理：
  - 支持多种视频格式（MP4, AVI, MOV）
  - 实时显示检测和追踪结果

## 技术栈

- 后端：
  - Python 3.x
  - Flask 2.0.1
  - OpenCV 4.5.3
  - NumPy 1.21.2
- 前端：
  - Vue.js
  - HTML5
  - CSS3

## 安装说明

1. 克隆项目：
```bash
git clone https://github.com/nilahd/opencv.git
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动后端服务：
```bash
python app.py
```
服务将在 http://localhost:5000 启动

2. 启动前端服务：
```bash
cd web/demo
npm install
npm run serve
```
前端服务将在 http://localhost:8080 启动

3. 使用系统：
   - 打开浏览器访问 http://localhost:8080
   - 上传视频文件
   - 选择目标类型（行人/车辆/动物）
   - 点击处理按钮
   - 等待处理完成后下载结果视频

## 项目结构

```
.
├── app.py                 # Flask后端主程序
├── requirements.txt       # Python依赖
├── detectors/            # 检测器模块
│   ├── human_detector.py # 行人检测器
│   ├── car_detector.py   # 车辆检测器
│   └── dog_detector.py   # 动物检测器
├── web/                  # 前端代码
│   └── demo/            # Vue.js项目
├── uploads/             # 上传文件目录
└── outputs/             # 输出文件目录
```


## 注意事项

1. 确保系统已安装Python 3.x
2. 视频文件大小建议不超过100MB
3. 处理时间取决于视频长度和复杂度
4. 建议使用现代浏览器访问系统
5. 配置前端开发环境

