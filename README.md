# 计算机视觉目标检测项目

这是一个基于OpenCV的目标检测项目，使用Flask作为后端框架，实现了三种不同的目标检测算法。

## 项目结构

```
.
├── app.py              # Flask后端主程序
├── requirements.txt    # 项目依赖
├── web                 # 前端目录
├── detectors/          # 目标检测算法目录
│   ├── human_detector.py  # 人检测器
│   ├── dog_detector.py    # 狗检测器
│   └── car_detector.py    # 汽车检测器
├── uploads/           # 上传文件目录
└── outputs/           # 输出文件目录
```

## 环境要求

- Python 3.7+
- OpenCV 4.5.3
- Flask 2.0.1
- NumPy 1.21.2
- Vue3

## 安装

1. 克隆项目
```bash
https://github.com/nilahd/opencv.git
```
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

服务器将在 http://localhost:5000 启动。

## API文档

### 处理视频

**URL**: `/process_video`

**方法**: POST

**请求参数**:
- `video`: 视频文件（支持mp4, avi, mov格式）
- `target_type`: 目标类型（human/dog/car）

**响应**:
- 成功：返回处理后的视频文件
- 失败：返回错误信息

**示例请求**:
```bash
curl -X POST -F "video=@test.mp4" -F "target_type=human" http://localhost:5000/process_video
```

## 算法说明

1. 人检测器（HumanDetector）：
   - 使用HSV颜色空间和轮廓检测
   - 通过肤色特征识别人体
   - 使用形态学操作优化检测结果

2. 狗检测器（DogDetector）：
   - 使用KCF（Kernelized Correlation Filters）目标追踪
   - 通过肤色特征识别
   - 支持追踪失败后重新选择

3. 汽车检测器（CarDetector）：
   - 使用KCF（Kernelized Correlation Filters）目标追踪
   - 基于特征点检测和追踪
   - 显示目标运动轨迹

## 注意事项

1. 视频文件大小限制：默认无限制，可根据需要修改
2. 支持的视频格式：mp4, avi, mov
3. 处理大视频文件可能需要较长时间
4. 确保有足够的磁盘空间存储上传和处理后的视频 