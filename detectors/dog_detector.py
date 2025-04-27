import cv2
import numpy as np
import os

class DogDetector:
    def __init__(self):
        # 初始化颜色范围（适合柯基犬的橙黄色）
        self.lower_color = np.array([10, 100, 100], dtype=np.uint8)  # 黄橙色下限
        self.upper_color = np.array([30, 255, 255], dtype=np.uint8)  # 黄橙色上限
        
        # 初始化追踪器
        self.tracker = None
        self.bbox = None
        self.tracking = False
        
    def process_video(self, input_path, output_path):
        """
        处理视频文件，检测并追踪狗
        :param input_path: 输入视频路径
        :param output_path: 输出视频路径
        """
        # 打开视频文件
        cap = cv2.VideoCapture(input_path)
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建视频写入器，使用H.264编码
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用H.264编码
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 如果正在追踪
            if self.tracking:
                # 更新追踪器
                success, bbox = self.tracker.update(frame)
                
                if success:
                    # 绘制追踪框
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Dog", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # 追踪失败，重置追踪状态
                    self.tracking = False
            
            # 如果没有在追踪或是第一帧，尝试检测狗
            if not self.tracking or frame_count == 1:
                # 转换为HSV进行颜色检测
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # 创建颜色掩码
                color_mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
                
                # 形态学操作去除噪声
                kernel = np.ones((5,5), np.uint8)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                
                # 查找轮廓
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                largest_area = 0
                largest_rect = None
                
                # 找到最大的符合颜色条件的区域
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # 过滤小轮廓
                        x, y, w, h = cv2.boundingRect(contour)
                        # 计算宽高比，过滤不合理的区域
                        aspect_ratio = float(w) / h
                        if 0.5 < aspect_ratio < 2.0 and w > 30 and h > 30:
                            if area > largest_area:
                                largest_area = area
                                largest_rect = (x, y, w, h)
                
                # 如果找到足够大的区域，初始化追踪器
                if largest_rect:
                    x, y, w, h = largest_rect
                    # 绘制检测框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Dog Detected", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # 初始化追踪器
                    self.tracker = cv2.TrackerKCF_create()
                    self.tracker.init(frame, largest_rect)
                    self.tracking = True
            
            # 写入输出视频
            out.write(frame)
            
        # 释放资源
        cap.release()
        out.release()
        
        # 确保视频文件被正确写入
        if not os.path.exists(output_path):
            raise Exception("视频文件未成功生成") 