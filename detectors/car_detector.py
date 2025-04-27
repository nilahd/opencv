import cv2
import numpy as np
import os
import time

class CarDetector:
    def __init__(self):
        # 尝试加载汽车检测级联分类器
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_car.xml'
            if os.path.exists(cascade_path):
                self.car_cascade = cv2.CascadeClassifier(cascade_path)
                self.cascade_loaded = not self.car_cascade.empty()
            else:
                self.cascade_loaded = False
                print(f"无法找到级联分类器文件: {cascade_path}")
        except Exception as e:
            self.cascade_loaded = False
            print(f"加载级联分类器时出错: {str(e)}")
            
        # 追踪器列表和背景减除器
        self.trackers = []
        self.detection_interval = 30  # 每30帧进行一次检测
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50,
            detectShadows=False
        )
        
    def detect_cars_by_motion(self, frame, width, height):
        """使用背景减除法检测移动的汽车"""
        # 应用背景差分
        fg_mask = self.bg_subtractor.apply(frame)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        car_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 2000:  # 过滤小轮廓
                x, y, w, h = cv2.boundingRect(contour)
                # 过滤不合理的区域并检查宽高比
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 3.0 and w > 80 and h > 60 and w < width/2 and h < height/2:
                    car_regions.append((x, y, w, h))
        
        return car_regions
        
    def process_video(self, input_path, output_path):
        """
        处理视频文件，检测并追踪汽车
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
        
        # 处理视频帧
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 创建帧的副本
            result_frame = frame.copy()
            
            # 每隔一定帧数重新检测车辆
            if frame_count % self.detection_interval == 0 or frame_count == 1 or len(self.trackers) == 0:
                # 重置追踪器
                self.trackers = []
                car_regions = []
                
                # 使用级联分类器检测汽车（如果可用）
                if self.cascade_loaded:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cars = self.car_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.1, 
                            minNeighbors=5,
                            minSize=(80, 80),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        for (x, y, w, h) in cars:
                            if w > 60 and h > 60 and w < width/2 and h < height/2:
                                car_regions.append((x, y, w, h))
                    except Exception as e:
                        print(f"级联分类器检测错误: {str(e)}")
                
                # 如果级联分类器检测失败或找不到车辆，使用背景差分法
                if not car_regions:
                    car_regions = self.detect_cars_by_motion(frame, width, height)
                
                # 为每个检测到的车辆创建追踪器
                for (x, y, w, h) in car_regions:
                    try:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        
                        # 在检测阶段画红色框
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(result_frame, "Car Detected", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"创建追踪器错误: {str(e)}")
            
            # 更新所有追踪器
            trackers_to_keep = []
            for tracker in self.trackers:
                try:
                    success, bbox = tracker.update(frame)
                    
                    if success:
                        # 绘制追踪框
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_frame, "Car", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 保留成功的追踪器
                        trackers_to_keep.append(tracker)
                except Exception as e:
                    print(f"更新追踪器错误: {str(e)}")
            
            # 更新追踪器列表
            self.trackers = trackers_to_keep
            
            # 如果没有追踪到任何车辆，使用背景差分法检测
            if not self.trackers and frame_count % 5 == 0:
                car_regions = self.detect_cars_by_motion(frame, width, height)
                
                for (x, y, w, h) in car_regions:
                    try:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x, y, w, h))
                        self.trackers.append(tracker)
                        
                        # 在背景差分检测阶段画蓝色框
                        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(result_frame, "Moving Car", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"创建追踪器错误: {str(e)}")
            
            # 添加帧计数和时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(result_frame, f"Frame: {frame_count} | Time: {timestamp}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 写入输出视频
            out.write(result_frame)
            
        # 释放资源
        cap.release()
        out.release()
        
        # 确保视频文件被正确写入
        if not os.path.exists(output_path):
            raise Exception("视频文件未成功生成") 