import cv2
import numpy as np
import os

class HumanDetector:
    def __init__(self):
        # 初始化HSV颜色范围
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # 初始化HOG行人检测器
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 初始化追踪器列表
        self.trackers = []  # 存储追踪器对象
        self.tracking_bboxes = []  # 存储追踪框
        self.tracking_colors = []  # 存储每个目标的颜色
        
        # 检测参数
        self.win_stride = (8, 8)
        self.padding = (8, 8)
        self.scale = 1.05
        
        # 追踪参数
        self.max_trackers = 5  # 最大追踪目标数
        self.min_distance = 50  # 最小目标间距（像素）
        
    def _is_overlapping(self, bbox1, bbox2):
        """检查两个边界框是否重叠"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算中心点距离
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance < self.min_distance
        
    def _get_random_color(self):
        """生成随机颜色"""
        return tuple(np.random.randint(0, 255, 3).tolist())
        
    def process_video(self, input_path, output_path):
        """
        处理视频文件，检测并追踪多个人体目标
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
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_interval = 30  # 每30帧进行一次检测
        
        while cap.isOpened():
            ret, frame = frame_copy = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 更新所有追踪器
            good_trackers = []
            good_bboxes = []
            good_colors = []
            
            for tracker, bbox, color in zip(self.trackers, self.tracking_bboxes, self.tracking_colors):
                success, new_bbox = tracker.update(frame)
                if success:
                    good_trackers.append(tracker)
                    good_bboxes.append(new_bbox)
                    good_colors.append(color)
                    
                    # 绘制追踪框
                    x, y, w, h = [int(v) for v in new_bbox]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"ID: {len(good_trackers)}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 更新追踪器列表
            self.trackers = good_trackers
            self.tracking_bboxes = good_bboxes
            self.tracking_colors = good_colors
            
            # 定期进行检测或当追踪目标数量不足时
            if frame_count % detection_interval == 0 or len(self.trackers) < self.max_trackers:
                # 使用HOG检测行人
                boxes, weights = self.hog.detectMultiScale(
                    frame, 
                    winStride=self.win_stride,
                    padding=self.padding,
                    scale=self.scale
                )
                
                # 过滤检测结果
                boxes = [box for box, weight in zip(boxes, weights) if weight > 0.5]
                
                # 过滤掉与现有追踪目标重叠的检测框
                new_boxes = []
                for box in boxes:
                    is_valid = True
                    for tracked_box in self.tracking_bboxes:
                        if self._is_overlapping(box, tracked_box):
                            is_valid = False
                            break
                    if is_valid:
                        new_boxes.append(box)
                
                # 添加新的追踪目标
                for box in new_boxes:
                    if len(self.trackers) >= self.max_trackers:
                        break
                        
                    x, y, w, h = box
                    
                    # 初始化新的KCF追踪器
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    
                    # 添加到追踪器列表
                    self.trackers.append(tracker)
                    self.tracking_bboxes.append((x, y, w, h))
                    self.tracking_colors.append(self._get_random_color())
                    
                    # 绘制检测框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # 写入输出视频
            out.write(frame)
            
        # 释放资源
        cap.release()
        out.release()
        
        # 确保视频文件被正确写入
        if not os.path.exists(output_path):
            raise Exception("视频文件未成功生成") 