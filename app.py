from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from detectors.human_detector import HumanDetector
from detectors.dog_detector import DogDetector
from detectors.car_detector import CarDetector

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# 允许的文件类型
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    处理视频的API端点
    接收视频文件和目标类型，返回处理后的视频
    """
    if 'video' not in request.files:
        return jsonify({'error': '没有视频文件'}), 400
    
    video_file = request.files['video']
    target_type = request.form.get('target_type', '')
    
    if video_file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(video_file.filename):
        return jsonify({'error': '不支持的文件类型'}), 400
    
    if not target_type:
        return jsonify({'error': '未指定目标类型'}), 400
    
    # 保存上传的视频
    filename = secure_filename(video_file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(input_path)
    
    # 根据目标类型选择对应的检测器
    if target_type == 'human':
        detector = HumanDetector()
    elif target_type == 'dog':
        detector = DogDetector()
    elif target_type == 'car':
        detector = CarDetector()
    else:
        return jsonify({'error': '不支持的目标类型'}), 400
    
    # 处理视频
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'processed_{filename}')
    detector.process_video(input_path, output_path)
    
    # 返回处理后的视频
    return send_file(
        output_path,
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f'processed_{filename}'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000) 