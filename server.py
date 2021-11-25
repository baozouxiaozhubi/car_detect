from flask import Flask, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
# 文件上传目录
app.config['UPLOAD_FOLDER'] = r'.\yolov5-5.0\data\mydata'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}  # 集合类型

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/upload', methods=['POST'])
def uploadAndDetect():
    def upload():
        cnt=request.form.get('cnt')
        for i in range(1,int(cnt)+1):
            upload_file = request.files['image%d'%i]
            if upload_file and allowed_file(upload_file.filename):
                filename = secure_filename(upload_file.filename)
                # 将文件保存到 static/uploads 目录，文件名同上传时使用的文件名
                upload_file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))
                print('image%d'%i,"upload success")
            else:
                print('image%d'%i,"upload failed")

    def mydetect():
        import sys
        import argparse
        from setuptools.dist import check_requirements
        sys.path.append(r'.\yolov5-5.0')
        import detect
        import argparse
        import time
        from pathlib import Path

        import cv2
        import torch
        import torch.backends.cudnn as cudnn
        from numpy import random
        from models.experimental import attempt_load
        from utils.datasets import LoadStreams, LoadImages
        from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
            apply_classifier, \
            scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
        from utils.plots import plot_one_box
        from utils.torch_utils import select_device, load_classifier, time_synchronized

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default=r'.\yolov5-5.0\runs\train\exp\weights\best.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=r'.\yolov5-5.0\data\mydata',
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=r'.\yolov5-5.0\runs\detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)
        check_requirements(exclude=('pycocotools', 'thop'))

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    ss=detect.detect(opt)
                    strip_optimizer(opt.weights)
            else:
                ss=detect.detect(opt)

        #返回每张图片的预测结果，结果为字典{class1：cnt1,class2,cnt2}，视频则每一帧检测一次
        for s in ss:
            cnt=s.split()[1:]
            classify={}
            for i in range(int(len(cnt)/2)):
                classify[cnt[i]]=cnt[i+1].replace(',','')
            print(classify)



    def clear():
        import os
        import shutil
        delList = []
        delDir = r'.\yolov5-5.0\data\mydata'
        delList = os.listdir(delDir)
        for f in delList:
            filePath = os.path.join(delDir, f)
        if os.path.isfile(filePath):
            os.remove(filePath)
        elif os.path.isdir(filePath):
            shutil.rmtree(filePath, True)
    #上传
    upload()
    #预测
    mydetect()
    #清理
    clear()
    return ""
if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)