from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
from flask import Flask, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from flask import send_from_directory
import cv2
import numpy as np
import time
import json
import numpy as np
import cv2
from struct import *
import mmcv
import os
from mmdet.datasets.imgprocess  import show_results_selected
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config_file = 'configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = 'tools/logs/latest.pth'#_aug/epoch_40.pth'#'../../../Dataset/Seoulchallenge/test_0226/best_log/epoch_41.pth'#"logs/epoch_63.pth"

CLASSES = ('Building','Trash')

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = mmcv.Config.fromfile(config_file)
model.cfg.data.val.pipeline = cfg.test_pipeline
model.cfg.data.test.pipeline = cfg.test_pipeline

Infer_METHOD = ['Detection','Segmentation']
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__, static_url_path='/static')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER = '/home/user/test/rest_test'#/1584500178490-0.png'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_bbox(bbox_result):
    object_coords = []

    for class_id in range(len(bbox_result)):
        for object_num in range(len(bbox_result[class_id])):
            x1 = str((bbox_result[class_id][object_num][0]))
            y1 = str((bbox_result[class_id][object_num][1]))
            x2 = str((bbox_result[class_id][object_num][2]))
            y2 = str((bbox_result[class_id][object_num][3]))
            obj_class = class_id + 1
            object_coords.append([x1, y1, x2, y2, obj_class])

    if not object_coords:
        object_coords = [[0, 0, 0, 0, 0]]

    x1_ = []
    x2_ = []
    y1_ = []
    y2_ = []
    cls_id_ = []

    for obj_num in object_coords:
        x1_.append(int(round(float(obj_num[0]))))
        y1_.append(int(round(float(obj_num[1]))))
        x2_.append(int(round(float(obj_num[2]))))
        y2_.append(int(round(float(obj_num[3]))))
        cls_id_.append(int(round(int(obj_num[4]))))

    return [x1_, y1_, x2_, y2_, cls_id_]

def highlighting_bbox(image,bbox):
    grey = (128,128,128)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    black = (0, 0, 0)
    white = (255,255,255)
    purple = (128,0,128)
    notorange = (255,127,80)
    object_color = {0: black, 1: red, 2: grey, 3: black, 4: blue, 5: purple, 6: white}#fire,somke,car,building,person,cemetery
    idx = 0
    for object_id in bbox[4]:
        image = cv2.rectangle(image, (bbox[0][idx] - 5, bbox[1][idx] - 5), (bbox[2][idx] + 5, bbox[3][idx] + 5), object_color[object_id], 1)#3)#5)#visdrone etc :2,
        idx += 1
    return image

def inference_object(stringData_decode):


    print("received!!")
    timecheck1 = time.time()
    result = inference_detector(model, stringData_decode)

    ####
    infer_time = time.time() - timecheck1
    print("infer time: ", round(infer_time, 3))
    ####

    return result

@app.route('/', methods=['GET', 'POST'])
def ReceivingMsg():
    if request.method == 'POST':


        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            stringData = file.stream.read()
            stringData_decode = cv2.imdecode(np.fromstring(stringData, dtype='uint8'), 1)
            height,width,_ = stringData_decode.shape
            print(height,width)
            result = inference_object(stringData_decode)
            # infer_method = 1 if request.form.get("method")=="Segmentation" else 0
            if request.form.get("method") == "Segmentation":
                infer_method = 1
                filename = filename[:-4]+"_segm.png"
            else:
                infer_method =0
                filename = filename[:-4]+"_detect.png"
            # if  == "Segmentation":
            #     detection = 0
            #
            # else: # Object detection(only bbox)
            #     bbox_result, _ = result
            #     bbox_info = extract_bbox(bbox_result)  ## need to revise !!! no efficient!!
            #     bboxed_img = highlighting_bbox(stringData_decode, bbox_info)
            #     cv2.imwrite('static/' + filename, bboxed_img)

            # show_result_pyplot(stringData_decode, result, CLASSES, infer_method, out='static/' + filename)  # ,fig_size=(1920,1080))
            show_results_selected(stringData_decode,result,CLASSES,infer_method,score_thr=0.3,out_file='static/' + filename)
            print("saved!")

            return render_template('img_test.html', image_file=filename)#,wid=str(width), hei=str(height))

    return render_template('main.html', image_file="innopam2.jpg",methods=Infer_METHOD)


if __name__ == '__main__':
    app.run(host='192.168.0.24')

# import requests
# from flask import Flask, Response, stream_with_context
#
# app = Flask(__name__)
#
# my_path_to_server01 = 'http://localhost:5000/'
#
# @app.route("/")
# def streamed_proxy():
#     r = requests.get(my_path_to_server01, stream=True)
#     return Response(r.iter_content(chunk_size=10*1024),
#                     content_type=r.headers['Content-Type'])
#
# if __name__ == "__main__":
#     app.run(port=1234)
