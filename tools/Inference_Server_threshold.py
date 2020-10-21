from mmdet.apis import init_detector, inference_detector
import socket
import time
import json
import numpy as np
import cv2
from struct import *
from mmdet.datasets.imgprocess import server_det_bboxes
import mmcv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TCP_IP = '192.168.0.24'
# TCP_PORT = 5010
TCP_IP = 'localhost'
TCP_PORT = 28080
# TCP_IP = '172.31.8.152' #p2.xlarge
# TCP_PORT = 20000
# TCP_IP = '172.31.31.71' #g4dn.xlarge
# TCP_PORT = 20000

score_thr = 0.5
config_file = '../configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
# checkpoint_file = '../../../Dataset/Seoulchallenge/test_0226/best_log/epoch_41.pth'#"logs/epoch_63.pth"
checkpoint_file = 'epoch_41.pth'#"logs/epoch_63.pth"

CLASSES = ('fire','smoke','car','building','person','cemetery')

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = mmcv.Config.fromfile(config_file)
model.cfg.data.val.pipeline = cfg.test_pipeline
model.cfg.data.test.pipeline = cfg.test_pipeline

def recvall(sock,headersize):
    buf = b''
    header = unpack('>2sI',sock.recv(headersize)) # 'st'(2byte)+ length(4byte)
    if header[0] == b'st':
        count = header[1]
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf, header[1]

# def timecheck():

def ReceivingMsg(conn):
    infer_num = 0
    total_time = 0
    while True:
        # print("receiving..")
        try:
            stringData, imgsize = recvall(conn, 6)  # 6 = headersize
            stringData_decode = cv2.imdecode(np.fromstring(stringData, dtype='uint8'), 1)

            print("received!!")
            timecheck1 = time.time()

            timecheck2 = time.time()
            result = inference_detector(model, stringData_decode)

            ####
            # print("save:",timecheck2- timecheck1,"infer:",time.time()-timecheck2)
            infer_time = time.time()-timecheck1



            print("infer time: ",round(infer_time, 3))
            ####

            if isinstance(result, tuple):
                bbox_result, _ = result  # bbox_result, segm_result
            else:
                bbox_result, segm_result = result, None

            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            bboxes, labels = server_det_bboxes(bboxes, labels, class_names=CLASSES, score_thr=score_thr)

            object_coords = []
            for bbox, label in zip(bboxes, labels):
                bbox_int = bbox.astype(np.int32)
                x1 = str(bbox_int[0])
                y1 = str(bbox_int[1])
                x2 = str(bbox_int[2])
                y2 = str(bbox_int[3])
                obj_class = int(label)+1

                object_coords.append([x1, y1, x2, y2, obj_class])

            if not object_coords:
                object_coords = [[0,0,0,0,0]]

            object_coords_bytes = json.dumps(object_coords).encode('utf-8')
            object_coords_len = pack('>H',len(object_coords_bytes))
            conn.send(object_coords_len+object_coords_bytes)
            print("send!")
            test = 0
        except Exception:
            import traceback
            print(traceback.format_exc())
            s.listen(True)
            print('wait for client...')
            conn, addr = s.accept()
            print('connected!')


if __name__ == '__main__':

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(True)
    print('wait for client...')
    conn, addr = s.accept()
    print('connected!')

    ReceivingMsg(conn)
