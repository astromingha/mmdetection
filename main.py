import os
from socket import *
from _thread import start_new_thread


from mmdet.apis import init_detector
import json
import mmcv
from time import sleep
from server_func import client_thread


config_file = 'configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = 'configs/Seoulchallenge_6class_epoch_41.pth'#"logs/epoch_63.pth"

model = init_detector(config_file, checkpoint_file, device='cuda:0')
cfg = mmcv.Config.fromfile(config_file)
model.cfg.data.val.pipeline = cfg.test_pipeline
model.cfg.data.test.pipeline = cfg.test_pipeline

with open("configs/config.json") as f:
    data = json.load(f)

SERVER_PORT = data["server"]["PORT"]
QUEUE_LIMIT = data["server"]["QUEUE_LIMIT"]     # 서버 대기 큐

CLIENT_IP = data["client"]["IP"]
CLIENT_PORT = data["client"]["PORT"]

server = socket(AF_INET, SOCK_STREAM)    # 소켓 생성 (UDP = SOCK_DGRAM, TCP = SOCK_STREAM)
server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
server.bind(('', SERVER_PORT))           # 포트 설정
server.listen(QUEUE_LIMIT)               # 포트 ON

client = socket(AF_INET, SOCK_STREAM)
#client.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
# client.connect((CLIENT_IP, CLIENT_PORT))

while client.connect_ex((CLIENT_IP, CLIENT_PORT)) != 0:
    print("connect retry..")
    sleep(3)
print("viewer server connected!")
print('tcp server ready')
print('wait for client ')


while True:
    try:
        s_sock, s_addr = server.accept()
        print('connected from {}:{}'.format(s_addr[0], s_addr[1]))
        start_new_thread(client_thread, (s_sock, model, client))
    except Exception:
        import traceback
        print(traceback.format_exc())

        server.listen(QUEUE_LIMIT)   # 포트 ON
        print('Re: tcp echo server ready')  # 준비 완료 화면에 표시
        print('Re: wait for client')   # 연결 대기

        s_sock, s_addr = server.accept()
        print('Re: connected from {}:{}'.format(s_addr[0], s_addr[1]))

        start_new_thread(client_thread, (s_sock, model, client))


