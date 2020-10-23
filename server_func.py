import numpy as np
import uuid
import json
from socket import *
from struct import *

import logging
import cv2

from mmdet.apis import init_detector, inference_detector
import time
from mmdet.datasets.imgprocess import server_det_bboxes
import mmcv
import os

import image_processing.drones_socket as drones
import image_processing.georeferencers_socket as georeferencers
from image_processing.georef import georef_inference, Rot3D, create_inference_metadata, geographic2plane

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

score_thr = 0.7
CLASSES = ('fire', 'smoke', 'car', 'building', 'person', 'cemetery')

with open("configs/config.json") as f:
    data = json.load(f)

SERVER_PORT = data["server"]["PORT"]
QUEUE_LIMIT = data["server"]["QUEUE_LIMIT"]     # 서버 대기 큐

CLIENT_IP = data["client"]["IP"]
CLIENT_PORT = data["client"]["PORT"]

# https://stackoverflow.com/questions/55014710/zero-fill-right-shift-in-python
def zero_fill_right_shift(val, n):
    return (val >> n) if val >= 0 else ((val + 0x100000000) >> n)


def parse_header(binary_header):
    domains = [{"name": "version", "offset": 0, "length": 2, "type": "int16"},
               {"name": "messageType", "offset": 2, "length": 4, "type": "int16"},
               {"name": "ping", "offset": 6, "length": 1, "type": "boolean"},
               {"name": "pong", "offset": 7, "length": 1, "type": "boolean"},
               {"name": "countOfImages", "offset": 8, "length": 4, "type": "int16"},
               {"name": "reservation", "offset": 12, "length": 4, "type": "int16"}]
    result = []
    # for domain in domains:
    #     dt = np.dtype(np.uint16)
    #     dt = dt.newbyteorder('<')
    #     value = zero_fill_right_shift(np.frombuffer(binary_header, dtype=dt)[0] << (16 + domain["offset"]),
    #                                   32 - domain["length"])
    #     result.append(value)

    return result


def receive(c_sock):
    binaryHeader = c_sock.recv(2)  # Read the length of header
    packetHeader = parse_header(binaryHeader)
    timeStamp = c_sock.recv(8)
    timeStamp = np.frombuffer(timeStamp, dtype="int64")[0]
    payloadLength = c_sock.recv(4)
    payloadLength = np.frombuffer(payloadLength, dtype="int32")[0]

    # https://docs.python.org/ko/3/library/uuid.html
    taskID = c_sock.recv(16)
    taskID = uuid.UUID(bytes=taskID)
    frameID = c_sock.recv(16)
    frameID = uuid.UUID(bytes=frameID)
    latitude = c_sock.recv(8)
    latitude = np.frombuffer(latitude, dtype="double")[0]
    longitude = c_sock.recv(8)
    longitude = np.frombuffer(longitude, dtype="double")[0]
    altitude = c_sock.recv(4)
    altitude = np.frombuffer(altitude, dtype="float32")[0]
    accuracy = c_sock.recv(4)
    accuracy = np.frombuffer(accuracy, dtype="float32")[0]
    jsonDataSize = c_sock.recv(4)
    jsonDataSize = np.frombuffer(jsonDataSize, dtype="int32")[0]

    jsonData = c_sock.recv(jsonDataSize)  # binary
    # https://stackoverflow.com/questions/40059654/python-convert-a-bytes-array-into-json-format
    my_json = jsonData.decode('utf8').replace("'", '"')
    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(my_json)

    # dumped_json = json.dumps(data, indent=4, sort_keys=True)

    # jsonObject
    imageBinaryLength = c_sock.recv(4)
    imageBinaryLength = np.frombuffer(imageBinaryLength, dtype="int32")[0]

    byteBuff = b''
    while len(byteBuff) < imageBinaryLength:
        byteBuff += c_sock.recv(imageBinaryLength - len(byteBuff))
    nparr = cv2.imdecode(np.fromstring(byteBuff, dtype='uint8'), 1)

    if len(byteBuff) == 0:
        return

    print(timeStamp, payloadLength, taskID, frameID, latitude, longitude, altitude, accuracy, jsonDataSize,
          data["roll"], data["pitch"], data["yaw"])

    with open('flight_logs/' + str(frameID) + '.txt', 'w') as f:
        f.write(str(taskID) + ',' + str(frameID) + ',' + str(float(latitude)) + ',' + str(float(longitude)) + ',' + str(
            float(altitude)) + ',' + str(float(accuracy)) + ',' +
                json.dumps(data))
    c_sock.send(b"Done")

    return taskID, frameID, latitude, longitude, altitude, data, nparr


def send(frame_id, task_id, name, img_type, img_boundary, objects, orthophoto, client):
    """
        Create a metadata of an orthophoto for tcp transmission
        :param uuid: uuid of the image | string
        :param uuid: task id of the image | string
        :param name: A name of the original image | string
        :param img_type: A type of the image - optical(0)/thermal(1) | int
        :param img_boundary: Boundary of the orthophoto | string in wkt
        :param objects: JSON object? array? of the detected object ... from create_obj_metadata
        :return: JSON object of the orthophoto ... python dictionary
    """
    img_metadata = {
        "uid": str(frame_id),  # string
        "task_id": str(task_id),  # string
        "img_name": str(name),  # string
        "img_type": img_type,  # int
        "img_boundary": img_boundary,  # WKT ... string
        "objects": objects
    }
    img_metadata_bytes = json.dumps(img_metadata).encode()
    print(objects)

    # # Write image to memory
    # orthophoto_encode = cv2.imencode('.png', orthophoto)
    # orthophoto_bytes = orthophoto_encode[1].tostring()
    orthophoto_bytes = str.encode(orthophoto)

    #############################################
    # Send object information to web map viewer #
    #############################################
    full_length = len(img_metadata_bytes) + len(orthophoto_bytes)
    fmt = '<4siii' + str(len(img_metadata_bytes)) + 's' + str(len(orthophoto_bytes)) + 's'  # s: string, i: int
    data_to_send = pack(fmt, b"INFE", full_length, len(img_metadata_bytes), len(orthophoto_bytes),
                        img_metadata_bytes, orthophoto_bytes)
    client.send(data_to_send)
    print("send!")
    client.close()
    while client.connect_ex((CLIENT_IP, CLIENT_PORT)) != 0:
        print("connect retry..")
        time.sleep(1)
    # try:
    #     client.send(data_to_send)
    #     print("send!")
    # except Exception:
    #     import traceback
    #     print(traceback.format_exc())
    #     client.close()
    #     print("pipe broked, close")
    #     time.sleep(1)
    #     client = socket(AF_INET, SOCK_STREAM)
    #     # client.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    #     print("reconnect")
    #     while client.connect_ex((CLIENT_IP, CLIENT_PORT)) != 0:
    #         print("connect retry..")
    #         time.sleep(1)
    #     client.send(data_to_send)
    #     print("send!")

# https://stackoverflow.com/questions/26445331/how-can-i-have-multiple-clients-on-a-tcp-python-chat-server
def client_thread(s_sock, model, client):
    # s_sock.send(b"Welcome to the Server. Type messages and press enter to send.\n")
    while True:
        # taskID, frameID, latitude, longitude, altitude, roll, pitch, yaw, img = receive(s_sock)
        taskID, frameID, latitude, longitude, altitude, jsondata, img = receive(s_sock)

        #if not taskID or not frameID or not latitude or not longitude or not altitude \
         #       or not roll or not pitch or not yaw:
         #   break

        # 1. Set IO
        if jsondata["exif"]["Model"] == "FC6310R":
            my_drone = drones.DJIPhantom4RTK(pre_calibrated=True)
        elif jsondata["exif"]["Model"] == "DSC-RX100M4":
            my_drone = drones.SonyRX100M4(pre_calibrated=False)

        # 2. System calibration & CCS converting
        # init_eo = np.array([longitude, latitude, altitude, roll, pitch, yaw])
        init_eo = np.array([longitude, latitude, altitude, jsondata["roll"], jsondata["pitch"], jsondata["yaw"]])
        init_eo[:2] = geographic2plane(init_eo, 3857)
        if my_drone.pre_calibrated:
            init_eo[3:] = init_eo[3:] * np.pi / 180
            adjusted_eo = init_eo
        else:
            my_georeferencer = georeferencers.DirectGeoreferencer()
            adjusted_eo = my_georeferencer.georeference(my_drone, init_eo)

        # 3. Inference
        timecheck1 = time.time()
        result = inference_detector(model, img)

        ####
        # print("save:",timecheck2- timecheck1,"infer:",time.time()-timecheck2)
        infer_time = time.time() - timecheck1

        print("infer time: ", round(infer_time, 3))
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
            x1 = (bbox_int[0])
            y1 = (bbox_int[1])
            x2 = (bbox_int[2])
            y2 = (bbox_int[3])
            obj_class = int(label) + 1
            #sandbox label## revise
            if obj_class == 3 or obj_class == 4 or obj_class == 5:
                obj_class -= 2
            else:
                continue
            ########################
            object_coords.append([x1, y1, x2, y1, x2, y2, x1, y2, obj_class])

        if not object_coords:
            object_coords = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]

        print(object_coords)

        # 4. Georeferencing
        img_rows = img.shape[0]
        img_cols = img.shape[1]
        pixel_size = my_drone.sensor_width / img_cols  # mm/px
        R_CG = Rot3D(adjusted_eo).T

        inference_metadata = []
        for inference_px in object_coords:
            inference_world = georef_inference(inference_px[:-1], img_rows, img_cols, pixel_size,
                                               my_drone.focal_length, adjusted_eo, R_CG, my_drone.ground_height)
            inference_metadata.append(create_inference_metadata(inference_px[-1], str(inference_px), inference_world))

        send(frameID, taskID, frameID, 0, "", inference_metadata, "", client)  # 메타데이터 생성/ send to client
