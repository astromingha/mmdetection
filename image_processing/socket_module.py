import numpy as np
import uuid
import json
from socket import *
from struct import *
import logging
import cv2
import time
from logger.logger import logger

def recvall(sock,datalength):
    buf = b''
    count = datalength
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

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
    for domain in domains:
        dt = np.dtype(np.uint16)
        dt = dt.newbyteorder('<')
        value = zero_fill_right_shift(np.frombuffer(binary_header, dtype=dt)[0] << (16 + domain["offset"]),
                                      32 - domain["length"])
        result.append(value)

    return result


def receive(c_sock):
    headersize = 2
    binaryHeader = c_sock.recv(headersize)  # Read the length of header

    while True:
        if parse_header(binaryHeader) == [1, 17, 34, 68, 1089, 17424]:
            temp = c_sock.recv(12)
            timeStamp, payloadLength = unpack('<QI', temp)
            imageHeaderLength = 14
            payloadLength -= imageHeaderLength

            payload1 = recvall(c_sock, 60)
            taskID, frameID, latitude, longitude, altitude, accuracy, jsonDataSize = unpack('<16s16sddffI', payload1)

            jsonData = recvall(c_sock, jsonDataSize)
            imageBytes = recvall(c_sock, unpack('<I', c_sock.recv(4))[0])

            taskID = uuid.UUID(bytes=taskID)
            frameID = uuid.UUID(bytes=frameID)
            my_json = jsonData.decode('utf8').replace("'", '"')
            data = json.loads(my_json)
            nparr = cv2.imdecode(np.fromstring(imageBytes, dtype='uint8'), 1)

            return taskID, frameID, latitude, longitude, altitude, data["roll"], data["pitch"], data["yaw"], data["exif"]["Model"], nparr


def send(frame_id, task_id, name, img_type, img_boundary, objects, orthophoto, client):
    """
        Create a metadata of an orthophoto for tcp transmission
        :param frame_id: uuid of the image | string
        :param task_id: task id of the image | string
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

    # print(img_metadata)

    # Write image to memory
    # orthophoto_encode = cv2.imencode('.png', orthophoto)
    # orthophoto_bytes = orthophoto_encode[1].tostring()
    orthophoto_bytes = str.encode(orthophoto)

    #############################################
    # Send object information to web map viewer #
    #############################################
    full_length = len(img_metadata_bytes) + len(orthophoto_bytes)
    fmt = '<4siii' + str(len(img_metadata_bytes)) + 's' + str(len(orthophoto_bytes)) + 's'  # s: string, i: int
    # print(fmt, b"INFE", full_length, len(img_metadata_bytes), len(orthophoto_bytes),
    #                     img_metadata_bytes)
    logger.debug('%s,INFE,%d,%d,%d,%s' % (json.dumps(fmt), full_length, len(img_metadata_bytes), len(orthophoto_bytes), json.dumps(img_metadata)))
    data_to_send = pack(fmt, b"INFE", full_length, len(img_metadata_bytes), len(orthophoto_bytes),
                        img_metadata_bytes, orthophoto_bytes)
    client.send(data_to_send)


# # https://stackoverflow.com/questions/26445331/how-can-i-have-multiple-clients-on-a-tcp-python-chat-server
# def client_thread(s_sock, client):
#     s_sock.send(b"Welcome to the Server. Type messages and press enter to send.\n")
#     while True:
#         start_time = time.time()
#         taskID, frameID, latitude, longitude, altitude, roll, pitch, yaw, img = receive(s_sock)
#         if not taskID or not frameID or not latitude or not longitude or not altitude \
#                 or not roll or not pitch or not yaw:
#             break
#
#         # 1. Set IO
#         my_drone = drones.DJIPhantom4RTK(pre_calibrated=True)
#         # sensor_width = my_drone.sensor_width
#         # focal_length = my_drone.focal_length
#         # gsd = my_drone.gsd
#         # ground_height = my_drone.ground_height
#         # R_CB = my_drone.R_CB
#         # comb = my_drone.comb
#         # manufacturer = my_drone.manufacturer
#
#         # 2. System calibration & CCS converting
#         init_eo = np.array([longitude, latitude, altitude, roll, pitch, yaw])
#         if my_drone.pre_calibrated:
#             init_eo[3:] = init_eo[3:] * np.pi / 180
#             adjusted_eo = init_eo
#         else:
#             my_georeferencer = georeferencers.DirectGeoreferencer()
#             adjusted_eo = my_georeferencer.georeference(my_drone, init_eo)
#
#         # 3. Rectify
#         my_rectifier = rectifiers.AverageOrthoplaneRectifier(height=my_drone.ground_height)
#         bbox_wkt, orthophoto = my_rectifier.rectify(img, my_drone, adjusted_eo)
#
#         logging.info('========================================================================================')
#         logging.info('========================================================================================')
#         logging.info('A new image is received.')
#         logging.info('File name: %s' % frameID)
#         logging.info('Current Drone: %s' % my_drone.__class__.__name__)
#         logging.info('========================================================================================')
#
#         send(frameID, taskID, frameID, 0, bbox_wkt, [], orthophoto, client)    # 메타데이터 생성/ send to client
#         print(time.time() - start_time)