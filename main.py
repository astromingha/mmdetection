import sys
import socket
import selectors
import types
from image_processing.socket_module import receive, send
import json
import numpy as np
from image_processing import drones
import image_processing.georef_for_eo as georeferencers
from image_processing.georef_for_gp import Rot3D,georef_inference,create_inference_metadata, geographic2plane
from image_processing import rectifiers
import time
import os
from mmdet.apis import init_detector, inference_detector
import time
from mmdet.datasets.imgprocess import server_det_bboxes, server_det_masks,server_det_masks_demo
import mmcv
from logger.logger import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sel_server = selectors.DefaultSelector()
sel_client = selectors.DefaultSelector()


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    logger.info("accepted connection from %s,%d" % (addr[0], addr[1]))
    # https://stackoverflow.com/questions/39145357/python-error-socket-error-errno-11-resource-temporarily-unavailable-when-s
    # conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel_server.register(conn, events, data=data)


def start_connections(host, port, num_conns):
    server_addr = (host, port)
    for i in range(0, num_conns):
        connid = i + 1
        logger.info('starting connection %d to %s' % (connid, server_addr))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(server_addr)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(connid=connid, outb=b'')
        sel_client.register(sock, events, data=data)


def service_connection(key_s, mask_s, sock_c):
    sock_s = key_s.fileobj
    data_s = key_s.data
    if mask_s & selectors.EVENT_READ:
        try:
            taskID, frameID, latitude, longitude, altitude, roll, pitch, yaw, camera, img = receive(sock_s)
            if taskID is None:
                return

            start_time = time.time()
            # 1. Set IO
            my_drone = drones.Drones(make=camera, pre_calibrated=False)
            # my_drone = drones.Drones(make=camera, ground_height=38.0, pre_calibrated=True)  # Only for test - Jeonju

            # 2. System calibration & CCS converting
            init_eo = np.array([longitude, latitude, altitude, roll, pitch, yaw])
            init_eo[:2] = geographic2plane(init_eo, 3857)
            if my_drone.pre_calibrated:
                init_eo[3:] *= np.pi / 180
                adjusted_eo = init_eo
            else:
                my_georeferencer = georeferencers.DirectGeoreferencer()
                adjusted_eo = my_georeferencer.georeference(my_drone, init_eo)

            # # 3. Rectify
            # my_rectifier = rectifiers.AverageOrthoplaneRectifier(height=my_drone.ground_height)
            # bbox_wkt, orthophoto = my_rectifier.rectify(img, my_drone, adjusted_eo)

            # 3. Inference
            timecheck1 = time.time()
            result = inference_detector(model, img)

            ####
            # print("save:",timecheck2- timecheck1,"infer:",time.time()-timecheck2)
            infer_time = time.time() - timecheck1

            logger.info("infer time: %.2f" % (round(infer_time, 3)))
            ####

            if isinstance(result, tuple):
                bbox_result, _ = result  # bbox_result, segm_result
            else:
                bbox_result, segm_result = result, None
            ##########MASK###########
            object_coords = server_det_masks(result, class_names=CLASSES, score_thr=score_thr)
            # object_coords = server_det_masks_demo(result, class_names=CLASSES, score_thr=score_thr)
            #####Bounding Box#######
            # bboxes = np.vstack(bbox_result)
            # labels = [
            #     np.full(bbox.shape[0], i, dtype=np.int32)
            #     for i, bbox in enumerate(bbox_result)
            # ]
            # labels = np.concatenate(labels)

            # bboxes, labels = server_det_bboxes(result, class_names=CLASSES, score_thr=score_thr)
            # object_coords = server_det_bboxes(result, class_names=CLASSES, score_thr=score_thr)

            # object_coords = []
            # for bbox, label in zip(bboxes, labels):
            #     bbox_int = bbox.astype(np.int32)
            #     x1 = (bbox_int[0])
            #     y1 = (bbox_int[1])
            #     x2 = (bbox_int[2])
            #     y2 = (bbox_int[3])
            #     obj_class = int(label) + 1
            #     # sandbox label## revise
            #     if obj_class == 3 or obj_class == 4 or obj_class == 5:
            #         obj_class -= 2
            #     else:
            #         continue
            #     ########################
            #     object_coords.append([x1, y1, x2, y1, x2, y2, x1, y2, obj_class])
            #################
            if object_coords:
                logger.debug(object_coords)

                # 4. Georeferencing
                img_rows = img.shape[0]
                img_cols = img.shape[1]
                pixel_size = my_drone.sensor_width / img_cols  # mm/px
                R_CG = Rot3D(adjusted_eo).T

                inference_metadata = []
                for inference_px in object_coords:
                    inference_world = georef_inference(inference_px[:-1], img_rows, img_cols, pixel_size,
                                                       my_drone.focal_length, adjusted_eo, R_CG, my_drone.ground_height)
                    inference_metadata.append(
                        create_inference_metadata(inference_px[-1], str(inference_px), inference_world))

                send(frameID, taskID, frameID, 0, "", inference_metadata, "", sock_c)  # 메타데이터 생성/ send to client
                logger.info("Sending completed! Elapsed time: %.2f" % (time.time() - start_time))
            else:
                logger.debug(object_coords)
                send(frameID, taskID, frameID, 0, "", [], "", sock_c)  # 메타데이터 생성/ send to client
                logger.info("Sending completed! Elapsed time: %.2f" % (time.time() - start_time))
        except Exception as e:
            logger.error(e)
            logger.warning("closing connection to %s" % (json.dumps(data_s.addr)))
            sock_c.close()
            global client_connection
            client_connection = 1
            sel_server.unregister(sock_s)
            sock_s.close()

def inference_module_init(config_file, checkpoint_file):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    cfg = mmcv.Config.fromfile(config_file)
    model.cfg.data.val.pipeline = cfg.test_pipeline
    model.cfg.data.test.pipeline = cfg.test_pipeline
    return model


if __name__ == '__main__':

    with open("configs/config.json") as f:
        data = json.load(f)

    model = inference_module_init(data["config_file"],data["checkpoint_file"])
    score_thr = data["score_thr"]#0.7
    CLASSES = ['fire', 'smoke', 'car', 'building', 'person', 'cemetery']



    ### SERVER
    SERVER_PORT = data["server"]["PORT"]
    QUEUE_LIMIT = data["server"]["QUEUE_LIMIT"]

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid bind() exception: OSError: [Errno 48] Address already in use
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("", SERVER_PORT))
    lsock.listen()
    logger.info("listening on %s,%d" % ("", SERVER_PORT))
    # logger.info("listening on", ("", SERVER_PORT))
    lsock.setblocking(False)
    sel_server.register(lsock, selectors.EVENT_READ, data=None)

    ### CLIENT
    CLIENT_IP = data["client"]["IP"]
    CLIENT_PORT = data["client"]["PORT"]
    num_conn = data["client"]["NoC"]
    logger.info('starting connection...')
    sock_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_connection = sock_client.connect_ex((CLIENT_IP, CLIENT_PORT))
    # client_connection = 1
    logger.info("Connected!")

    try:
        while True:
            events_servers = sel_server.select(timeout=None)
            # events_clients = sel_client.select(timeout=None)
            for key, mask in events_servers:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    service_connection(key, mask, sock_client)
            # Check for a socket being monitored to continue
            if client_connection:
                logger.info('starting connection...')
                sock_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_connection = sock_client.connect_ex((CLIENT_IP, CLIENT_PORT))
                # client_connection = 1
                logger.info("Connected!")
    except KeyboardInterrupt:
        logger.error("caught keyboard interrupt, exiting")
    finally:
        sel_server.close()
        sel_client.close()