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
import os
from mmdet.apis import init_detector, inference_detector
import time
from mmdet.datasets.imgprocess import server_det_bboxes, server_det_masks,server_det_masks_demo
import mmcv
from logger.logger import logger
from argparse import ArgumentParser

sel_server = selectors.DefaultSelector()
sel_client = selectors.DefaultSelector()


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    logger.info("accepted connection from %s,%d" % (addr[0], addr[1]))
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel_server.register(conn, events, data=data)


def service_connection(key_s, mask_s, sock_c, pre_calibrated, angle_threshold, ground_height, gsd):
    sock_s = key_s.fileobj
    data_s = key_s.data
    if mask_s & selectors.EVENT_READ:
        try:
            taskID, frameID, latitude, longitude, altitude, roll, pitch, yaw, camera, img = receive(sock_s)
            if taskID is None:
                logger.debug("No data received.")
                return

            start_time = time.time()
            # 1. Set IO
            my_drone = drones.Drones(make=camera)

            # 2. System calibration & CCS converting
            init_eo = np.array([longitude, latitude, altitude, roll, pitch, yaw])
            init_eo[:2] = geographic2plane(init_eo, 3857)

            if pre_calibrated:
                init_eo[3:] *= np.pi / 180
                adjusted_eo = init_eo
            else:
                my_georeferencer = georeferencers.DirectGeoreferencer()
                adjusted_eo = my_georeferencer.georeference(my_drone, init_eo)

            # 3. Inference
            timecheck1 = time.time()
            result = inference_detector(model, img)
            infer_time = time.time() - timecheck1
            logger.info("infer time: %.2f" % (round(infer_time, 3)))

            # contour(segmentation)
            object_coords = server_det_masks(result, class_names=CLASSES, score_thr=score_thr)
            # bounding box
            # object_coords = server_det_bboxes(result, score_thr=0)

            if object_coords:
                logger.debug(object_coords)

            # 4. Geo-referencing
                img_rows = img.shape[0]
                img_cols = img.shape[1]
                pixel_size = my_drone.sensor_width / img_cols  # mm/px
                R_CG = Rot3D(adjusted_eo).T

                inference_metadata = []
                for inference_px in object_coords:
                    inference_world = georef_inference(inference_px[:-1], img_rows, img_cols, pixel_size,
                                                       my_drone.focal_length, adjusted_eo, R_CG, ground_height)
                    inference_metadata.append(
                        # create_inference_metadata(inference_px[-1], str(inference_px), inference_world))
                        create_inference_metadata(inference_px[-1], inference_px[:-1], inference_world))
            else:
                logger.debug(object_coords)
                inference_metadata = []

            send(frameID, taskID, frameID, 0, "", inference_metadata, "", sock_c)
            logger.info("Sending completed! Elapsed time: %.2f" % (time.time() - start_time))
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
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

    parser = ArgumentParser(description="Configuration")
    parser.add_argument("--config", help="the name of config file", type=str, default='sandbox',
                        choices=['ndmi','sandbox','knps'])
    args = parser.parse_args()

    if args.config is 'ndmi':
        config_file = 'configs/config_ndmi.json'
    else:
        config_file = 'configs/config_sandbox.json'

    with open(config_file) as f:
        data = json.load(f)

    model = inference_module_init(data["config_file"],data["checkpoint_file"])
    score_thr = data["score_thr"]#0.7
    CLASSES = data["CLASSES"]



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

    logger.info("Viewer Connected!")

    pre_calibrated = eval(data["processing"]["pre_calibrated"])  # bool
    angle_threshold = data["processing"]["threshold_angle"]  # deg
    ground_height = data["processing"]["ground_height"]  # m
    gsd = data["processing"]["gsd"]  # m/px

    try:
        while True:
            events_servers = sel_server.select(timeout=None)
            # events_clients = sel_client.select(timeout=None)
            for key, mask in events_servers:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    service_connection(key, mask, sock_client,
                                       pre_calibrated, angle_threshold * np.pi / 180, ground_height, gsd)
            # Check for a socket being monitored to continue
            if client_connection:
                logger.info('starting connection...')
                sock_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_connection = sock_client.connect_ex((CLIENT_IP, CLIENT_PORT))
                logger.info("Viewer reconnected!")
    except KeyboardInterrupt:
        logger.error("caught keyboard interrupt, exiting")
    finally:
        sel_server.close()
        sel_client.close()
