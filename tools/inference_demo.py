# -*- coding: utf-8 -*-
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv, os

IMAGE_DIR = '/home/user/Dataset/NDMI/disaster/2nd/coco/images/val2017'
Save_txt_DIR ='/home/user/Dataset/NDMI/disaster/3rd/coco/result_miou/'
config_file = "../configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py"
checkpoint_file = 'logs/epoch_48.pth'

show = 0
makemAPtxt = 1
GPU_NUM = 1



model = init_detector(config_file, checkpoint_file, device='cuda:'+str(GPU_NUM))

# test a single image
filenames = [i for i in os.listdir(IMAGE_DIR) if '.jpg' in i.lower() or '.png' in i.lower()]
for filename in filenames:
    file_dir = os.path.join(IMAGE_DIR, filename)

    if ".jpg" or ".png" in filename.lower():
        result = inference_detector(model, file_dir)

    if show:
        show_result_pyplot(model, file_dir, result)

    if makemAPtxt:
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        with open(Save_txt_DIR + "/" + os.path.splitext(filename)[0] + ".txt", "w") as f:
            for class_id in range(len(bbox_result)):
                for object_num in range(len(bbox_result[class_id])):  # range(np.vstack(bbox_result)):
                    bbox_left = (bbox_result[class_id][object_num][0])  # str(int(r['rois'][idx][1]))
                    bbox_top = (bbox_result[class_id][object_num][1])
                    bbox_right = (bbox_result[class_id][object_num][2])
                    bbox_bottom = (bbox_result[class_id][object_num][3])
                    score = (bbox_result[class_id][object_num][4])

                    f.write('%d %.2f %.1f %.1f %.1f %.1f\n' % (class_id+1, score, bbox_left, bbox_top, bbox_right, bbox_bottom))
