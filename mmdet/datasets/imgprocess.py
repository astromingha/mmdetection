import cv2
from PIL import Image
import numpy as np
from skimage import measure
from PIL.ExifTags import TAGS, GPSTAGS

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=2):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def getExif(path):
    src_image = Image.open(path)
    info = src_image._getexif()
    test = 1
    # if info is not None:
    #     # Focal Length
    #     # focalLength = info[37386]
    #     # focal_length = focalLength[0] / focalLength[1] # unit: mm
    #     # focal_length = focal_length * pow(10, -3) # unit: m
    #
    #     # Orientation
    #     orientation = info[274]
    # else:
    #     orientation = None
    try:
        orientation = info[274]
        if orientation == 3:
            test = 1
    except:
        orientation = 0

    # return focal_length, orientation
    return orientation

def restoreOrientation(image, orientation):
    if orientation == 8:
        restored_image = rotate(image, -90)
    elif orientation == 6:
        restored_image = rotate(image, 90)
    elif orientation == 3:
        restored_image = rotate(image, 180)
    else:
        restored_image = image

    return restored_image



# def imgRotation(img_array,img_dir):
def imgRotation(img_dir):
    orientation = getExif(img_dir)
    img_original  = cv2.imread(img_dir)
    img_array = restoreOrientation(img_original,orientation)

    return img_array



def rotate(image, angle):
    # https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    height = image.shape[0]
    width = image.shape[1]
    center = (width/2, height/2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # compute the new bounding dimensions of the image
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # adjust the rotation matrix to take into account translation
    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]

    # perform the actual rotation and return the image
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def restoreVertices(bbox,ori): #dict = [x1_batch,y1_batch,x2_batch,y2_batch,class_batch]

    X1_batch = []
    Y1_batch = []
    X2_batch = []
    Y2_batch = []
    degree ={6: 90, 3: 180}

    cos = np.cos(degree[ori] * np.pi/180)
    sin = np.sin(degree[ori] * np.pi/180)

    for i in range(len(bbox[0])):
        x1 = bbox[0][i]
        y1 = bbox[1][i]
        x3 = bbox[2][i]
        y3 = bbox[3][i]

        X1 = x1 * cos + y1 * sin
        Y1 = -x1 * sin + y1 * cos
        X3 = x3 * cos + y3 * sin
        Y3 = -x3 * sin + y3 * cos
        ## origin shift
        X1 += abs(X3 - X1)
        X3 += abs(X3 - X1)

        if degree[ori] == 180:
            Y1 += abs(Y3-Y1)
            Y3 += abs(Y3-Y1)

        X1_batch.append(int(X1))
        X2_batch.append(int(X3))
        Y1_batch.append(int(Y1))
        Y2_batch.append(int(Y3))

    return [X1_batch,Y1_batch,X2_batch,Y2_batch,bbox[4]]

import mmcv
import pycocotools.mask as maskUtils

def server_det_masks(result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    assert isinstance(class_names, (tuple, list))
    # img = mmcv.imread(img)
    # img = imgRotation(img)
    # img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    polygons,plabels = [],[]
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]

        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            poly = [int(j) for j in binary_mask_to_polygon(mask)[0]]
            obj_class = int(labels[i])+1
            if obj_class == 3 or obj_class == 4 or obj_class == 5:
                obj_class -= 2
            else:
                continue
            polygons.append(poly + [obj_class])

            # plabels.append(labels[i].tolist())
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
        test = 0
    return polygons#, plabels
    # draw bounding boxes

def server_det_masks_demo(result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    assert isinstance(class_names, (tuple, list))
    # img = mmcv.imread(img)
    # img = imgRotation(img)
    # img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    polygons,plabels = [],[]
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]

        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            poly = [int(j) for j in binary_mask_to_polygon(mask)[0]]
            obj_class = int(labels[i])+1
            if obj_class == 3 or obj_class == 4 or obj_class == 5 or obj_class ==6:
                obj_class -= 2
            else:
                continue
            polygons.append(poly + [obj_class])

            # plabels.append(labels[i].tolist())
            # img[mask] = img[mask] * 0.5 + color_mask * 0.5
        test = 0
    return polygons#, plabels

# def server_det_bboxes(bboxes,
#                       labels,
#                       class_names=None,
#                       score_thr=0):  # ,
#     # bbox_color='green',
#     # text_color='green',
#     # thickness=1,
#     # font_scale=0.5,
#     # show=True,
#     # win_name='',
#     # wait_time=0,
#     # out_file=None):
#     """Draw bboxes and class labels (with scores) on an image.
#
#     Args:
#         img (str or ndarray): The image to be displayed.
#         bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
#             (n, 5).
#         labels (ndarray): Labels of bboxes.
#         class_names (list[str]): Names of each classes.
#         score_thr (float): Minimum score of bboxes to be shown.
#         bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
#         text_color (str or tuple or :obj:`Color`): Color of texts.
#         thickness (int): Thickness of lines.
#         font_scale (float): Font scales of texts.
#         show (bool): Whether to show the image.
#         win_name (str): The window name.
#         wait_time (int): Value of waitKey param.
#         out_file (str or None): The filename to write the image.
#     """
#     assert bboxes.ndim == 2
#     assert labels.ndim == 1
#     assert bboxes.shape[0] == labels.shape[0]
#     assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
#     # img = imread(img)
#
#     if score_thr > 0:
#         assert bboxes.shape[1] == 5
#         scores = bboxes[:, -1]
#         inds = scores > score_thr
#         bboxes = bboxes[inds, :]
#         labels = labels[inds]
#
#     return bboxes, labels

    # bbox_color = color_val(bbox_color)
    # text_color = color_val(text_color)
    #
    # for bbox, label in zip(bboxes, labels):
    #     bbox_int = bbox.astype(np.int32)
    #     left_top = (bbox_int[0], bbox_int[1])
    #     right_bottom = (bbox_int[2], bbox_int[3])
    #     cv2.rectangle(
    #         img, left_top, right_bottom, bbox_color, thickness=thickness)
    #     label_text = class_names[
    #         label] if class_names is not None else 'cls {}'.format(label)
    #     if len(bbox) > 4:
    #         label_text += '|{:.02f}'.format(bbox[-1])
    #     cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
    #                 cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

def server_det_bboxes(result, score_thr=0):
    # bbox_color='green',
    # text_color='green',
    # thickness=1,
    # font_scale=0.5,
    # show=True,
    # win_name='',
    # wait_time=0,
    # out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
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

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    # img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    test = 0
    object_coords = []
    for bbox, label in zip(bboxes,labels):
        bbox_int = bbox.astype(np.int32)
        x1 = bbox_int[0]
        y1 = bbox_int[1]
        x2 = bbox_int[2]
        y2 = bbox_int[3]
        obj_class = int(label)+1

        # sandbox label## revise
        if obj_class == 3 or obj_class == 4 or obj_class == 5:
            obj_class -= 2
        else:
            continue
        # object_coords.append(bbox_int.tolist())
        object_coords.append([x1, y1, x2, y1, x2, y2, x1, y2, obj_class])



    return object_coords


