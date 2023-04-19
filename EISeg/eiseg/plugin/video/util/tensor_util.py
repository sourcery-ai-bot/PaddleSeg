# The video propagation and fusion code was heavily based on https://github.com/hkchengrex/MiVOS
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/hkchengrex/MiVOS/blob/main/LICENSE

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.vision import transforms

im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inv_im_trans = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])


def images_to_paddle(frames):
    frames = paddle.to_tensor(frames.transpose([0, 3, 1, 2])).astype(
        'float32').unsqueeze(0) / 255
    b, t, c, h, w = frames.shape
    for ti in range(t):
        frames[0, ti] = im_normalization(frames[0, ti])
    return frames


def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).astype('float32').sum()
    union = (seg | gt).astype('float32').sum()

    return intersection, union


def compute_np_iu(seg, gt):
    intersection = (seg & gt).astype(np.float32).sum()
    union = (seg | gt).astype(np.float32).sum()

    return intersection, union


def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    return (intersection + 1e-6) / (union + 1e-6)


def compute_np_iou(seg, gt):
    intersection, union = compute_np_iu(seg, gt)
    return (intersection + 1e-6) / (union + 1e-6)


def compute_multi_class_iou(seg, gt):
    # seg -> k*h*w
    # gt -> k*1*h*w
    num_classes = gt.shape[0]
    pred_idx = paddle.argmax(seg, axis=0)
    iou_sum = sum(
        compute_tensor_iou(pred_idx == (ki + 1), gt[ki, 0] > 0.5)
        for ki in range(num_classes)
    )
    return (iou_sum + 1e-6) / (num_classes + 1e-6)


def compute_multi_class_iou_idx(seg, gt):
    # seg -> h*w
    # gt -> k*h*w
    num_classes = gt.shape[0]
    iou_sum = sum(
        compute_np_iou(seg == (ki + 1), gt[ki] > 0.5)
        for ki in range(num_classes)
    )
    return (iou_sum + 1e-6) / (num_classes + 1e-6)


def compute_multi_class_iou_both_idx(seg, gt):
    # seg -> h*w
    # gt -> h*w
    num_classes = gt.max()
    iou_sum = sum(
        compute_np_iou(seg == ki, gt == ki) for ki in range(1, num_classes + 1)
    )
    return (iou_sum + 1e-6) / (num_classes + 1e-6)


# STM
def pad_divide_by(in_img, d, in_size=None):
    h, w = in_img.shape[-2:] if in_size is None else in_size
    new_h = h + d - h % d if h % d > 0 else h
    new_w = w + d - w % d if w % d > 0 else w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = lw, int(uw), lh, int(uh)

    if len(in_img.shape) == 5:
        N, B, C, H, W = in_img.shape
        in_img = in_img.reshape([-1, C, H, W])
        out = F.pad(in_img, pad_array, data_format='NCHW').unsqueeze(0)
    else:
        out = F.pad(in_img, pad_array, data_format='NCHW')
    return out, pad_array


def unpad(img, pad):
    if pad[2] + pad[3] > 0:
        img = img[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        img = img[:, :, :, pad[0]:-pad[1]]
    return img


def unpad_3dim(img, pad):
    if pad[2] + pad[3] > 0:
        img = img[:, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        img = img[:, :, pad[0]:-pad[1]]
    return img
