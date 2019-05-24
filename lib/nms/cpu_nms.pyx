# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    """cython版本的nms算法，会被转换成c语言版本nms算法，跑在cpu上，具体nms算法介绍见readme。

    Args:
        dets: 需要经过nms处理的boxes，shape为n × 5，其中n代表有几个boxes，5列中1和2列代表boxes左上角坐标的xy，3和4列代表右下角坐标
        xy，第5列表示该boxes的得分。
        thresh: nms阈值

    Returns:
        经过nms之后保留下来的boxes的indexes
    """
    # 获取一下所有boxes的左上角坐标(x1, y1)，右下角坐标(x2, y2)，以及score。
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    # 计算每个boxes的面积
    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 自大到小排序一下scores
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]

    # suppressed 存储被抑制的boxes的indexes
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    # keep 存储保留下来的boxes的indexes
    keep = []

    # 按scores倒序遍历所有的boxes
    for _i in range(ndets):
        i = order[_i]

        # 如果该box已经被抑制，则跳过去
        if suppressed[i] == 1:
            continue

        # 将当前score最大的box的index添加到keep列表中
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]

        # 遍历所有剩下的boxes
        for _j in range(_i + 1, ndets):
            j = order[_j]

            # 如果该box已经被抑制，则跳过去
            if suppressed[j] == 1:
                continue

            # 计算该box与当前score最大的box的IoU值
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)

            # 如果IoU值超过了阈值，则将该box添加到抑制列表里
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
