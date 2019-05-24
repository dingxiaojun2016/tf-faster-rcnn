# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """纯python版本的nms算法，具体nms算法介绍见readme。

    Args:
        dets: 需要经过nms处理的boxes，shape为n × 5，其中n代表有几个boxes，5列中1和2列代表boxes左上角坐标的xy，3和4列代表右下角坐标
        xy，第5列表示该boxes的得分。
        thresh: nms阈值

    Returns:
        经过nms之后保留下来的boxes的indexes
    """
    # 获取一下所有boxes的左上角坐标(x1, y1)，右下角坐标(x2, y2)，以及score。
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # 计算每个boxes的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 自大到小排序一下scores
    order = scores.argsort()[::-1]

    # keep用来记录经过nms之后需要保留下来的boxes的index
    keep = []
    while order.size > 0:
        i = order[0]
        # 当前最大的score的index放到keep中
        keep.append(i)

        # 计算所有非最大score的boxes和最大score的box的交集区域左上和右下坐标。
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算所有交集面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的boxes的indexes
        inds = np.where(ovr <= thresh)[0]

        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来。
        # 在保留的boxes中继续进行迭代。
        order = order[inds + 1]

    return keep
