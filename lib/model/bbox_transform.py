# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
bounding-boxes模块，包含boxes与ground truth的偏移量计算；以及由boxes和偏移量计算predict boxes；还有限制boxes在图片尺寸内的计算。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def bbox_transform(ex_rois, gt_rois):
  """根据ex_rois boxes和ground truth boxes来计算偏移量，偏移量解释见bbox_transform_inv_tf函数。

  Args:
    ex_rois: ex_rois boxes
    gt_rois: ground truth boxes

  Returns:
    偏移量
  """
  # 获取一下ex_rois的宽高、中心点坐标
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  # 获取一下ground truth boxes的宽高，中心点坐标
  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  # 计算一下ex_rois和ground truth boxes的偏移量，偏移量的具体含义见bbox_transform_inv_tf函数
  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets


def bbox_transform_inv(boxes, deltas):
  """根据boxes和偏移量deltas来计算predict boxes，同bbox_transform正好是个反过程。

  Args:
    boxes: boxes
    deltas: 偏移量

  Returns:
    预测边框
  """
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  boxes = boxes.astype(deltas.dtype, copy=False)
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]
  
  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]

  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes


def clip_boxes(boxes, im_shape):
  """将boxes边框限制在image尺寸内。

  Args:
    boxes: boxes
    im_shape: image的尺寸描述

  Returns:
    限制在image尺寸内的boxes
  """

  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
  return boxes


def bbox_transform_inv_tf(boxes, deltas):
  """ bbox_transform_inv的tensorflow版本。
  根据boxes的原始坐标和偏移量deltas，来计算predict boxes。
  boxes和deltas和predict boxes之间的关系如下：
  假设boxes中心的坐标为(xa, ya)，宽高为wa，ha。
  假设predict boxes中心的坐标为(xp, yp)，宽高为wp，hp。
  则公式如下：
  dx = (xp - xa) / wa
  dy = (yp - ya) / ha
  dw = log(wp / wa)
  dh = log(hp / ha)

  Args:
    boxes: boxes
    deltas: 偏移量deltas

  Returns:
    返回predict boxes的左上坐标，和右下坐标。
  """
  boxes = tf.cast(boxes, deltas.dtype)

  # 计算boxes的宽高
  widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
  heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0


  """
  计算boxes的中心坐标，但是这里计算出来的中心坐标有问题？是不是应该下边这样？
  ctr_x = tf.add(boxes[:, 0], (widths - 1) * 0.5)
  ctr_y = tf.add(boxes[:, 1], (heights - 1) * 0.5)
  """
  # ctr_x = tf.add(boxes[:, 0], widths * 0.5)
  # ctr_y = tf.add(boxes[:, 1], heights * 0.5)
  ctr_x = tf.add(boxes[:, 0], (widths - 1) * 0.5)
  ctr_y = tf.add(boxes[:, 1], (heights - 1) * 0.5)

  # 获取坐标、宽高的偏移量
  dx = deltas[:, 0]
  dy = deltas[:, 1]
  dw = deltas[:, 2]
  dh = deltas[:, 3]

  # 根据转换公式，通过偏移量和boxes中心坐标和宽高计算predict boxes的中心坐标和宽高
  pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
  pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
  pred_w = tf.multiply(tf.exp(dw), widths)
  pred_h = tf.multiply(tf.exp(dh), heights)

  # 根据predict boxes的中心坐标和宽高来计算predict boxes的左上和右下坐标
  pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
  pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
  pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
  pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

  # 将坐标值合成到一个维度中
  return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes_tf(boxes, im_info):
  """clip_boxes的tensorflow版本"""
  b0 = tf.maximum(tf.minimum(boxes[:, 0], im_info[1] - 1), 0)
  b1 = tf.maximum(tf.minimum(boxes[:, 1], im_info[0] - 1), 0)
  b2 = tf.maximum(tf.minimum(boxes[:, 2], im_info[1] - 1), 0)
  b3 = tf.maximum(tf.minimum(boxes[:, 3], im_info[0] - 1), 0)
  return tf.stack([b0, b1, b2, b3], axis=1)


