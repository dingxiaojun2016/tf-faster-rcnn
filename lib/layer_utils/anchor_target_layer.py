# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.model.config import cfg
import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps
from lib.model.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  # 每个像素点对应anchors 个数
  A = num_anchors

  # 总共anchors个数
  total_anchors = all_anchors.shape[0]

  # 总共像素点个数
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  # feature map的高和宽
  height, width = rpn_cls_score.shape[1:3]

  # only keep anchors inside the image
  # 获取所有在图片尺寸内的anchors的indexes
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  # 只保留在图片内的anchors
  anchors = all_anchors[inds_inside, :]

  """
  label: 1 is positive, 0 is negative, -1 is dont care
  labels用来保存anchors的标签，1代表是正标签，即该anchor是正样本，即存在ground truth 和该anchor的IoU满足rpn正样本的要求；0代表负样本
  ；-1代表不关心该样本。正样本指当做前景物体的boxes，负样本指代表背景的boxes。
  关于rpn的正负样本的划分，如下描述：
  考察训练集中的每张图像（含有人工标定的ground true box）的所有anchors。
  a. 对每个标定的ground true box区域，与其重叠比例最大的anchor记为 正样本 (保证每个ground true 至少对应一个正样本anchor)
  b. 对a)剩余的anchor，如果其与某个标定区域重叠比例大于0.7，记为正样本（每个ground true box可能会对应多个正样本anchor。但每个正样本
  anchor 只可能对应一个grand true box）；如果其与任意一个标定的重叠比例都小于0.3，记为负样本。
  c. 对a),b)剩余的anchor，弃去不用。
  d. 跨越图像边界的anchor弃去不用
  """
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)

  """
  overlaps between the anchors and the gt boxes
  overlaps (ex, gt)
  计算anchors boxes和groud_truth boxes的IoU，overlaps为二维，每行为一个anchors与所有gt boxes之间IoU大小。
  假设overlaps值如下：
  array([[0.48557798, 0.12723372, 0.22120988, 0.93015104],
         [0.41377397, 0.91378704, 0.10855037, 0.19918476],
         [0.86859926, 0.21069385, 0.48224216, 0.63019202],
         [0.38623382, 0.62772807, 0.91166306, 0.91971408],
         [0.25904543, 0.31076955, 0.5593479 , 0.41681275],
         [0.58562328, 0.18718799, 0.28859296, 0.57199318],
         [0.35817963, 0.94432939, 0.80717617, 0.84104114],
         [0.17751499, 0.42397581, 0.40260994, 0.10210093]])
  shape为(8, 4)，即一副图片总共有8个anchors boxes和4个ground truth boxes。
  """
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))

  """
  找到每个anchors boxes IoU最大的gt boxes
  argmax_overlaps为array([3, 1, 0, 3, 2, 0, 1, 1])
  """
  argmax_overlaps = overlaps.argmax(axis=1)

  """
  每行最大的IoU组成max_overlaps数组
  max_overlaps为：
  array([0.93015104, 0.91378704, 0.86859926, 0.91971408, 0.5593479 , 0.58562328, 0.94432939, 0.42397581])
  """
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

  """
  找到每个gt boxes IoU最大的anchors boxes
  gt_argmax_overlaps为：
  array([2, 6, 3, 0])
  """
  gt_argmax_overlaps = overlaps.argmax(axis=0)

  """
  每列最大的IoU组成gt_max_overlaps数组
  gt_max_overlaps为：
  array([0.86859926, 0.94432939, 0.91166306, 0.93015104])
  """
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]

  """
  根据上述正样本划分的条件a，所以每个ground truth box至少得到一个anchor，gt_argmax_overlaps就是行号，就是对应的anchor，为：
  array([0, 2, 3, 6])。
  """
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    """
    assign bg labels first so that positive labels can clobber them
    first set the negatives
    如果某个anchor最大IoU小于negative阈值，则先设置为负样本。
    """
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  """
  fg label: for each gt, anchor with highest overlap
  满足条件a的anchors的labels设置为1。
  """
  labels[gt_argmax_overlaps] = 1

  """
  fg label: above threshold IOU
  根据上述正样本划分的条件b，剩余的anchors中，如果存在gt boxes与它的IoU大于阈值，则是正样本。
  """
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  """
  subsample positive labels if we have too many
  如果正样本个数超过限定值则随机选择特定个。
  """
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  """
  subsample negative labels if we have too many
  如果负样本个数超过限定值，则随机选择特定个。
  """
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1


  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  """
  bbox_targets是anchors boxes和ground truth的偏移量，偏移量见lib.model.bbox_transorm.bbox_transform_inv_tf函数解释。
  该偏移量就是rpn的Bounding-box regression学习的目标。
  """
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  """
  因为前边把不在图片尺寸内的anchors给删除了，这里再把labels扩充到原始anchors的个数，用-1来扩充。
  后边bbox_targets、bbox_inside_weights、bbox_outside_weights同理。
  """
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  """
  假设labels为：
  array([0, 1, 0, 1, 0, 1, 0, 1])
  假设feature map有两行两列，每个像素点有两个anchors。
  经过转置和reshape，rpn_labels为：
  array([[[[0, 0],    ---feature map第一行两个像素点对应的第一个anchor的label
           [0, 0],    ---feature map第二行两个像素点对应的第一个anchor的label
           [1, 1],    ---feature map第一行两个像素点对应的第二个anchor的label
           [1, 1]]]]) ---feature map第二行两个像素点对应的第二个anchor的label
  shape为(1, 4, 2, 2)
  """
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
