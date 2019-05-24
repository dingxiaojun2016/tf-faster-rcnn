# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from lib.model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  order = scores.ravel().argsort()[::-1]
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  scores = scores[order]

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  """
  Get the scores and bounding boxes
  获取所有anchors“是物体”的概率，接着_region_proposal里面的例子，这里得到的scores应该如下：
  array([[[[0.88079708, 0.88079708],
           [0.88079708, 0.88079708]],
          [[0.88079708, 0.88079708],
           [0.88079708, 0.88079708]]]])
  shape为(1, 2, 2, 2)，即每行代表一个像素点对应的两个anchors“是物体”的概率。
  """
  scores = rpn_cls_prob[:, :, :, num_anchors:]

  """
  scores reshape成1维，scores如下：
  array([0.88079708, 0.88079708, 0.88079708, 0.88079708, 0.88079708, 0.88079708, 0.88079708, 0.88079708])
  """
  scores = tf.reshape(scores, shape=(-1,))

  """
  接着_region_proposal里面的例子，rpn_bbox_pred reshape一下：
  array([[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]])
  shape为(8, 4)，每一行代表一个anchor boxes经过rpn回归之后得到proposal boxes相对于anchor boxes的中心坐标和宽高的偏移量，具体解释见
  bbox_transform中的bbox_transform_inv_tf函数。
  """
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  """
  这里的anchors即为feature map中像素点映射回原始图生成的所有anchors boxes的坐标，shape同rpn_bbox_pred，这里假设，anchors如下：
  array([[ 0,  0,  2,  2],
         [ 4,  4,  6,  6],
         [ 8,  8, 10, 10],
         [12, 12, 14, 14],
         [16, 16, 18, 18],
         [20, 20, 22, 22],
         [24, 24, 26, 26],
         [28, 28, 30, 30]])
  shape同样为(8, 4)，每一行代表一个原始anchor的坐标描述。
  通过计算，proposal boxes的坐标如下：
  array([[ 0.,  0.,  3.,  3.],
         [ 4.,  4.,  7.,  7.],
         [ 8.,  8., 11., 11.],
         [12., 12., 15., 15.],
         [16., 16., 19., 19.],
         [20., 20., 23., 23.],
         [24., 24., 27., 27.],
         [28., 28., 31., 31.]])
  shape为(8, 4)，本来设置偏移量为0的，为何算出来的proposal boxes和anchor boxes有点不一样呢？ 通过分析发现bbox_transform_inv_tf中的
  计算中心坐标过程有点问题。但是可能影响比较小吧。
  """
  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  proposals = clip_boxes_tf(proposals, im_info[:2])

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

  boxes = tf.gather(proposals, indices)
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)

  return blob, scores


