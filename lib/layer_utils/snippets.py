# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from lib.layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  """
  根据feature map的尺寸来计算该feature map所有点对应到原始图片上的所有anchors。每个对应的anchors的个数由anchor_scales和
  anchor_ratios决定，比如anchor_scales有8,16,32三种扩大倍数，anchor_ratios有0.5,1,2三种纵横比，则对应feature map每个点，映射回原
  始图片的感受野后的中心点，会生成3 × 3中组合，比如0.5纵横比会组合8,16,32三种扩大倍数，所以对于每个点有9个anchors。
  Args:
    height: feature map的高
    width: feature map的宽
    feat_stride: 从原始图片到该feature map总共缩小的倍数，由每层的stride决定
    anchor_scales: anchors的扩大倍数
    anchor_ratios: anchors的纵横比

  Returns:
    feature map所有点对应到原始图中的anchors
  """

  """
  首先获取feature map所有点对应到原始图片的点的坐标，x原始 = x(feature map) * feat_stride y原始 = x(feature map) * feat_stride.
  比如3 × 3的feature map对应到原始图片中所有的坐标点shifts如下：
  Out[133]:
  array([[[ 0,  0,  0,  0]],
         [[16,  0, 16,  0]],
         [[32,  0, 32,  0]],
         [[ 0, 16,  0, 16]],
         [[16, 16, 16, 16]],
         [[32, 16, 32, 16]],
         [[ 0, 32,  0, 32]],
         [[16, 32, 16, 32]],
         [[32, 32, 32, 32]]], dtype=int32)
  能看到第一列和第三列一致，第二列和第四列一致。一二列就组成了所有原始图中的点，增加三四列是为了方便计算anchor，因为一个anchor用左上角坐标
  和右下角坐标来表示
  """
  shift_x = tf.range(width) * feat_stride # width
  shift_y = tf.range(height) * feat_stride # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))


  """
  依据一个16 × 16的base anchor生成9个anchors（由scales和ratios组成），anchor_constant如下：
  array([[ -84.,  -40.,   99.,   55.],    0.5纵横比，8倍扩大
         [-176.,  -88.,  191.,  103.],    0.5纵横比，16倍扩大
         [-360., -184.,  375.,  199.],    0.5纵横比，32倍扩大
         [ -56.,  -56.,   71.,   71.],    1.0纵横比，8倍扩大
         [-120., -120.,  135.,  135.],    1.0纵横比，16倍扩大
         [-248., -248.,  263.,  263.],    1.0纵横比，32倍扩大
         [ -36.,  -80.,   51.,   95.],    2.0纵横比，8倍扩大
         [ -80., -168.,   95.,  183.],    2.0纵横比，16倍扩大
         [-168., -344.,  183.,  359.]])   2.0纵横比，32倍扩大
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)


  """
  让shifts与anchor_constant相加组成feature map对应到原始图片中的anchors。
  在这里发现上边生成的shifts实际并非anchors的中心点。而是中心点偏左上一些的点。
  """
  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  
  return tf.cast(anchors_tf, dtype=tf.float32), length
