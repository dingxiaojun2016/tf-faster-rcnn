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
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from lib.layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from lib.layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from lib.layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from lib.layer_utils.anchor_target_layer import anchor_target_layer
from lib.layer_utils.proposal_target_layer import proposal_target_layer
from lib.utils.visualization import draw_bounding_boxes

from lib.model.config import cfg

class Network(object):
  def __init__(self):
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image + cfg.PIXEL_MEANS
    # BGR to RGB (opencv uses BGR)
    resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
    self._gt_image = tf.reverse(resized, axis=[-1])

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    if self._gt_image is None:
      self._add_gt_image()

    # 为当前训练的图片绘制ground truth boxes
    image = tf.py_func(draw_bounding_boxes, 
                      [self._gt_image, self._gt_boxes, self._im_info],
                      tf.float32, name="gt_boxes")

    # 将绘制好gt boxes的图片增加到tensorboard的summary中
    return tf.summary.image('GROUND_TRUTH', image)

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                      tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)

      # softmax需要针对几类，所以把bottom reshape成2维，并且第二维是2，代表有2类，第一维代表有多少组数据需要分类，即总共的anchors的数量。
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)

      # 恢复一下shape
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores = proposal_top_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_top_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal_top")
        
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      if cfg.USE_E2E_TF:
        rois, rpn_scores = proposal_layer_tf(
          rpn_cls_prob,
          rpn_bbox_pred,
          self._im_info,
          self._mode,
          self._feat_stride,
          self._anchors,
          self._num_anchors
        )
      else:
        rois, rpn_scores = tf.py_func(proposal_layer,
                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                               self._feat_stride, self._anchors, self._num_anchors],
                              [tf.float32, tf.float32], name="proposal")

      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])

      """
      Get the normalized coordinates of bounding boxes
      获取bounding boxes标准化后的坐标，标准化过程见crop_and_resize中boxes参数的解释。
      """
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

      """
      Won't be back-propagated to rois anyway, but to save time
      意思是在训练fast r-cnn时，rpn不再更新参数？
      """
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))

      """
      依据标准化后的坐标进行裁剪，并且resize到14*14
      """
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    # 增加一层max pool
    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    """生成anchors boxes的labels以及rpn bounding-box regression的学习目标。

    label: 1 is positive, 0 is negative, -1 is dont care
    labels用来保存anchors的标签，1代表是正标签，即该anchor是正样本，即存在ground truth 和该anchor的IoU满足rpn正样本的要求；0代表负样
    本；-1代表不关心该样本。正样本指当做前景物体的boxes，负样本指代表背景的boxes。

    rpn的bounding box regression的回归是需要将“predict boxes与anchors的偏移量rpn_bbox_pred”和“anchors boxes和ground truth
    boxes之间的偏移量rpn_bbox_targets”进行接近

    Args:
      rpn_cls_score: 只用来传递feature map的shape
      name: scope name

    Returns:
      anchors boxes的labels
    """
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      # rpn_bales用来保存anchors的labels，labels代表anchor是正样本还是负样本。
      rpn_labels.set_shape([1, 1, None, None])

      # rpn_bbox_targets用来保存anchors boxes和ground truth boxes的偏移量，他是rpn的bbox-regression的学习目标。
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    """按规则选取一些proposal boxes，并生成r-cnn bounding boxes regression的学习目标。

    r-cnn bounding boxes regression的学习目标是需要将”proposal boxes与predict boxes的偏移量bbox_pred“与”proposal boxes与ground
    truth boxes的偏移量“进行接近。

    Args:
      rois: fast r-cnn的proposal boxes。
      roi_scores: ros_scores用来保存proposal boxes的“是物体”的概率。
      name: scope name

    Returns:
      按规则选取后的rois和roi_scores。
    """
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
        proposal_target_layer,
        [rois, roi_scores, self._gt_boxes, self._num_classes],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      """
      just to get the shape right. Get the feature map(block3-out) shape.
      根据总stride（conv1->block3-out）16，计算feature map的宽高
      总stride(16) = conv1-stride(2) * pool1-stride(2) * block1-stride(2) * block2-stride(2) * block3-stride(1)
      比如224 / 16 = 14
      """
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))

      # 如果是端到端模型
      if cfg.USE_E2E_TF:
        # 为当前最后一层的feature map中的所有像素点（像素点的选取顺序是从第一行开始自左到右的方向，处理完一行后自上到下进入下一行）生成映
        # 射回原始图中的所有anchors。
        anchors, anchor_length = generate_anchors_pre_tf(
          height,
          width,
          self._feat_stride,
          self._anchor_scales,
          self._anchor_ratios
        )
      else:
        anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                            [height, width,
                                             self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                            [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    # 使用特定的卷积网络对image进行特征提取，生成feature maps。
    net_conv = self._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      """
      build the anchors for the image
      根据feature map和strides为image生成anchors
      """
      self._anchor_component()

      """
      region proposal network
      构造rpn(region proposal network)网络，rois是rpn网络输出的proposal boxes。
      """
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        # 依据proposal boxes的坐标对net_conv的feature maps进行裁剪和resize成14*14
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError

    # 在pool5后增加特定的卷积和pool层，生成full-connected7。
    fc7 = self._head_to_tail(pool5, is_training)
    with tf.variable_scope(self._scope, self._scope):

      """
      region classification
      根据fc7层，生成softmax分类器和bbox回归器
      """
      cls_prob, bbox_pred = self._region_classification(fc7, is_training,
                                                        initializer, initializer_bbox)

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    """计算bbox_pred和bbox_targets的smooth_l1_loss。
    bbox_pred和bbox_targets的shape为(1, feature map的高, feature map的宽, feature map每个像素点对应的anchors个数)

    Args:
      bbox_pred: anchors boxes-predict boxes的偏移量
      bbox_targets: anchors boxes-ground truth boxes的偏移量
      bbox_inside_weights: smooth_l1_loss的参数
      bbox_outside_weights: smooth_l1_loss的参数
      sigma: smooth_l1_loss的参数
      dim: 求平均值的维度

    Returns:
      smooth_l1_loss
    """
    sigma_2 = sigma ** 2

    # 两者差值是smooth_l1_loss的输入
    box_diff = bbox_pred - bbox_targets

    # bbox_inside_weights对于fg来说为1，其他为0
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    # 这里stop_gradient是什么作用？
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      """
      RPN, class loss
      利用rpn的分类logits和rpn labels来构成rpn分类损失函数
      """
      # 将rpn的分类logits和分类标签rpn_label reshape成同样的一维长度，代表anchors的个数。
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      # 选取正负样本
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      # 由rpn的分类logits层rpn_cls_score和分类标签rpn_label构成交叉熵losses作为分类损失。
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      """
      RPN, bbox loss
      利用“rpn的anchors boxes和predict boxes的偏移量rpn_bbox_pred”与“rpn的anchors boxes和ground truth boxes的偏移量
      rpn_bbox_targets”，构成rpn的bounding boxes回归损失。
      """
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      """
      RCNN, class loss
      利用r-cnn的分类器logits和proposal boxes的labels来构成r-cnn的分类损失函数
      """
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"], [-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

      """
      RCNN, bbox loss
      利用”r-cnn的proposal boxes和predict boxes的偏移量bbox_pred“和”r-cnn的proposal boxes与ground truth boxes的偏移量
      bbox_targets“，构成r-cnn的bounding boxes回归损失。
      """
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
      self._losses['total_loss'] = loss + regularization_loss

      self._event_summaries.update(self._losses)

    return loss

  def _region_proposal(self, net_conv, is_training, initializer):
    # 增加一个rpn卷积层
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)

    """
    使用kernel size为1的卷积层来当做全连接层，全连接层的输出为anchors个数的2倍，即代表一个anchor可以是一个物体或者不是一个物体两类。
    用于分类学习。
    下边以一个例子来分析后边每一步的变化，假设rpn_cls_score为如下值：
    array([[[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7]],
            [[ 8,  9, 10, 11],
             [12, 13, 14, 15]]]])
    shape为(1, 2, 2, 4)，这里的每个点的anchor个数为2，总共有4个像素点，以第一行为例，第一行就代表第一个像素点对应的2个anchors“是物体”
    和“非物体”的logits，即0和1代表两个anchors是“非物体”的logit，2和3代表两个anchors“是物体”的logit。
    """
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')

    """
    1)
    change it so that the score has 2 as its channel size
    接着上边的例子，rpn_cls_score_reshape为：
    array([[[[ 0,  2],
             [ 4,  6]],
            [[ 8, 10],
             [12, 14]],
            [[ 1,  3],
             [ 5,  7]],
            [[ 9, 11],
             [13, 15]]]])
    shape为(1, 4, 2, 2)，看到reshape过后每一行代表一个anchor“是物体”和“非物体”的logits。
    """
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')

    """
    2)
    为2分类增加softmax层，生成概率。
    添加softmax层后shape没有变化，rpn_cls_prob_reshape为：
    array([[[[0.11920292, 0.88079708],
             [0.11920292, 0.88079708]],
            [[0.11920292, 0.88079708],
             [0.11920292, 0.88079708]],
            [[0.11920292, 0.88079708],
             [0.11920292, 0.88079708]],
            [[0.11920292, 0.88079708],
             [0.11920292, 0.88079708]]]])
    shape为(1, 4, 2, 2)，看到每一行代表一个anchor“是物体”和“非物体”的概率。
    比如[0, 2]softmax后就是[0.11920292, 0.88079708]，计算过程：
    p(非物体) = e^0 / (e^0 + e^2)
    p(是物体) = e^2 / (e^0 + e^2)
    """
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")

    """
    3)
    为2分类softmax层生成prediction。
    根据上述概率求出分类预测，rpn_cls_pred为：
    array([1, 1, 1, 1, 1, 1, 1, 1])
    就是说从上述概率的表述来看，预测每个anchor“是物体”还是“非物体”，当然是概率大的预测。
    """
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")

    """
    4)
    对第二步生成的概率，再转一下，转成如下格式，rpn_cls_prob为
    array([[[[0.11920292, 0.11920292, 0.88079708, 0.88079708],
             [0.11920292, 0.11920292, 0.88079708, 0.88079708]],
            [[0.11920292, 0.11920292, 0.88079708, 0.88079708],
             [0.11920292, 0.11920292, 0.88079708, 0.88079708]]]])
    shape为(1, 2, 2, 4)，这里的概率对应到最初始的rpn_cls_score，以第一行为例，第一行就代表第一个像素点对应的2个anchors是物体和非物体
    的概率，其中第1,2列分别为该像素点两个anchor是“非物体”的概率，3，4列分别为该像素点两个anchor是“是物体”的概率。
    """
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

    """
    增加一个全连接层，输出是anchors个数的4倍，代表一个predict boxes相对于anchor boxes的中心坐标和宽高的偏移量，用于回归学习，具体偏移
    量的解释见bbox_transform中的bbox_transform_inv_tf函数。
    这里假设rpn_bbox_pred的值如下：
    array([[[[ 0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0]],
            [[ 0,  0,  0,  0,  0,  0,  0,  0],
             [ 0,  0,  0,  0,  0,  0,  0,  0]]]])
    shape为(1, 2, 2, 8)，即feature map总共有4个像素点，每个像素点有两个anchors，每行代表某个像素点两个anchors经过rpn回归之后得到的
    predict boxes相对于anchor boxes的中心坐标和宽高的偏移量。这里为了简单，假设偏移量都为0，即代表predict boxes和anchors boxes完
    全一致，第一行的前4个0代表feature map第一个像素点的第一个anchor boxes，经过rpn回归后得到predict boxes与anchor boxes中心坐标偏
    移量和宽高偏移量。以此类推。
    """
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      """
      rois用来保存predict boxes的边框描述。
      shape为(anchors个数, 5)
      ros_scores用来保存predict boxes的“是物体”的概率。
      shape为(anchors个数, 1)
      """
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")

      """
      rpn_labels用来保存anchors的labels，labels代表anchor是正样本还是负样本。
      shape为(1, 1, 每一个像素点anchors个数*feature map高度, feature map的宽度)
      另外_anchor_target_layer还计算了anchors boxes和ground truth boxes之间的偏移量rpn_bbox_targets，我们应该知道rpn的bounding
      box regression的回归是需要将“predict boxes与anchors的偏移量rpn_bbox_pred”和“anchors boxes和ground truth boxes之间的偏移
      量rpn_bbox_targets”进行接近。
      """
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        """
        上述rpn的predict boxes作为fast r-cnn的proposal boxes。
        按规则选取一些proposal boxes，并生成r-cnn bounding boxes regression的学习目标。
        """
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
    """根据fc7层，生成softmax分类器和bbox回归器

    Args:
      fc7: fc7层
      is_training: 是否是training
      initializer: 权重初始化器
      initializer_bbox: bounding box初始化器

    Returns:
      softmax分类器和bbox回归器
    """
    cls_score = slim.fully_connected(fc7, self._num_classes, 
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       activation_fn=None, scope='cls_score')
    cls_prob = self._softmax_layer(cls_score, "cls_prob")
    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     activation_fn=None, scope='bbox_pred')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    # im_info保存当前训练图片的高宽和被缩放比例
    self._im_info = tf.placeholder(tf.float32, shape=[3])

    # ground truth 最后一列代表类别。
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    # anchor and ratios combined to recommend boxes.
    self._num_anchors = self._num_scales * self._num_ratios

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)):
      """
      构建完整的faster r-cnn网络模型。
      rois为rpn输出的proposal boxes。
      cls_prob为r-cnn分类器输出，保存proposal boxes对应的分类标签。
      bbox_pred为r-cnn bounding boxes回归器输出，保存proposal boxes与预测boxes之间的偏移量。
      rpn用来给r-cnn提供proposal boxes，r-cnn则对这些proposal boxes进行分类和边框回归。
      """
      rois, cls_prob, bbox_pred = self._build_network(training)

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if testing:
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      # 计算rpn和r-cnn的losses
      self._add_losses()
      layers_to_output.update(self._losses)

      val_summaries = []
      with tf.device("/cpu:0"):
        # 增加图片summaries
        val_summaries.append(self._add_gt_image_summary())

        # 增加losses scalar summaries
        for key, var in self._event_summaries.items():
          val_summaries.append(tf.summary.scalar(key, var))

        # 增加scores histogram summaries
        for key, var in self._score_summaries.items():
          self._add_score_summary(key, var)

        # 增加act histogram summaries和act零占比的scalars
        for var in self._act_summaries:
          self._add_act_summary(var)

        # 增加所有trainable的变量的histogram的summaries
        for var in self._train_summaries:
          self._add_train_summary(var)

      self._summary_op = tf.summary.merge_all()
      self._summary_op_val = tf.summary.merge(val_summaries)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}

    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                     self._predictions['cls_prob'],
                                                     self._predictions['bbox_pred'],
                                                     self._predictions['rois']],
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                 self._losses['rpn_loss_box'],
                                                                                 self._losses['cross_entropy'],
                                                                                 self._losses['loss_box'],
                                                                                 self._losses['total_loss'],
                                                                                 self._summary_op,
                                                                                 train_op],
                                                                                feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

