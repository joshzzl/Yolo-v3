#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolo_v3 import Yolo_v3
from core.config import cfg


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.img_size            = 416
        self.strides             = np.array(cfg.YOLO.STRIDES)
        self.output_sizes        = self.img_size // self.strides
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./data/log/train"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size, 3],name='input_data')
            
            self.mask_placeholders = self._build_mask_placeholders(self.output_sizes)

            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = Yolo_v3(inputs=self.input_data, 
                                mask_placeholders=self.mask_placeholders,
                                trainable=self.trainable, 
                                n_classes=self.num_classes, 
                                model_size=(self.trainset.train_input_size, self.trainset.train_input_size))
            self.net_var = tf.global_variables()
            self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.obj_conf_loss + self.no_obj_conf_loss + 0.1 * self.obj_class_loss

        with tf.name_scope('learn_rate'):
            
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        '''
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        gvs = optimizer.compute_gradients(self.loss)
        self.gvs = gvs
        self.train_op = optimizer.apply_gradients(gvs)
        
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        grad_check = tf.check_numerics(clipped_gradients, 'check_numerics caught bad gradients')
        with tf.control_dependencies([grad_check]):
            self.train_op = optimizer.apply_gradients(clipped_gradients)
        '''

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')

                if var_name_mess[2] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()


        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("obj_conf_loss",  self.obj_conf_loss)
            tf.summary.scalar("no_obj_conf_loss",  self.no_obj_conf_loss)
            tf.summary.scalar("obj_class_loss",  self.obj_class_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = "./data/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)


    def _build_mask_placeholders(self, output_sizes):
        with tf.name_scope('define_masks'):
            mask_holders = {}

            mask_holders['noobj_sb'] = tf.placeholder(dtype=tf.float32, shape=[None, output_sizes[0],
                                                    output_sizes[0], 3], name='noobj_mask_sb')
            mask_holders['noobj_mb'] = tf.placeholder(dtype=tf.float32, shape=[None, output_sizes[1],
                                                    output_sizes[1], 3], name='noobj_mask_mb')
            mask_holders['noobj_lb'] = tf.placeholder(dtype=tf.float32, shape=[None, output_sizes[2],
                                                    output_sizes[2], 3], name='noobj_mask_lb')
            return mask_holders


    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables
            
            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                batch_img, label_boxes, boxes, noobj_masks = train_data
                feed_dict = utils.construct_feed_dict(self.mask_placeholders, *noobj_masks)
                feed_dict.update({self.input_data:   batch_img,
                                self.label_sbbox:  label_boxes[0],
                                self.label_mbbox:  label_boxes[1],
                                self.label_lbbox:  label_boxes[2],
                                self.true_sbboxes: boxes[0],
                                self.true_mbboxes: boxes[1],
                                self.true_lbboxes: boxes[2],
                                self.trainable:    True})
                _, summary, train_step_loss, global_step_val, obj_cf_loss, nobj_cf_loss, obj_prob_loss = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step, \
                     self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss],feed_dict=feed_dict)
                #print(type(gradient))

                print("Obj_conf: {}; No_obj_Conf: {}; Prob: {}".format(obj_cf_loss, nobj_cf_loss, obj_prob_loss))

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            for test_data in self.testset:
                batch_img, label_boxes, boxes, noobj_masks = test_data
                feed_dict = utils.construct_feed_dict(self.mask_placeholders, *noobj_masks)
                feed_dict.update({self.input_data:   batch_img,
                                self.label_sbbox:  label_boxes[0],
                                self.label_mbbox:  label_boxes[1],
                                self.label_lbbox:  label_boxes[2],
                                self.true_sbboxes: boxes[0],
                                self.true_mbboxes: boxes[1],
                                self.true_lbboxes: boxes[2],
                                self.trainable:    False})
                test_step_loss = self.sess.run( self.loss, feed_dict=feed_dict)

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)



if __name__ == '__main__': YoloTrain().train()




