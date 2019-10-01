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
import math
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

            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        self.model = Yolo_v3(inputs=self.input_data, 
                                trainable=self.trainable, 
                                n_classes=self.num_classes, 
                                model_size=(self.trainset.train_input_size, self.trainset.train_input_size))
        
        with tf.name_scope("define_loss"):
            
            self.net_var = tf.global_variables()
            '''
            self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss, self.obj_loc_loss, \
                self.mid_iou, self.areas, self.enc_areas = \
                                                self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)'''
            self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss, self.obj_loc_loss = \
                    self.model.compute_loss(self.label_sbbox,  self.label_mbbox,  self.label_lbbox,\
                                            self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)        
            self.loss = self.obj_conf_loss + self.no_obj_conf_loss + self.obj_class_loss + self.obj_loc_loss

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
        
        
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')

                if var_name_mess[1] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
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
            tf.summary.scalar("obj_location_loss", self.obj_loc_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = './data/log/'+self.time+'/'
            if not os.path.exists(logdir):
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
            #self.first_stage_epochs = 0

        
        ckpt_dir = './checkpoint/'+self.time+'/'
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables
            
            #train_op = self.train_op

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []
            batch_num = 0

            for train_data in pbar:
                batch_img, label_boxes, boxes, _ = train_data
                feed_dict = {}
                feed_dict.update({self.input_data:   batch_img,
                                self.label_sbbox:  label_boxes[0],
                                self.label_mbbox:  label_boxes[1],
                                self.label_lbbox:  label_boxes[2],
                                self.true_sbboxes: boxes[0],
                                self.true_mbboxes: boxes[1],
                                self.true_lbboxes: boxes[2],
                                self.trainable:    True})
                '''
                _, summary, train_step_loss, global_step_val, \
                    obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss,\
                    mid_iou, areas, enc_areas = self.sess.run( \
                        [train_op, self.write_op, self.loss, self.global_step, \
                         self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss, self.obj_loc_loss,\
                         self.mid_iou, self.areas, self.enc_areas],feed_dict=feed_dict)'''
                _, summary, train_step_loss, global_step_val, \
                    obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss \
                    = self.sess.run([train_op, self.write_op, self.loss, self.global_step, \
                         self.obj_conf_loss, self.no_obj_conf_loss, self.obj_class_loss, self.obj_loc_loss],\
                         feed_dict=feed_dict)
                #print(type(gradient))
                #if epoch <= self.first_stage_epochs:
                #    train_step_loss += min(0.01 * obj_loc_loss, 100.)
                #print("Obj: {0:.2f}; No_obj: {1:.2f}; Prob: {2:.2f}; Loc: {3:.2f}"\
                #    .format(obj_cf_loss, nobj_cf_loss, 0.05*obj_prob_loss, 0.01*obj_loc_loss))

                if math.isnan(train_step_loss):
                    print("Train: Epoch {4}-Batch {5}; Obj: {0:.2f}; No_obj: {1:.2f}; Prob: {2:.2f}; Loc: {3:.2f}"\
                    .format(obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss, epoch, batch_num))
                    #self._print_err_msg(mid_iou, areas, enc_areas)
                    #raise ValueError('Nan value for loss')

                batch_num += 1
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("Loss: {4:.2f}; Obj: {0:.2f}; No_obj: {1:.2f}; Prob: {2:.2f}; Loc: {3:.2f}"\
                    .format(obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss, train_step_loss))

            batch_num = 0
            for test_data in self.testset:
                batch_img, label_boxes, boxes, _ = test_data
                feed_dict = {}
                feed_dict.update({self.input_data:   batch_img,
                                self.label_sbbox:  label_boxes[0],
                                self.label_mbbox:  label_boxes[1],
                                self.label_lbbox:  label_boxes[2],
                                self.true_sbboxes: boxes[0],
                                self.true_mbboxes: boxes[1],
                                self.true_lbboxes: boxes[2],
                                self.trainable:    False})
                '''
                obj_cf_loss, nobj_cf_loss, obj_prob_loss, \
                    obj_loc_loss, test_step_loss, \
                    mid_iou, areas, enc_areas = self.sess.run([self.obj_conf_loss, self.no_obj_conf_loss, \
                                                self.obj_class_loss, self.obj_loc_loss,\
                                                self.loss, self.mid_iou, self.areas, self.enc_areas], \
                                                feed_dict=feed_dict)'''
                obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss, test_step_loss \
                     = self.sess.run([self.obj_conf_loss, self.no_obj_conf_loss, \
                                      self.obj_class_loss, self.obj_loc_loss, self.loss], \
                                      feed_dict=feed_dict)
                #if epoch <= self.first_stage_epochs:
                #    test_step_loss += 0.01 * obj_loc_loss
                if math.isnan(test_step_loss):
                    print("Test: Epoch {4}-Batch {5}; Obj: {0:.2f}; No_obj: {1:.2f}; Prob: {2:.2f}; Loc: {3:.2f}"\
                    .format(obj_cf_loss, nobj_cf_loss, obj_prob_loss, obj_loc_loss, epoch, batch_num))
                    #self._print_err_msg(mid_iou, areas, enc_areas)

                    #raise ValueError('Nan value for test loss')
                batch_num += 1
                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = ckpt_dir + ("yolov3_test_loss=%.4f.ckpt" % test_epoch_loss)
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


    def _print_err_msg(self, mid_iou, areas, enc_areas):
        for i in range(3):
            mid_iou_has_nan = np.isnan(mid_iou[i]).any()
            enc_has_nan = np.isnan(enc_areas[i]).any()
            areas_has_nan = [np.isnan(ar).any() for ar in areas[i]]
            print("Mid IOU has NaN: {}".format(mid_iou_has_nan))
            print("Enclose area has NaN: {}".format(enc_has_nan))
            print("Box1: {0}; Box2: {1}; inter: {2}; union: {3}".format(*areas_has_nan))

if __name__ == '__main__': YoloTrain().train()




