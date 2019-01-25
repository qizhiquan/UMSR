#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import UMSR_g, Vgg19_simple_api
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
beta2 = config.TRAIN.beta2
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init

lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 30, 30, 3], name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 120, 120, 3], name='t_target_image')

    net_g = UMSR_g(t_image, is_train=True, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb1, vgg_target_emb2, vgg_target_emb3, vgg_target_emb4, vgg_target_emb5 = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb1, vgg_predict_emb2, vgg_predict_emb3, vgg_predict_emb4, vgg_predict_emb5 = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = UMSR_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###

    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb5.outputs, vgg_target_emb5.outputs, is_mean=True)
    gram_loss1 = 1e-6 * gram_scale_loss1(vgg_target_emb1.outputs,vgg_predict_emb1.outputs)
    gram_loss2 = 1e-6 * gram_scale_loss2(vgg_target_emb3.outputs,vgg_predict_emb3.outputs)
    gram_loss = gram_loss1 + gram_loss2
    #tf.summary.scalar('loss', mse_loss)
    g1_loss = mse_loss + vgg_loss + gram_loss

    g_vars = tl.layers.get_variables_with_name('UMSR_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(g1_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    #sample_imgs = train_hr_imgs[0:batch_size]
    sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_120 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    #print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_30 = tl.prepro.threading_data(sample_imgs_120, fn=downsample_fn)
    #print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_30, [ni, ni], save_dir_ginit + '/_train_sample_30.png')
    tl.vis.save_images(sample_imgs_120, [ni, ni], save_dir_ginit + '/_train_sample_120.png')

    ###========================= initialize G ====================###
    ## fixed learning rate

    #sess.run(tf.assign(lr_v, lr_init))
    for epoch in range(0, n_epoch_init + 1):
        if epoch != 0 and (epoch % decay_every_init == 0):
            new_lr_decay_init = lr_decay_init**(epoch // decay_every_init)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay_init))
            log = " ** new learning rate: %f (for Generator)" % (lr_init * new_lr_decay_init)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for Generator)" % (lr_init, decay_every_init, lr_decay_init)
            print(log)

        epoch_time = time.time()
        total_g1_loss, n_iter = 0, 0

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_120 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)  #in order to get the fix size of inputs to be suitable for the network.
            b_imgs_30 = tl.prepro.threading_data(b_imgs_120, fn=downsample_fn)
            ## update G
            errG1, _ = sess.run([g1_loss, g_optim_init], {t_image: b_imgs_30, t_target_image: b_imgs_120})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, g1: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errG1))
            total_g1_loss += errG1
            n_iter += 1
#            tf.summary.scalar('loss', mse_loss)
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g1: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_g1_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 50 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_30})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        if (epoch != 0) and (epoch % 20 == 0):
            average_lossG1 = total_g1_loss / n_iter
            f = open('testG1.text', 'a')
            f.write(str(average_lossG1) + '\n')
            f.close()


        ## save model
        if (epoch != 0) and (epoch % 500 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_%{}_init.npz'.format(tl.global_flag['mode']) % epoch, sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###

    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))


    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    imid = 0
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'Your own direction')  # if you want to test your own image

    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = UMSR_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_init.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")

    out[0]= out[0] * 255

    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

#the comparing image by Bicubic:
    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')

#session run:
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='UMSR', help='UMSR, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'UMSR':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
