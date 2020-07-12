from __future__ import division

import os
# python lib
import random
import sys

import numpy as np
# tf_render
import tensorflow as tf

#self
_curr_path = os.path.abspath(__file__) # /home/..../face
_cur_dir = os.path.dirname(_curr_path) # ./
_tf_dir = os.path.dirname(_cur_dir) # ./
_deep_learning_dir = os.path.dirname(_tf_dir) # ../
print(_deep_learning_dir)
sys.path.append(_deep_learning_dir) # /home/..../pytorch3d

from src_tfGraph.graph_train import MGC_TRAIN


flags = tf.app.flags

# mode
flags.DEFINE_boolean("debug", False, "Debug mode")
flags.DEFINE_string("debug_dump_root", '/', "")

flags.DEFINE_boolean("debug_visual", False, "")
flags.DEFINE_boolean("mode_refactor", False, "")
flags.DEFINE_string("name_refactor_file", '', "")

flags.DEFINE_boolean("train_visual", False, "")
flags.DEFINE_string("dic_train_visual", "../default_visual/", "Directory name to save the checkpoints")

flags.DEFINE_boolean("timeline", False, "Time test mode")

# data
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("dataset_loader", "", "data_loader_semi_unsupervised_skin")
flags.DEFINE_string("dataset_name_list", "train", "train train_debug")
flags.DEFINE_string("checkpoint_dir", "../default_checkpoints/", "Directory name to save the checkpoints")

flags.DEFINE_boolean("dataset_smooth_semi", False, "source images (seq_length-1)")
flags.DEFINE_integer("batch_single", 3, "source images (seq_length-1)")
flags.DEFINE_integer("batch_mul", 2, "source images (seq_length-1)")

flags.DEFINE_boolean("flag_shuffle", True, "source images (seq_length-1)")

#
flags.DEFINE_string("gpu_list", "0", "Directory name to save the checkpoints")


# continue training
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_string("init_ckpt_file", None, "Specific checkpoint file to initialize from")

flags.DEFINE_boolean("flag_data_aug", False, "The size of of a sample batch")
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 224, "Image height")
flags.DEFINE_integer("img_width", 224, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("num_source", 2, "source images (seq_length-1)")
flags.DEFINE_integer("num_scales", 1, "number of scales")

flags.DEFINE_integer("mode_intrinsic", 0, "0: 4700, 1:800")

# save
flags.DEFINE_integer("min_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("max_d", 64, "Maximum depth step when training.")
flags.DEFINE_integer("summary_freq", 1, "Logging every log_freq iterations")
flags.DEFINE_integer("save_freq", 50000, "Save the model every save_freq iterations (overwrites the previous latest model)")

# opt
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam or decay rate for RMSProp")

# loss
flags.DEFINE_float("MULTIVIEW_weight", 0.1, "Weight for smoothness")

flags.DEFINE_float("photom_weight", 0.15, "Weight for SSIM loss")
flags.DEFINE_float("ssim_weight", 0.85, "Weight for SSIM loss")
flags.DEFINE_float("depth_weight", 0.1, "Weight for depth loss")
flags.DEFINE_float("epipolar_weight", 0.0, "Weight for epipolar_weight loss")

flags.DEFINE_float("gpmm_lm_loss_weight", 0.0, "")
flags.DEFINE_float("gpmm_pixel_loss_weight", 0.0, "")
flags.DEFINE_float("gpmm_id_loss_weight", 0.0, "")
flags.DEFINE_float("gpmm_regular_shape_loss_weight", 1.0, "3DMM coeffient rank")
flags.DEFINE_float("gpmm_regular_color_loss_weight", 1.0, "3DMM coeffient rank")

# aug
flags.DEFINE_integer("aug_crop_size", 12, "")

flags.DEFINE_integer("match_num", 0, "Train with epipolar matches")
flags.DEFINE_boolean("with_pose", False, "Train with pre-computed pose")

flags.DEFINE_boolean("is_read_pose", False, "Train with pre-computed pose")
flags.DEFINE_boolean("is_read_gpmm", False, "Train with pre-computed pose")
flags.DEFINE_boolean("disable_log", False, "Disable image log in tensorboard to accelerate training")

# net
flags.DEFINE_string("net", '', "| facenet | resnet |")
flags.DEFINE_string("net_id", '', "| facenet | resnet |")

# gpmm
flags.DEFINE_string("ckpt_face_pretrain", None, "Dataset directory")
flags.DEFINE_string("ckpt_face_id_pretrain", None, "Dataset directory")
flags.DEFINE_string("ckpt_face_3dmm", None, "Dataset directory")
flags.DEFINE_string("path_gpmm", "/home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_trim_exp_uv_presplit.h5", "Dataset directory")

flags.DEFINE_integer("flag_fore", 1, "")
flags.DEFINE_integer("gpmm_rank", 80, "3DMM coeffient rank")
flags.DEFINE_integer("gpmm_exp_rank", 64, "3DMM coeffient rank")

flags.DEFINE_string("mode_id_input_clip", 'clip', "| origin | clip |")
flags.DEFINE_string("mode_depth_pixel_loss", 'origin', '| origin | relight |')

# TODO: MVS
flags.DEFINE_float("depth_min", 0.0, "Depth minimum")
flags.DEFINE_float("depth_max", 7500.0, "Depth minimum")

flags.DEFINE_float("lm_detail_weight", 1.0, "Depth minimum")

FLAGS = flags.FLAGS


"""
CUDA_VISIBLE_DEVICES=${gpu} python train_unsupervise.py --dataset_name_list train_quick --debug=False --debug_dump_root /data0 \
--dataset_loader data_loader_semi_unsupervised_skin \
--dataset_dir /data/0_eccv2020_final/54_All_MUL_MERGE \
--checkpoint_dir /home/jshang/SHANG_Exp/ECCV2020/release_2020.07.10/0_local \
--learning_rate 0.0001 --MULTIVIEW_weight 1.0 \
--photom_weight 0.15 --ssim_weight 0.0 --epipolar_weight 0.0015 --depth_weight 0.0001 \
--gpmm_lm_loss_weight 0.001 --gpmm_pixel_loss_weight 1.9 --gpmm_id_loss_weight 0.2 \
--gpmm_regular_shape_loss_weight 0.0001 --gpmm_regular_color_loss_weight 0.0003 \
--flag_fore 1 \
--batch_size 2 --img_height 224 --img_width 224 --num_scales 1 \
--min_steps 2000 --max_steps 20001 --save_freq 20000 --summary_freq 100 \
--seq_length 3 --num_source 2 --match_num 68 \
--net resnet --net_id facenet \
--ckpt_face_pretrain ./pretrain/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt \
--ckpt_face_id_pretrain ./pretrain/facenet_vgg2/model-20180402-114759.ckpt-275 \
--path_gpmm /home/jshang/SHANG_Data/ThirdLib/BFM2009/bfm09_trim_exp_uv_presplit.h5 \
--lm_detail_weight 5.0
"""

def main(_):
    # static random and shuffle
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # print and store all flags
    print('**************** Arguments ******************')
    for key in FLAGS.__flags.keys():
        print '  {}: {}'.format(key, getattr(FLAGS, key))
    print('**************** Arguments ******************')

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    path_arg_log = os.path.join(FLAGS.checkpoint_dir, "flag.txt")
    with open(path_arg_log, 'w') as f:
        for key in FLAGS.__flags.keys():
            v = '{} : {}'.format(key, getattr(FLAGS, key))
            f.write(v)
            f.write('\n')

    system = MGC_TRAIN(FLAGS)
    if 1:
        if FLAGS.train_visual:
            system.train_visual(FLAGS)
        else:
            system.train(FLAGS)
    else:
        opt = FLAGS

        test_seed0 = np.random.normal(size=[1, 10])
        test_seed1 = np.random.normal(size=[1, 10])

        pred_rank = 2 * opt.gpmm_rank + opt.gpmm_exp_rank + 6 + 27
        coeff_tar = tf.constant(np.random.normal(size=[1, pred_rank]), shape=[1, pred_rank],dtype=tf.float32)
        coeff_tar = [coeff_tar]
        coeff_src0 = tf.constant(np.random.normal(size=[1, pred_rank]), shape=[1, pred_rank],dtype=tf.float32)
        coeff_src1 = tf.constant(np.random.normal(size=[1, pred_rank]), shape=[1, pred_rank], dtype=tf.float32)
        coeff_src = [coeff_src0, coeff_src1]
        coeff = coeff_tar + coeff_src

        system.tgt_image = tf.constant(np.random.normal(size=[opt.batch_size, opt.img_height, opt.img_width, 3]),dtype=tf.float32)
        system.src_image_stack = tf.constant(np.random.normal(size=[opt.batch_size, opt.img_height, opt.img_width, 3 * opt.num_source]),dtype=tf.float32)
        system.list_tar_image = [system.tgt_image]
        system.list_src_image = [system.src_image_stack[:, :, :, i * 3:(i + 1) * 3] for i in range(opt.num_source)]
        system.list_image = system.list_tar_image + system.list_src_image
        #
        system.tgt_skin = tf.constant(np.random.normal(size=[opt.batch_size, opt.img_height, opt.img_width, 3]),dtype=tf.float32)
        system.list_tar_skin = [system.tgt_skin]

        system.src_skin = tf.constant(np.random.normal(size=[opt.batch_size, opt.img_height, opt.img_width, 3 * opt.num_source]),dtype=tf.float32)
        system.list_src_skin = [system.src_skin[:, :, :, i * 3:(i + 1) * 3] for i in range(opt.num_source)]
        system.list_skin = system.list_tar_skin + system.list_src_skin
        #
        system.flag_sgl_mul = tf.constant(np.ones(shape=[opt.batch_size]),dtype=tf.float32)

        system.matches = tf.constant(np.random.normal(size=[opt.batch_size, (opt.num_source+1), opt.match_num, 2]),dtype=tf.float32)
        system.list_lm2d_gt_tar = [system.matches[:, 0, :, :]]
        system.list_lm2d_gt_src = [system.matches[:, i, :, :] for i in range(1, system.matches.shape[1])]
        system.list_lm2d_gt = system.list_lm2d_gt_tar + system.list_lm2d_gt_src

        system.build_train_graph(coeff)

        face_variables_to_restore = tf.model_variables("InceptionResnetV1")
        print("Identity variables number: %d" % (len(face_variables_to_restore)))
        # saver = tf_render.train.Saver([var for var in test_var])
        system.face_id_restorer = tf.train.Saver(face_variables_to_restore)

        """
        run
        """
        sv = tf.train.Supervisor()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            system.face_id_restorer.restore(sess, opt.ckpt_face_id_pretrain)
            fetches={}
            fetches["ga_loss"] = system.ga_loss
            fetches["pixel_loss"] = system.pixel_loss
            fetches["ssim_loss"] = system.ssim_loss
            fetches["depth_loss"] = system.depth_loss
            fetches["epipolar_loss"] = system.epipolar_loss

            fetches["gpmm_pixel_loss"] = system.gpmm_pixel_loss
            fetches["gpmm_lm_loss"] = system.gpmm_lm_loss
            fetches["gpmm_id_loss"] = system.gpmm_id_loss
            fetches["gpmm_reg_shape_loss"] = system.gpmm_regular_shape_loss
            fetches["gpmm_reg_color_loss"] = system.gpmm_regular_color_loss
            """
            *********************************************   Start Trainning   *********************************************
            """
            results = sess.run(fetches)

            print("ga/pixel/ssim/depth/epipolar loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                results["ga_loss"], results["pixel_loss"], results["ssim_loss"], results["depth_loss"],
                results["epipolar_loss"]))

            print("(weight)ga/pixel/ssim/depth/epipolar loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                results["ga_loss"] * opt.MULTIVIEW_weight,
                results["pixel_loss"] * (1 - opt.ssim_weight),
                results["ssim_loss"] * opt.ssim_weight,
                results["depth_loss"] * opt.depth_weight,
                results["epipolar_loss"] * opt.epipolar_weight)
                  )

            # 3dmm loss
            print("mm_pixel/mm_lm/mm_id/mm_reg_s/mm_reg_c loss: [%.4f/%.4f/%.4f/%.4f/%.4f]" % (
                results["gpmm_pixel_loss"], results["gpmm_lm_loss"], results["gpmm_id_loss"],
                results["gpmm_reg_shape_loss"], results["gpmm_reg_color_loss"]))

            print("(weight)mm_pixel/mm_lm/mm_id/mm_reg_s/mm_reg_c loss: [%.4f/%.4f/%.4f/%.4f/%.4f]\n" % (
                results["gpmm_pixel_loss"] * opt.gpmm_pixel_loss_weight,
                results["gpmm_lm_loss"] * opt.gpmm_lm_loss_weight,
                results["gpmm_id_loss"] * opt.gpmm_id_loss_weight,
                results["gpmm_reg_shape_loss"] * opt.gpmm_regular_shape_loss_weight,
                results["gpmm_reg_color_loss"] * opt.gpmm_regular_color_loss_weight))



if __name__ == '__main__':
    tf.app.run()
