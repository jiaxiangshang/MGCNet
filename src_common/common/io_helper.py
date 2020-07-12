#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jiaxiang Shang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: jiaxiang.shang@gmail.com
@time: 3/25/20 12:47 PM
@desc:
'''
import tensorflow as tf

def unpack_image_sequence(image_seq, img_height, img_width, num_source):
    if len(image_seq.shape) == 2:
        image_seq = tf.expand_dims(image_seq, -1)
    channel = image_seq.shape[2]

    # Assuming the center image is the target frame
    tgt_start_idx = int(img_width * (num_source // 2))
    tgt_image = tf.slice(image_seq,
                         [0, tgt_start_idx, 0],
                         [-1, img_width, -1])
    # Source frames before the target frame
    src_image_1 = tf.slice(image_seq,
                           [0, 0, 0],
                           [-1, int(img_width * (num_source // 2)), -1])
    # Source frames after the target frame
    src_image_2 = tf.slice(image_seq,
                           [0, int(tgt_start_idx + img_width), 0],
                           [-1, int(img_width * (num_source // 2)), -1])
    src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
    # Stack source frames along the color channels (i.e. [H, W, N*3])
    src_image_stack = tf.concat([tf.slice(src_image_seq,
                                          [0, i * img_width, 0],
                                          [-1, img_width, -1])
                                 for i in range(num_source)], axis=2)
    src_image_stack.set_shape([img_height, img_width, num_source * channel])
    tgt_image.set_shape([img_height, img_width, channel])
    return tgt_image, src_image_stack

def unpack_image_batch_list(image_seq, img_height, img_width, num_source):
    tar_list = []
    src_list = []
    for i in range(image_seq.shape[0]):
        tgt_image, src_image_stack = unpack_image_sequence(image_seq[i], img_height, img_width, num_source)
        tar_list.append(tgt_image)
        src_list.append(src_image_stack)
    tgt_image_b = tf.stack(tar_list)
    src_image_stack_b = tf.stack(src_list)

    list_tar_image = [tgt_image_b]
    list_src_image = [src_image_stack_b[:, :, :, i * 3:(i + 1) * 3] for i in range(num_source)]
    list_image = list_tar_image + list_src_image

    return list_image

# np
def unpack_image_np(image_seq, img_height, img_width, num_source):

    tgt_start_idx = int(img_width * (num_source // 2))

    tgt_image = image_seq[:, tgt_start_idx:tgt_start_idx+img_width, :]
    src_image_1 = image_seq[:, 0:int(img_width * (num_source // 2)), :]
    src_image_2 = image_seq[:, tgt_start_idx+img_width:tgt_start_idx+img_width+int(img_width * (num_source // 2)), :]

    return src_image_1, tgt_image, src_image_2, [tgt_image, src_image_1, src_image_2]