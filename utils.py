import numpy as np
import tensorflow as tf


def remove_out_of_bounds_anchors(anchors, dimension, target_size):
    offset = 0
    for h in anchors:
        if h <= (dimension-target_size):
            offset += 1
        else:
            break

    return anchors[:offset]


def extract_inference_patches(image, target_image_size, stride):
    H = image.shape[0]
    W = image.shape[1]

    h_anchors = np.arange(0, H, stride)
    w_anchors = np.arange(0, W, stride)

    h_anchors = remove_out_of_bounds_anchors(h_anchors, H, target_image_size)
    w_anchors = remove_out_of_bounds_anchors(w_anchors, W, target_image_size)

    if image.shape[0]-target_image_size not in h_anchors:
        h_anchors = np.concatenate([h_anchors, [H-target_image_size]], axis=0)

    if image.shape[1]-target_image_size not in w_anchors:
        w_anchors = np.concatenate([w_anchors, [W-target_image_size]], axis=0)

    patches = []
    for h in h_anchors:
        for w in w_anchors:
            patch = image[h:h+target_image_size, w:w+target_image_size, :]
            patches.append(patch)

    x = tf.stack(patches)

    return x, h_anchors, w_anchors
