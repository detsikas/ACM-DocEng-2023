import tensorflow as tf
import numpy as np
from model import dilated_multires_visual_attention
from utils import extract_inference_patches
import argparse
import cv2
import os
import sys

parser = argparse.ArgumentParser(
    description='Predict image')
parser.add_argument('input_image', help='Path input image')
parser.add_argument('output_image', help='Output image')
args = parser.parse_args()

input_image = args.input_image
output_image = args.output_image
stride = 200

if not os.path.exists(input_image) or not os.path.isfile(input_image):
    print('Input image does not exist')
    sys.exit(0)

target_image_size = 256
input_shape = [target_image_size, target_image_size, 3]

# Load the model and its weights
model = dilated_multires_visual_attention(
    input_shape=input_shape, starting_filters=16, with_dropout=True)
model.load_weights(os.path.join('model', 'weights')).expect_partial()


def reconstruct_image(predictions_, h_anchors, w_anchors, whole_image_shape):
    target_image_size_ = predictions_.shape[1]
    number_of_classes_ = predictions_.shape[-1]
    predictions_ = tf.reshape(predictions_, [h_anchors.shape[0],
                                             w_anchors.shape[0], target_image_size_, target_image_size_, number_of_classes_])
    merged = np.zeros(
        [whole_image_shape[0], whole_image_shape[1], predictions_.shape[-1]])
    count = np.zeros_like(merged)

    for i, h in enumerate(h_anchors):
        for j, w in enumerate(w_anchors):
            merged[h:h+target_image_size_, w:w +
                   target_image_size_] += predictions_[i, j]
            count[h:h+target_image_size_, w:w+target_image_size_] += 1

    merged = merged/count

    return merged


def threshold_image(img):
    local_image = np.copy(img)
    local_image[local_image <= 0.5] = 0
    local_image[local_image > 0.5] = 1
    return local_image


x = cv2.imread(input_image).astype(np.float32)/255.0
image_size = tf.shape(x)
patches, h_anchors, w_anchors = extract_inference_patches(
    x, target_image_size, stride)
predictions = model.predict(patches, verbose=0)
reconstructed_image = reconstruct_image(
    predictions, h_anchors, w_anchors, [image_size[0], image_size[1], x.shape[-1]])

y = threshold_image(reconstructed_image)
y = np.clip(y*255, 0, 255).astype('uint8')
cv2.imwrite(output_image, y)
