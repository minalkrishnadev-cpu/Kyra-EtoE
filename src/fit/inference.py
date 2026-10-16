import os
import sys
import tarfile
import argparse

import numpy as np
import cv2
from PIL import Image
from six.moves import urllib
import tensorflow as tf

import src.config
from demo import main

# TF1 API (HMR is TF1 code). Fall back to tf.compat.v1 if a TF2 install is used.
tf1 = tf.compat.v1 if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1') else tf

# Pillow >= 10 removed Image.ANTIALIAS
RESAMPLE = getattr(Image, 'LANCZOS', None) or getattr(Image, 'ANTIALIAS')

HERE = os.path.dirname(os.path.abspath(__file__))


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf1.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf1.import_graph_def(graph_def, name='')

        self.sess = tf1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single PIL image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, RESAMPLE)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def get_args():
    parser = argparse.ArgumentParser(description='Deeplab Segmentation + HMR height fit')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to the input image. (required)')
    parser.add_argument('-ht', '--height', type=int, required=True,
                        help='Known height of the person in cm. (required)')
    return parser.parse_args()


def load_deeplab(model_name='xception_coco_voctrainval'):
    url_prefix = 'http://download.tensorflow.org/models/'
    model_urls = {
        'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug': 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    tarball_name = model_urls[model_name]

    deeplab_dir = os.path.join(HERE, 'deeplab_model')
    if not os.path.exists(deeplab_dir):
        os.makedirs(deeplab_dir)

    download_path = os.path.join(deeplab_dir, tarball_name)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(url_prefix + tarball_name, download_path)
        print('download completed! loading DeepLab model...')

    model = DeepLabModel(download_path)
    print('model loaded successfully!')
    return model


if __name__ == '__main__':
    args = get_args()

    if not os.path.isfile(args.input_dir):
        sys.exit('Input image not found: %s' % args.input_dir)

    model = load_deeplab()

    image = Image.open(args.input_dir)

    res_im, seg = model.run(image)

    seg = cv2.resize(seg.astype(np.uint8), image.size,
                     interpolation=cv2.INTER_NEAREST)
    mask_sel = (seg == 15).astype(np.float32)  # 15 = 'person'
    mask = 255 * mask_sel.astype(np.uint8)

    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    res = cv2.bitwise_and(img, img, mask=mask)
    bg_removed = res + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    main(bg_removed, args.height, None)