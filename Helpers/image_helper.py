import math
import re
import uuid
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# fix display issues
# plt.style.use('dark_background')


plt.style.use('default')

def obtain_file_paths(root_path, search_term, file_name_only=False):
    file_paths = []
    r1 = re.compile(search_term)
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            if r1.search(file):
                if file_name_only:
                    file_paths.append(file)
                else:
                    file_paths.append(os.path.join(subdir, file))
    return file_paths


def initialise_tf():
    import tensorflow as tf
    try:
        # fix memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


def load_image_from_file(file_path, label=None, desired_shape=(256, 256)):
    import tensorflow as tf
    initialise_tf()
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(contents=image, channels=3)
    if desired_shape is not None:
        image = tf.image.resize(image, size=desired_shape)
    if label is None:
        return image
    else:
        return image, label


def create_image_grid(images, num_cols=None, must_undo_preprocess=False, undo_process_formula=None, scale=3,
                      file_name=None, enable_unique=False, must_show=True, title=None, class_names=None, dpi=None, fontsize=13):
    """
    display an image grid
    :param images: the images to display
    :param num_cols: The number of columns in the grid
    :param must_undo_preprocess: True to undo preprocessing to the image
    :param undo_process_formula: Provide a formula to be used to undo the preprocessing.
    If None, ((images + 1.0) * 127.5).astype(np.uint8) is used
    :param scale: scale up the image size
    :param file_name: provide a filename to save the image grid
    :param enable_unique: True to generate a unique file name
    :param must_show: True to show the image grid
    :param title: The title of the image grid
    :param class_names: Titles for each image in the grid
    :param dpi: The DPI to save the image grid
    :return: The file path of the saved image or None if no file path is provided
    """

    # calculate the number of rows based of the number of columns and images
    if num_cols is None:
        num_cols = math.ceil(len(images) ** (1 / 2))

    # use interger division to drop any remainder
    num_rows = (len(images) - 1) // num_cols + 1

    images = np.array(images, dtype=np.int32)

    # test if the image has already been converted to [-1,1]
    if must_undo_preprocess:
        if undo_process_formula is None:
            # convert the images back
            images = ((images + 1.0) * 127.5).astype(np.uint8)
        else:
            images = undo_process_formula(images)

    #  test if the images is binary
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
        cmap = 'binary'
    else:
        cmap = None
    if not must_show:
        with plt.ioff():
            # set the plot dimensions
            fig = plt.figure(figsize=(num_cols * scale, num_rows * scale), dpi=dpi)
            if title is not None:
                fig.suptitle(title)

            for index, image in enumerate(images):
                plt.subplot(int(num_rows), int(num_cols), int(index + 1))
                plt.imshow(image, cmap=cmap)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                if class_names is None:
                    plt.xlabel(str(index), fontsize=fontsize)
                else:
                    plt.xlabel(class_names[int(index)], fontsize=fontsize)

    else:
        with plt.ion():
            # set the plot dimensions
            fig = plt.figure(figsize=(num_cols * scale, num_rows * scale), dpi=dpi)
            if title is not None:
                fig.suptitle(title)

            for index, image in enumerate(images):
                plt.subplot(int(num_rows), int(num_cols), int(index + 1))
                plt.imshow(image, cmap=cmap)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                if class_names is None:
                    plt.xlabel(str(index), fontsize=13)
                else:
                    plt.xlabel(class_names[int(index)], fontsize=13)

    file_path = None
    if file_name is not None:
        if enable_unique:
            name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            split = os.path.splitext(file_path)
            file_path = os.path.join(split[0], name, split[1])
        else:
            file_path = file_name
        if os.path.isfile(file_path):
            name = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")
            split = os.path.splitext(file_path)

            file_path = os.path.join(split[0] + name + split[1])
        file_root = os.path.dirname(file_path)
        if len(file_root)>0 and not os.path.exists(file_root):
            os.makedirs(file_root)
        plt.savefig(file_path, bbox_inches='tight')
    if must_show == False:
        plt.ioff()
        plt.close(fig)
    else:
        plt.ion()
        plt.show()
    return file_path


def create_image_grid_from_paths(image_paths, num_cols=None, must_undo_preprocess=False, undo_process_formula=None,
                                 scale=3, file_name=None, enable_unique=False, must_show=True, title=None,
                                 class_names=None, dpi=None, desired_image_shape=256, fontsize=13):
    """
        display an image grid
        :param image_paths: the filenames of the images
        :param num_cols: The number of columns in the grid
        :param must_undo_preprocess: True to undo preprocessing to the image
        :param undo_process_formula: Provide a formula to be used to undo the preprocessing.
        If None, ((images + 1.0) * 127.5).astype(np.uint8) is used
        :param scale: scale up the image size
        :param file_name: provide a filename to save the image grid
        :param enable_unique: True to generate a unique file name
        :param must_show: True to show the image grid
        :param title: The title of the image grid
        :param class_names: Titles for each image in the grid
        :param dpi: The DPI to save the image grid
        :param desired_image_shape: The dimension to resize the image to.
        :return: The file path of the saved image or None if no file path is provided
        """
    images = [load_image_from_file(file_path, desired_shape=(desired_image_shape, desired_image_shape)) for file_path in image_paths]
    images = np.array(images, dtype=np.int32)
    return create_image_grid(images, num_cols, must_undo_preprocess, undo_process_formula, scale, file_name,
                             enable_unique, must_show, title, class_names, dpi, fontsize=fontsize)


def save_image(image, file_name):
    """
    Write an image to storage
    :param image: The image to save
    :param file_name: the name of the image file including the extension. If a directory path is provided with the image
    name, the directory structure is created if it does not exist.
    :return: None
    """
    folder_path = os.path.dirname(file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
