import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector


def save_projector_features(x_processed, model, feature_save_path, feature_layer_index=-1):
    """
    save the features for the projector.
    :param x_processed: The pro-processed images
    :param model: The model to extract the features
    :param feature_save_path: The path to save the features
    :param feature_layer_index: The index of the layer with the features. Default is -1
    :return: None
    """
    if ".pkl" not in feature_save_path:
        feature_save_path += ".pkl"
    base = os.path.dirname(feature_save_path)
    if not os.path.exists(base):
        os.makedirs(base)

    # create the embedding model
    embeddings = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[feature_layer_index - 1].output)
    print(embeddings.summary())
    features = []
    for image in tqdm(x_processed):
        features.append(embeddings.predict(tf.expand_dims(image, axis=0)))
    image_features_arr = np.asarray(features)
    del features  # del to get free space
    image_features_arr = np.rollaxis(image_features_arr, 1, 0)
    image_features_arr = image_features_arr[0, :, :]
    pickle.dump(image_features_arr, open(feature_save_path, 'wb'))

def visualise_features(x, y, log_dir, feature_file_path, class_names):

    embedding_dir = os.path.join(log_dir, "embedding_logs")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)

    # create the tsv file
    with open(os.path.join(embedding_dir, "metadata.tsv"), 'w') as meta_file:
        meta_file.write('Class\tName\n')
        for label in y:
            meta_file.write('{}\t{}\n'.format(class_names.index(label), label))

    # prepare sprite images
    # unprocessed images
    img_data = []
    for img in tqdm(x):
        input_img_resize = tf.image.resize(img, (32, 32))
        img_data.append(input_img_resize)
    img_data = np.array(img_data)

    # Taken from: https://github.com/tensorflow/tensorflow/issues/6322
    def images_to_sprite(data):
        """Creates the sprite image along with any necessary padding
        Args:
          data: NxHxW[x3] tensor containing the images.
        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
        # Inverting the colors seems to look better for MNIST
        # data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

    # %%
    sprite = images_to_sprite(img_data)
    cv2.imwrite(os.path.join(embedding_dir, 'sprite_classes.png'), cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR))

    # load features
    with open(feature_file_path, 'rb') as f:
        feature_vectors = pickle.load(f)
    # feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
    print("feature_vectors_shape:", feature_vectors.shape)
    print("num of images:", feature_vectors.shape[0])
    print("size of individual feature vector:", feature_vectors.shape[1])

    features = tf.Variable(feature_vectors, name='features')
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=features)
    checkpoint.save(os.path.join(embedding_dir, "embedding.ckpt"))
    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'
    # Comment out if you don't want sprites
    embedding.sprite.image_path = 'sprite_classes.png'
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(embedding_dir, config)

# def create_projection(datagen, save_folder, class_names, features_name=FEATURE_NAME):
#     unprocessed_images = []
#     all_labels = []
#     steps = datagen.n // datagen.batch_size + 1
#     for step, batch in enumerate(tqdm(datagen)):
#         if step >= steps:
#             break
#         unprocessed_images.extend(batch[0])
#         all_labels.extend([np.argmax(item) for item in batch[1]])
#
#      # create the tsv file
#     with open(os.path.join(save_folder, "metadata.tsv"), 'w') as meta_file:
#         meta_file.write('Class\tName\n')
#         for label in all_labels:
#             meta_file.write('{}\t{}\n'.format(label, class_names[label]))
#
#     unprocessed_images = np.array(unprocessed_images, dtype=np.int32)
#
#     sprite = images_to_sprite(unprocessed_images)
#     cv2.imwrite(os.path.join(save_folder, 'sprite_classes.png'), cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR))
#     del sprite
#
#     # load features
#     with open(os.path.join(save_folder, features_name), 'rb') as f:
#         feature_vectors = pickle.load(f)
#     print("feature_vectors_shape:", feature_vectors.shape)
#     print("num of images:", feature_vectors.shape[0])
#     print("size of individual feature vector:", feature_vectors.shape[1])
#
#     features = tf.Variable(feature_vectors, name='features')
#     # Create a checkpoint from embedding, the filename and key are
#     # name of the tensor.
#     checkpoint = tf.train.Checkpoint(embedding=features)
#     checkpoint.save(os.path.join(save_folder, "embedding.ckpt"))
#     # Set up config
#     config = projector.ProjectorConfig()
#     embedding = config.embeddings.add()
#     # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
#     embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
#     # Link this tensor to its metadata file (e.g. labels).
#     embedding.metadata_path = 'metadata.tsv'
#     # Comment out if you don't want sprites
#     embedding.sprite.image_path = 'sprite_classes.png'
#     embedding.sprite.single_image_dim.extend([unprocessed_images.shape[1], unprocessed_images.shape[1]])
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(save_folder, config)
#     del feature_vectors
#     del all_labels

if __name__ == "__main__":
    pass
    # get the dataset
    # processed_gen, unprocessed_gen = obtain_datagens(DATASET_ROOT, CSV_NAME, IS_TRAIN, TRANSITION_FOLDER, BATCH_SIZE,
    #                                                  IMAGE_SIZE, AUGMENTATION_PATH, AUGMENTATION_PERCENTAGE)
    # image_features_array, y_values = extract_features(embedding_model, processed_gen, SAVE_FOLDER)
    # create_projection(unprocessed_gen, SAVE_FOLDER, CLASSES)