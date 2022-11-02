import os
import pickle
import shutil
from datetime import datetime
from random import sample

import cv2
import numpy as np
import pandas as pd
from tensorboard.plugins import projector
from tqdm import tqdm
from vit_keras import vit
import tensorflow as tf

from helpers.dataset_preparer import move_to_folder
from helpers.projector_helper import save_projector_features, visualise_features
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

FEATURE_NAME="feature_samples.pkl"
SPRITE_SIZE = 30
def create_model():
    vit_model = vit.vit_b32(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=2
    )

    vit_model.trainable = False
    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Dense(len(CLASSES), 'softmax')
    ],
        name='vision_transformer')

    model.summary()
    return model

def copy_augmentation(augmentation_paths, train_augmentation_root, AUGMENTATION_PERCENTAGE):

    if isinstance(augmentation_paths, list):
        # calculate number of images to move
        # num_images = round(
        #     (image_count * percentage_augmentation / (1 - percentage_augmentation)) / len(augmentation_paths))
        # get the file paths
        file_path_dic = {}
        for aug_path in augmentation_paths:
            image_file_paths = []
            for subdir, dirs, files in os.walk(aug_path):
                for file in files:
                    # print(os.path.join(subdir, file))
                    if "seed" in file:
                        image_file_paths.append(os.path.join(subdir, file))
            file_path_dic[aug_path] = image_file_paths
        all_image_paths = []
        for image_file_paths in file_path_dic.values():
            image_paths = sample(image_file_paths, int(len(image_file_paths) * AUGMENTATION_PERCENTAGE))
            all_image_paths.extend(image_paths)
    else:
        # calculate number of images to move
        # num_images = round((image_count * percentage_augmentation) / (1 - percentage_augmentation))
        # get the file paths
        image_file_paths = []
        for subdir, dirs, files in os.walk(augmentation_paths):
            for file in files:
                # print(os.path.join(subdir, file))
                if "seed" in file:
                    image_file_paths.append(os.path.join(subdir, file))

        all_image_paths = sample(image_file_paths, int(len(image_file_paths) * AUGMENTATION_PERCENTAGE))

    destination_real = os.path.join(train_augmentation_root,  'real')
    destination_spoof = os.path.join(train_augmentation_root,  'spoof')
    if not os.path.exists(destination_real):
        os.makedirs(destination_real)
    if not os.path.exists(destination_spoof):
        os.makedirs(destination_spoof)

    for path in all_image_paths:
        file_name = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"
        if "N1" in path or "N2" in path:
            shutil.copy(path, os.path.join(destination_real, f"train_real_{file_name}"))
        else:
            shutil.copy(path, os.path.join(destination_spoof, f"train_spoof_{file_name}"))

def obtain_datagens(dataset_root,csv_name, is_train, transition_folder, batch_size, image_size, AUGMENTATION_PATH=None, AUGMENTATION_PERCENTAGE=0.1):
    if AUGMENTATION_PATH is None:
        DATASET_CSV = os.path.join(dataset_root, csv_name)

        # Dataframe Management
        DF_ALL = pd.read_csv(DATASET_CSV)
        if is_train:
            DF = DF_ALL.loc[(DF_ALL["usage_type"] == "train")]
            transition_flow_folder = os.path.join(transition_folder, 'train')
        else:
            DF = DF_ALL.loc[(DF_ALL["usage_type"] == "test")]
            transition_flow_folder = os.path.join(transition_folder, 'test')


        move_to_folder(DF, transition_folder, dataset_root)

    else:
        transition_flow_folder = transition_folder
        copy_augmentation(AUGMENTATION_PATH, transition_folder, AUGMENTATION_PERCENTAGE)


    processed_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                              rescale=1. / 255,
                                                              samplewise_center=True,
                                                              samplewise_std_normalization=True,


                                                              )
    processed_gen = processed_data_gen.flow_from_directory(
        directory=transition_flow_folder,
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(image_size, image_size),

    )

    unprocessed_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    unprocessed_gen = unprocessed_data_gen.flow_from_directory(
        directory=transition_flow_folder,
        batch_size=batch_size,
        seed=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical',
        target_size=(SPRITE_SIZE, SPRITE_SIZE))
    return processed_gen, unprocessed_gen

def load_best_model(model_path):
    # create the model
    print("restoring top model")
    # load best model
    latest = tf.train.latest_checkpoint(model_path)
    print(latest)
    model = create_model()
    model.load_weights(latest)
    return model

# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
  """Creates the sprite image along with any necessary padding
  Args:
    data: NxHxW[x3] tensor containing the images.
  Returns:
    data: Properly shaped HxWx3 image with any necessary padding.
  """
  if len(data.shape) == 3:
      data = np.tile(data[...,np.newaxis], (1,1,1,3))
  data = data.astype(np.float32)
  min = np.min(data.reshape((data.shape[0], -1)), axis=1)
  data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
  max = np.max(data.reshape((data.shape[0], -1)), axis=1)
  data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
  # Inverting the colors seems to look better for MNIST
  #data = 1 - data

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

def extract_features(model, datagen, save_path, features_name=FEATURE_NAME):
    if ".pkl" not in features_name:
        features_name += ".pkl"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    features = model.predict(datagen, steps=datagen.n // datagen.batch_size + 1)

    image_features_arr = np.asarray(features)
    del features  # del to get free space
    # image_features_arr = np.rollaxis(image_features_arr, 1, 0)
    # image_features_arr = image_features_arr[0, :, :]
    with open(os.path.join(save_path, features_name), 'wb') as file:
        pickle.dump(image_features_arr, file)

    del image_features_arr



def create_projection(datagen, save_folder, class_names, features_name=FEATURE_NAME):
    unprocessed_images = []
    all_labels = []
    steps = datagen.n // datagen.batch_size + 1
    for step, batch in enumerate(tqdm(datagen)):
        if step >= steps:
            break
        unprocessed_images.extend(batch[0])
        all_labels.extend([np.argmax(item) for item in batch[1]])

     # create the tsv file
    with open(os.path.join(save_folder, "metadata.tsv"), 'w') as meta_file:
        meta_file.write('Class\tName\n')
        for label in all_labels:
            meta_file.write('{}\t{}\n'.format(label, class_names[label]))

    unprocessed_images = np.array(unprocessed_images, dtype=np.int32)

    sprite = images_to_sprite(unprocessed_images)
    cv2.imwrite(os.path.join(save_folder, 'sprite_classes.png'), cv2.cvtColor(sprite, cv2.COLOR_RGB2BGR))
    del sprite

    # load features
    with open(os.path.join(save_folder, features_name), 'rb') as f:
        feature_vectors = pickle.load(f)
    print("feature_vectors_shape:", feature_vectors.shape)
    print("num of images:", feature_vectors.shape[0])
    print("size of individual feature vector:", feature_vectors.shape[1])

    features = tf.Variable(feature_vectors, name='features')
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=features)
    checkpoint.save(os.path.join(save_folder, "embedding.ckpt"))
    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'
    # Comment out if you don't want sprites
    embedding.sprite.image_path = 'sprite_classes.png'
    embedding.sprite.single_image_dim.extend([unprocessed_images.shape[1], unprocessed_images.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(save_folder, config)
    del feature_vectors
    del all_labels



if __name__ == "__main__":
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    IS_TRAIN = False
    IS_KF = False
    if IS_TRAIN:
        str_train = "Train_"
    else:
        str_train = "Test_"

    CLASSES = ['real', 'spoof']  # TP: it is an attack, TN: It is a real person
    # MODEL_PATH_Root = "/home/jarred/Documents/Code/PycharmProjects/Masters2022/Tune_Individual/AugmentedModelCheckpoints"
    # MODEL_Name = "TUNE_02_08_2022_12_12_18/KF/20/W1/20_W1"
    MODEL_PATH_Root = ""
    AUGMENTATION_PERCENTAGE = 0.20
    if IS_KF:
        # TOP KF Trained
        MODEL_Name  = "/home/jarred/Documents/Code/PycharmProjects/Masters2022/Tune_RUNS/KF_RUN_3/TUNE_Combined_03_01_2022_15_46_54/AugmentedModelCheckpoints/KF|40_C2,KF|40_N2,KF|40_R2,KF|40_W2/15/03_02_2022_10_12_09.518789/15_03_02_2022_10_12_09.518789"
        SAVE_FOLDER = "./TSNE/KF_WITH_GEN_" + str(AUGMENTATION_PERCENTAGE)
    else:
        # TOP AF Trained
        MODEL_Name  = "/home/jarred/Documents/Code/PycharmProjects/Masters2022/Tune_RUNS/RUN_3/TUNE_Combined_02_20_2022_10_50_33/AugmentedModelCheckpoints/KF|40_C1,KF|40_N1,KF|40_R1,KF|40_W1,KF|40_C2,KF|40_N2,KF|40_R2,KF|40_W2/15/02_20_2022_20_02_16.225713/15_02_20_2022_20_02_16.225713"
        SAVE_FOLDER = "./TSNE/AF_WITH_GEN_" + str(AUGMENTATION_PERCENTAGE)

    # SAVE_FOLDER = "./TSNE/FEATURES_"+str_train+MODEL_Name.replace("/", "_")
    # DATASET_ROOT = "/home/jarred/Documents/Datasets/CASIA_KF"
    DATASET_ROOT = "/home/jarred/Documents/Datasets/CASIA_REFACTORED"
    TRANSITION_FOLDER = os.path.join(SAVE_FOLDER,"TRANSITION_"+datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    if not os.path.exists(TRANSITION_FOLDER):
        os.makedirs(TRANSITION_FOLDER)
    CSV_NAME = "casia.csv"
    MODEL_PATH = os.path.join(MODEL_PATH_Root, MODEL_Name)

    augmentation_items = [
        "Full|40_C1,Full|40_N1,Full|40_R1,Full|40_W1",
        "Full|40_C2,Full|40_N2,Full|40_R2,Full|40_W2",
        "Full|40_C1,Full|40_N1,Full|40_R1,Full|40_W1,Full|40_C2,Full|40_N2,Full|40_R2,Full|40_W2",
        "KF|40_C1,KF|40_N1,KF|40_R1,KF|40_W1",
        "KF|40_C2,KF|40_N2,KF|40_R2,KF|40_W2",
        "KF|40_HR_N,KF|40_HR_W,KF|40_HR_C,KF|40_HR_R",
        "KF|40_C1,KF|40_N1,KF|40_R1,KF|40_W1,KF|40_C2,KF|40_N2,KF|40_R2,KF|40_W2",
        "KF|40_C1,KF|40_N1,KF|40_R1,KF|40_W1,KF|40_C2,KF|40_N2,KF|40_R2,KF|40_W2,KF|40_HR_N,KF|40_HR_W,KF|40_HR_C,KF|40_HR_R",
    ]

    AUGMENTATION_PATH = augmentation_items[-1]
    AUGMENTATION_PATH = AUGMENTATION_PATH.replace("|", "/")
    AUGMENTATION_PATH = AUGMENTATION_PATH.split(",")
    if isinstance(AUGMENTATION_PATH, list):

        for i in range(len(AUGMENTATION_PATH)):
            AUGMENTATION_PATH[i] = os.path.join("/home/jarred/Documents/Datasets/ordered_seeds",AUGMENTATION_PATH[i])

    # get the model
    model = load_best_model(MODEL_PATH)

    # create the emedding model
    embedding_model = tf.keras.models.Model(inputs=model.layers[0].inputs, outputs=model.layers[0].output)
    embedding_model.trainable = False

    # get the dataset
    processed_gen, unprocessed_gen = obtain_datagens(DATASET_ROOT, CSV_NAME,IS_TRAIN,TRANSITION_FOLDER,BATCH_SIZE,IMAGE_SIZE, AUGMENTATION_PATH, AUGMENTATION_PERCENTAGE)
    extract_features(embedding_model,processed_gen,SAVE_FOLDER)
    create_projection(unprocessed_gen, SAVE_FOLDER, CLASSES)

    # clean up transition folder
    if os.path.exists(TRANSITION_FOLDER):
        shutil.rmtree(TRANSITION_FOLDER)