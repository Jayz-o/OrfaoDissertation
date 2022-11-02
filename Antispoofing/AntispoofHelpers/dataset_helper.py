import os.path

import numpy as np
import pandas as pd
import tensorflow
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from Antispoofing.AntispoofHelpers.spoof_metric import determine_spoof_metrics
from Helpers.image_helper import obtain_file_paths, create_image_grid

Y_COL = 'ground_truth'

X_COL = 'file_path'
Z_COL = "class"

TRAIN_USAGE_TYPE = 'train'
TEST_USAGE_TYPE = 'test'

USAGE_TYPE_COL = 'usage_type'

DIRECTORY_PATH_COL = 'directory_path'

GROUND_TRUTH_COL = 'ground_truth'


def initialise_tf():
    import tensorflow as tf
    try:
        # fix memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


def pre_process(x, y):
    from vit_keras import vit
    return vit.preprocess_inputs(x), y


def get_file_path_ground_truth_frame(dataset_root, directory_path, ground_truth, image_file_name=r"^frame",
                                     stratified_name=None):
    image_files = obtain_file_paths(os.path.join(dataset_root, directory_path), image_file_name)
    ground_truths = [ground_truth] * len(image_files)
    if stratified_name is not None:
        stratified_names = [stratified_name] * len(image_files)
        df = pd.DataFrame.from_dict({X_COL: image_files, Y_COL: ground_truths, Z_COL: stratified_names})
    else:
        df = pd.DataFrame.from_dict({X_COL: image_files, Y_COL: ground_truths})
    return df

def get_aug_file_path_ground_truth_frame(dataset_root, directory_path, ground_truth, image_file_name=r"^frame",
                                     stratified_name_list=None):
    image_files = obtain_file_paths(os.path.join(dataset_root, directory_path), image_file_name)
    ground_truths = [ground_truth] * len(image_files)
    if stratified_name_list is not None:
        stratified_names = [stratified_name_list[i % len(stratified_name_list)] for i in range(len(image_files))]
        df = pd.DataFrame.from_dict({X_COL: image_files, Y_COL: ground_truths, Z_COL: stratified_names})
    else:
        df = pd.DataFrame.from_dict({X_COL: image_files, Y_COL: ground_truths})
    return df


def get_dataframe_by_query(csv_frame, query):
    if type(csv_frame) is str:
        frame = pd.read_csv(csv_frame)
    else:
        frame = csv_frame
    return frame.query(query)


def get_dataframe_by_attack_category(csv_frame, attack_category):
    return get_dataframe_by_query(csv_frame, f'attack_category == "{attack_category}"')


def get_dataframe_by_medium_name(csv_frame, medium_name):
    return get_dataframe_by_query(csv_frame, f'medium_name == "{medium_name}"')


def get_dataframe_by_usage_type(csv_frame, usage_type):
    if type(csv_frame) is str:
        frame = pd.read_csv(csv_frame)
    else:
        frame = csv_frame
    return frame.query(f'usage_type == "{usage_type}"')


def obtain_datagen_frame(dataset_root, csv_name, usage_type):
    csv_path = os.path.join(dataset_root, csv_name)
    if not os.path.exists(csv_path):
        raise TypeError(f"Could not locate data csv: {csv_path}")

    # load the csv
    data_frames = []
    df = pd.read_csv(csv_path)
    df = df[[DIRECTORY_PATH_COL, GROUND_TRUTH_COL]][df[USAGE_TYPE_COL] == usage_type]
    for index, row in df.iterrows():
        data_frames.append((dataset_root, row[DIRECTORY_PATH_COL], row[GROUND_TRUTH_COL]))

    df = pd.concat(data_frames, axis=0, ignore_index=True)
    return df


def split_frames(df, validation_split):
    train, validate = train_test_split(df, test_size=validation_split)
    return train, validate


def test_split(train, val, col_key):
    files_detected = train[col_key].isin((val[col_key])).unique().tolist()

    if len(files_detected) != 1:
        # inner merge
        df_merge = pd.merge(train, val, how='inner', left_on=['file_path'], right_on=['file_path'])
        raise TypeError("Found file in the training set that are present in the validation set")

    if files_detected[0]:
        raise TypeError("Found file in the training set that are present in the validation set")


def get_train_validation_generator(train_frame, val_frame, batch_size=32, target_size=224, dataset_root=None, shuffle_seed=None, use_hsv=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    initialise_tf()
    def to_hsv(x):
        return tensorflow.image.rgb_to_hsv(x)


    if use_hsv:
        train_gen = ImageDataGenerator(preprocessing_function=to_hsv)
        val_gen = ImageDataGenerator(preprocessing_function=to_hsv)

    else:
        train_gen = ImageDataGenerator()
        val_gen = ImageDataGenerator()



    train_generator = train_gen.flow_from_dataframe(
        dataframe=train_frame,
        directory=dataset_root,
        x_col=X_COL,
        y_col=Y_COL,
        batch_size=batch_size,
        seed=shuffle_seed,
        shuffle=True,
        class_mode="categorical",
        target_size=(target_size, target_size))

    valid_generator = val_gen.flow_from_dataframe(
        dataframe=val_frame,
        directory=dataset_root,
        x_col=X_COL,
        y_col=Y_COL,
        batch_size=batch_size,
        seed=shuffle_seed,
        shuffle=False,
        class_mode="categorical",
        target_size=(target_size, target_size))

    return train_generator, valid_generator


def get_test_generator(test_df, batch_size=32, target_size=224, dataset_root=None, use_hsv=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    initialise_tf()

    def to_hsv(x):
        return tensorflow.image.rgb_to_hsv(x)

    if use_hsv:
        datagen = ImageDataGenerator(preprocessing_function=to_hsv)
    else:
        datagen = ImageDataGenerator()

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=dataset_root,
        x_col=X_COL,
        y_col=Y_COL,
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode="categorical",
        target_size=(target_size, target_size))

    return test_generator


def get_antispoof_frame(frame, dataset_root, image_file_name=r"^frame", stratified_name=None):
    frames = []

    for row_dict in frame.to_dict(orient="records"):
        if stratified_name is None:
            frames.append(
                get_file_path_ground_truth_frame(dataset_root, row_dict['directory_path'], row_dict["ground_truth"],
                                                 image_file_name))
        else:
            frames.append(
                get_file_path_ground_truth_frame(dataset_root, row_dict['directory_path'], row_dict["ground_truth"],
                                                 image_file_name, row_dict[stratified_name]))

    return pd.concat(frames)

def get_aug_antispoof_frame(frame, dataset_root, image_file_name=r"^frame", stratified_name_list_func=None):
    frames = []

    for row_dict in frame.to_dict(orient="records"):
        if stratified_name_list_func is None:
            frames.append(
                get_aug_file_path_ground_truth_frame(dataset_root, row_dict['directory_path'], row_dict["ground_truth"],
                                                 image_file_name))
        else:
            frames.append(
                get_aug_file_path_ground_truth_frame(dataset_root, row_dict['directory_path'], row_dict["ground_truth"],
                                                 image_file_name, stratified_name_list_func(row_dict['attack_category'])))

    return pd.concat(frames)


def get_random_selection_on_aug_category(aug_frame, categories, aug_root, num_aug_images, seed=None, stratified_name_list_func=None, use_last_only=False):
    frames = []
    if use_last_only: # to only augment with N
        categories = [categories[-1]]
    # if "-" in categories:
    #     categories = categories.split('-')

    for category in categories:
        category_frame = aug_frame.loc[aug_frame['attack_category'] == category]
        if category_frame.shape[0] <= 0:
            raise TypeError(f"No augmentation has been done for category: {category}")
        category_frame = get_aug_antispoof_frame(category_frame, aug_root, image_file_name=r"^frame",stratified_name_list_func=stratified_name_list_func)
        if seed is not None:
            np.random.seed(seed)
        if category_frame.shape[0] < num_aug_images:
            raise TypeError(
                f"Cannot facilitate {num_aug_images} augmentation images. Only {category_frame.shape[0]} generated images are present.")
        rows = np.random.choice(category_frame.index.values, num_aug_images, replace=False)
        frames.append(category_frame.loc[rows])
    return pd.concat(frames)


def make_attack_category_frame(frame, categories):
    frames = []
    for category in categories:
        frames.append(get_dataframe_by_attack_category(frame, category))

    return pd.concat(frames)


def make_medium_name_frame(frame, medium_names):
    frames = []
    for medium_name in medium_names:
        frames.append(get_dataframe_by_medium_name(frame, medium_name))

    return pd.concat(frames)


if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import RMSprop

    image_size = 32
    initialise_tf()
    train_frame = obtain_datagen_frame("/home/jarred/Documents/Datasets/CASIA_KF", "casia.csv", TRAIN_USAGE_TYPE)
    train_generator, valid_generator = get_train_validation_generator(train_frame, target_size=image_size)

    # NB doing something like this will require you to reset the generator
    # images, labels = next(train_generator)
    # train_generator.reset()
    #
    # class_names = list(train_generator.class_indices.keys())
    # labels = [class_names[np.argmax(i)] for i in labels]
    # create_image_grid(images, 4,must_show=True, class_names=labels)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(RMSprop(lr=0.0001, decay=1e-6), loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    # STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    model.fit(x=train_generator,
              steps_per_epoch=STEP_SIZE_TRAIN,
              validation_data=valid_generator,
              validation_steps=STEP_SIZE_VALID,
              epochs=2
              )
    predicted = np.argmax(model.predict(valid_generator, STEP_SIZE_VALID, verbose=1), axis=1)
    ground_truth = valid_generator.classes

    determine_spoof_metrics(ground_truth, predicted, save_dir="./temp/", must_show=True)
    temp = 1
