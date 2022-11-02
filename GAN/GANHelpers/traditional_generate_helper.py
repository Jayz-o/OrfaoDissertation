import math
import os.path

import random
import shutil
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
from ray import tune


from Antispoofing.AntispoofHelpers.dataset_helper import make_medium_name_frame, make_attack_category_frame, \
    get_dataframe_by_attack_category
from DatasetProcessing.DatasetCreators.FaceHelpers.face_detection_helper import detect_face_in_folder
from Helpers.image_helper import obtain_file_paths, save_image, initialise_tf, load_image_from_file
from NVIDIA_STYLEGAN3.gen_images import generate_images
from constants import GENERATED_IMAGE_FOLDER_ROOT, BEST_GAN_FOLDER_ROOT, TRADITIONAL_IMAGE_FOLDER_ROOT


def _generate_images(args):
    try:
        generate_images(args)
    except:
        print("catch all")


def generate_traditional_images(dataset_root, dataset_csv, num_train_images_func, output_root=TRADITIONAL_IMAGE_FOLDER_ROOT,
                        subject_number=None, aug_percentage=0.4, is_ray=True):
    tune_cpu = 3
    tune_gpu = 0.5
    dataset_name = os.path.basename(dataset_root)

    # docs/Generated/Casia

    if subject_number is not None:
        best_model_dataset_folder_name = f"{dataset_name}_{subject_number}"
    else:
        best_model_dataset_folder_name = dataset_name
    output_root = os.path.join(output_root, best_model_dataset_folder_name)
    best_models_root = os.path.join(BEST_GAN_FOLDER_ROOT, best_model_dataset_folder_name)
    best_model_frame = pd.read_csv(os.path.join(best_models_root, f"{best_model_dataset_folder_name}.csv"))

    if is_ray:
        config = {

            "row_dict": tune.grid_search(best_model_frame.to_dict(orient="records")),
            "dataset_csv": dataset_csv,
            "num_train_images_func": num_train_images_func,
            "aug_percentage": aug_percentage,
            "dataset_root": dataset_root,
            "output_root": output_root,
            "subject_number": subject_number,
        }
        tune_experiment_name = f"{dataset_name}_image_generator"#_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
        tune.run(_generate_category, config=config, local_dir=output_root, name=tune_experiment_name,
                 resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
                 # resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="ERRORED_ONLY")
    else:
        items = best_model_frame.to_dict(orient="records")
        for item in items:
            config = {

                "row_dict": item,
                # "row_dict": {'fid': 115.13280934940124, 'kimg': 960, 'attack_category': 'ASUS-IP7P-IPP2017', 'model_path': 'ASUS-IP7P-IPP2017/best_model.pkl', 'ground_truth': 'spoof'},
                "dataset_csv": dataset_csv,
                "num_train_images_func": num_train_images_func,
                "aug_percentage": aug_percentage,
                "dataset_root": dataset_root,
                "best_models_root": best_models_root,
                "output_root": output_root,
                "subject_number": subject_number,
            }
            _generate_category(config)
    csv_files = obtain_file_paths(output_root, "^aug_info.csv")
    combine = []
    for path in csv_files:
        combine.append(pd.read_csv(path))
    combined = pd.concat(combine)
    combined.to_csv(os.path.join(output_root, f"{best_model_dataset_folder_name}.csv"), index=False)


def augment_folder(folder_root, save_root, is_ray=False):
    tune_cpu = 4
    tune_gpu = 0.33

    all_folders =  next(os.walk(folder_root))[1]
    interested_folders = []
    for folder in all_folders:
        if "ANOM" not in folder and 'SIW_image_generator' not in folder:
            interested_folders.append(os.path.join(folder_root, folder))
    dataset_name = os.path.basename(folder_root)
    if is_ray:
        config = {
            "folder_path": tune.grid_search(interested_folders),
            "save_root": save_root,
        }
        tune_experiment_name = f"{folder_root}_traditional_aug"#_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
        tune.run(_augment_folder, config=config, local_dir=save_root, name=tune_experiment_name,
                 resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
                 # resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="ERRORED_ONLY")
    else:
        for item in interested_folders:
            config = {
                "folder_path": item,
                "save_root": save_root,
            }
            _augment_folder(config)
    csv_files = obtain_file_paths(save_root, "^aug_info.csv")
    combine = []
    for path in csv_files:
        combine.append(pd.read_csv(path))
    combined = pd.concat(combine)

    combined.to_csv(os.path.join(save_root, f"{dataset_name}.csv"), index=False)







def _augment_folder(config):
    folder_path = config['folder_path']
    folder_name = os.path.basename(folder_path)

    save_root = os.path.join(config['save_root'], folder_name)
    all_file_paths = obtain_file_paths(folder_path, "^frame")
    initialise_tf()
    import tensorflow as tf
    degrees = 15
    percentage = (degrees * (math.pi / 180.0)) / (2 * math.pi)
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(percentage),
            tf.keras.layers.RandomZoom((-0.2, 0)),
        ]
    )



    num_parts = 7
    chunk_size = int(len(all_file_paths)/num_parts)
    all_images = []
    for i in range(1,num_parts+1):
        start_index = (i-1) * chunk_size
        if i == num_parts:
            all_images.append(all_file_paths[start_index:])
        else:
            end_index = i * chunk_size
            all_images.append(all_file_paths[start_index:end_index])

    image_counter = 0
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for file_paths in all_images:
        images = [load_image_from_file(file_path, desired_shape=(224, 224)) for
                  file_path in file_paths]

        augmented = data_augmentation(np.array(images)).numpy()

        for image in augmented:
            save_image(image, os.path.join(save_root, f"frame_{image_counter}.png"))
            image_counter += 1

    shutil.copy(os.path.join(folder_path, 'aug_info.csv'), os.path.join(save_root, 'aug_info.csv'))




def _generate_category(config):
    initialise_tf()
    import tensorflow as tf
    row_dict = config[ "row_dict"]
    dataset_csv = config["dataset_csv"]
    num_train_images_func= config["num_train_images_func"]
    aug_percentage=config[ "aug_percentage"]
    dataset_root=config["dataset_root"]
    output_root = config['output_root']
    subject_number = config['subject_number']
    # (Itot * aug %) / (1 - aug %)
    save_path = os.path.join(output_root, os.path.dirname(row_dict['model_path']))
    if os.path.exists(save_path):
        return
    combined_frame, category_frame = num_train_images_func(dataset_root, dataset_csv, row_dict['attack_category'], subject_number, )
    num_train_images = combined_frame['frames_present'].sum()

    num_images_to_generate = int(round( ((num_train_images * aug_percentage) / (1 - aug_percentage))/category_frame['attack_category'].unique().shape[0]))
    image_counter = 0
    # num_images_to_generate = 10
    degrees = 15
    percentage = (degrees * (math.pi/180.0) )/ (2*math.pi)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(percentage, fill_mode="constant"),
            tf.keras.layers.RandomZoom((-0.2, 0)),
        ]
    )

    for row in category_frame.to_dict(orient="records"):
        # get the frames for the directory
        file_paths = obtain_file_paths(os.path.join(dataset_root,row['directory_path']), "^frame")
        random.shuffle(file_paths)
        images = [load_image_from_file(file_path, desired_shape=(224,224)) for
                  file_path in file_paths]
        # flipped_images = tf.image.flip_left_right(images).numpy()
        augmented = []
        # num_images_to_generate = int(round(((len(file_paths) * aug_percentage) / (1 - aug_percentage)) /
        #                                    category_frame['attack_category'].unique().shape[0])*4)
        for gen_counter in range(0, num_images_to_generate):
            augmented.append(data_augmentation(np.array(images[gen_counter%len(images)])).numpy())
        for image in augmented:
            save_image(image, os.path.join(save_path, f"frame_{image_counter}.png"))
            image_counter += 1
        # break
    temp_dic = row_dict.copy()
    temp_dic['directory_path'] = os.path.dirname(row_dict['model_path']) + "/"
    temp_dic['frames_present'] = image_counter
    temp_dic['aug_ratio'] = image_counter/(num_train_images +image_counter)
    info_frame = pd.DataFrame.from_dict([temp_dic])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    info_frame.to_csv(os.path.join(save_path, 'aug_info.csv'), index=False)

# def casia_num_train_images_func(dataset_root, dataset_csv, folder_name, subject_number):
#     categories = [folder_name]#folder_name.split("|")
#     dataset_frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
#     if subject_number is None:
#         dataset_frame = dataset_frame.query("usage_type == 'train'")
#     else:
#         dataset_frame = dataset_frame.query(f"subject_number == {subject_number}")
#     category_frame = make_attack_category_frame(dataset_frame, categories)
#     return category_frame['frames_present'].sum()
def casia_num_train_images_func(dataset_root, dataset_csv, folder_name, subject_number, must_return_frame=True):
    categories = [folder_name]#folder_name.split("|")
    dataset_frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is None:
        dataset_frame = dataset_frame.query("usage_type == 'train'")
    else:
        dataset_frame = dataset_frame.query(f"subject_number == {subject_number}")
    combined_categories = categories.copy()
    if "1" in folder_name:
        combined_categories.append("N1")
    if "2" in folder_name:
        combined_categories.append("N2")
    if "HR_" in folder_name:
        combined_categories.append("HR_N")


    # return dataset_frame['frames_present'].sum()
    category_frame = make_attack_category_frame(dataset_frame, categories)

    combined_categories_frame = make_attack_category_frame(dataset_frame, combined_categories)
 #
    if must_return_frame:
        return combined_categories_frame['frames_present'].sum(), category_frame
    else:
        return combined_categories_frame['frames_present'].sum()




def generate_casia_images(dataset_root, dataset_csv):
    generate_traditional_images(dataset_root, dataset_csv, casia_num_train_images_func)


def siw_num_train_images_func(dataset_root, dataset_csv, folder_name, subject_number=None, must_return_frame=True):
    categories = folder_name.split("-")
    dataset_frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is None:
        dataset_frame = dataset_frame.query("usage_type == 'train'")
    else:
        dataset_frame = dataset_frame.query(f"subject_number == {subject_number}")
    if type(categories) is list and len(categories) == 1:
        cat = categories[0]
        if "P" == cat or "R" == cat or "N" == cat:
            category_frame = make_attack_category_frame(dataset_frame, categories)
        else:
            category_frame = make_medium_name_frame(dataset_frame, categories)

    else:
        category_frame = make_medium_name_frame(dataset_frame, categories)
    real_frame = get_dataframe_by_attack_category(dataset_frame, "N")
    combined_category_frame= pd.concat([real_frame, category_frame])
    if must_return_frame:
        return combined_category_frame, category_frame
    else:
        return category_frame['frames_present'].sum()


def generate_siw_images_for_subject(dataset_root, dataset_csv, subject_number=None):
    generate_traditional_images(dataset_root, dataset_csv, siw_num_train_images_func, subject_number=subject_number)

if __name__ == '__main__':
    augment_folder("/home/jarred/Documents/Generated/SIW_90", "/home/jarred/Documents/Generated_Traditional/SIW_90")