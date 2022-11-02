import os.path
import shutil
from datetime import datetime

import pandas as pd
from ray import tune

from Antispoofing.AntispoofHelpers.dataset_helper import make_medium_name_frame, make_attack_category_frame, \
    get_dataframe_by_attack_category
from DatasetProcessing.DatasetCreators.FaceHelpers.face_detection_helper import detect_face_in_folder
from Helpers.image_helper import obtain_file_paths, save_image
from NVIDIA_STYLEGAN3.gen_images import generate_images
from constants import GENERATED_IMAGE_FOLDER_ROOT, BEST_GAN_FOLDER_ROOT


def _generate_images(args):
    try:
        generate_images(args)
    except Exception as e:
        print(e)
    except:
        print("catch all")
        # print(e)


def generate_gan_images(dataset_root, dataset_csv, num_train_images_func, output_root=GENERATED_IMAGE_FOLDER_ROOT,
                        subject_number=None, aug_percentage=0.4, trunc_psi=1, is_ray=True, dataset_csv_root=None,
                        ignore_detection=False):
    tune_cpu = 3
    tune_gpu = 0.25
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
            "dataset_csv_root":dataset_csv_root,
            "dataset_csv": dataset_csv,
            "num_train_images_func": num_train_images_func,
            "aug_percentage": aug_percentage,
            "trunc_psi": trunc_psi,
            "dataset_root": dataset_root,
            "best_models_root": best_models_root,
            "output_root": output_root,
            "subject_number": subject_number,
            "ignore_detection":ignore_detection,
        }
        tune_experiment_name = f"{dataset_name}_image_generator"#_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
        tune.run(_generate_category, config=config, local_dir=output_root, name=tune_experiment_name,
                 resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
                 # resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="ERRORED_ONLY")
    else:
        for record in best_model_frame.to_dict(orient="records"):
            config = {

                "row_dict": record,
                "dataset_csv": dataset_csv,
                "dataset_csv_root": dataset_csv_root,
                "num_train_images_func": num_train_images_func,
                "aug_percentage": aug_percentage,
                "trunc_psi": trunc_psi,
                "dataset_root": dataset_root,
                "best_models_root": best_models_root,
                "output_root": output_root,
                "subject_number": subject_number,
                "ignore_detection": ignore_detection,
            }
            _generate_category(config)
    csv_files = obtain_file_paths(output_root, "^aug_info.csv")
    combine = []
    for path in csv_files:
        combine.append(pd.read_csv(path))
    combined = pd.concat(combine)
    combined.to_csv(os.path.join(output_root, f"{best_model_dataset_folder_name}.csv"), index=False)

def _generate_category(config):
    row_dict = config["row_dict"]
    # row_dict = config[ "row_dict"][0]
    dataset_csv = config["dataset_csv"]
    num_train_images_func= config["num_train_images_func"]
    aug_percentage=config[ "aug_percentage"]
    trunc_psi=config["trunc_psi"]
    dataset_root=config["dataset_root"]
    best_models_root=config["best_models_root"]
    output_root = config['output_root']
    subject_number = config['subject_number']
    dataset_csv_root = config['dataset_csv_root']
    ignore_detection = config['ignore_detection']
    if dataset_csv_root is not None:
        dataset_root = dataset_csv_root
    # (Itot * aug %) / (1 - aug %)
    num_train_images = num_train_images_func(dataset_root, dataset_csv, row_dict['attack_category'], subject_number)

    num_images_to_generate = int(round((num_train_images * aug_percentage) / (1 - aug_percentage)))
    # num_images_to_generate=2
    # num_train_images = 7200
    # num_images_to_generate = 7200
    model_path = os.path.join(best_models_root, row_dict['model_path'])
    error_save_path = os.path.join(output_root, os.path.dirname(row_dict['model_path']) + "_ANOMALIES")
    temp_save_path = os.path.join(output_root, os.path.dirname(row_dict['model_path']) + "_TEMP")
    save_path = os.path.join(output_root, os.path.dirname(row_dict['model_path']))
    if os.path.exists(save_path):
        return
    start_seed = 1
    end_seed = num_images_to_generate
    successful_generated = 0

    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path)

    if os.path.exists(save_path):
        return

    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    # if os.path.exists(error_save_path):
    #     shutil.rmtree(error_save_path)
    generated_counter = 0
    while successful_generated != num_images_to_generate:
        args = [
            f"--network={model_path}",
            f"--seeds={start_seed}-{end_seed}",
            f"--outdir={temp_save_path}",
            f"--trunc={trunc_psi}",
        ]
        _generate_images(args)

        # process the generated images
        # video_information = detect_face_in_folder(temp_save_path, zoom_factor=0.23, vertical_face_location_shift=25)
        video_information = detect_face_in_folder(temp_save_path, zoom_factor=0.31, vertical_face_location_shift=50, ignore_detection=ignore_detection)
        # save the images

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(error_save_path):
            os.makedirs(error_save_path)

        # for i, image in enumerate(video_information['aligned_faces']):
        for i, image in enumerate(video_information['non_error_frames']):
            save_image(image, os.path.join(save_path, f"frame_{generated_counter}.png"))
            generated_counter += 1


        successful_generated += len(video_information['non_error_frames'])
        start_seed = end_seed + 1
        end_seed += (num_images_to_generate - successful_generated)
        for error in video_information['error_frames']:
            destination_name = os.path.join(error_save_path, os.path.basename(error))
            shutil.move(error, destination_name)
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)
    temp_dic = row_dict.copy()
    temp_dic['directory_path'] = os.path.dirname(row_dict['model_path']) + "/"
    temp_dic['frames_present'] = num_images_to_generate
    temp_dic['aug_ratio'] = num_images_to_generate / (num_train_images + num_images_to_generate)
    info_frame = pd.DataFrame.from_dict([temp_dic])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    info_frame.to_csv(os.path.join(save_path, 'aug_info.csv'), index=False)

def casia_num_train_images_func(dataset_root, dataset_csv, folder_name, subject_number):
    categories = [folder_name]#folder_name.split("|")
    dataset_frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is None:
        dataset_frame = dataset_frame.query("usage_type == 'train'")
    else:
        dataset_frame = dataset_frame.query(f"subject_number == {subject_number}")

    if "1" in folder_name:
        categories.append("N1")
    if "2" in folder_name:
        categories.append("N2")
    if "HR_" in folder_name:
        categories.append("HR_N")


    # return dataset_frame['frames_present'].sum()
    category_frame = make_attack_category_frame(dataset_frame, categories)
 #
    return category_frame['frames_present'].sum()




def generate_casia_images(dataset_root, dataset_csv, is_ray, ignore_detection,dataset_csv_root=None,):
    generate_gan_images(dataset_root, dataset_csv, casia_num_train_images_func, is_ray=is_ray, ignore_detection=ignore_detection, dataset_csv_root=dataset_csv_root)


def siw_num_train_images_func(dataset_root, dataset_csv, folder_name, subject_number=None, must_return_frame=False):
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
    category_frame= pd.concat([real_frame, category_frame])
    if must_return_frame:
        return category_frame
    else:
        return category_frame['frames_present'].sum()


def generate_siw_images_for_subject(dataset_root, dataset_csv, subject_number=None, dataset_csv_root=None, is_ray=False, ignore_detection=False):
    generate_gan_images(dataset_root, dataset_csv, siw_num_train_images_func, subject_number=subject_number, dataset_csv_root=dataset_csv_root, is_ray=is_ray, ignore_detection=ignore_detection)
