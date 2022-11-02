import os
import shutil
from datetime import datetime
import pandas as pd
from ray import tune
from DatasetProcessing.helper import move_error_files
from Helpers.image_helper import obtain_file_paths, save_image

os.environ['TUNE_RESULT_DELIM'] = '/'
VIDEO_CSV_NAME = "info.csv"

def begin_processing(config):
    video_path = config['video_path']
    save_root_directory = config['save_root']
    csv_name = config['csv_name']
    detect_video_face_func = config['detect_video_face_func']
    produce_info_csv = config['produce_info_csv_func']
    verbose = config["verbose"]
    # create the save root directory if it does not exist
    if not os.path.exists(save_root_directory):
        os.makedirs(save_root_directory)
    if verbose:
        print(f"Processing: {video_path}")

    # obtain the detected faces
    video_information = detect_video_face_func(video_path)

    # produce the information dictionary
    info_dic = produce_info_csv(video_path, video_frames=video_information['video_frames'],
                                frames_present=video_information['aligned_faces_count'])
    # handle any errors
    if video_information['aligned_faces_count'] <= 1:
        error_name = f"ERROR_LOCATING_FACE_{info_dic['directory_path'].replace('/', '_')}.txt"
        with open(os.path.join(save_root_directory, error_name), 'w') as file:
            message = f"Could not locate faces in video: {video_path}"
            file.write(message)
            if verbose:
                print(message)
            error_video_path = os.path.join(save_root_directory, "VIDEO_ERRORS")
            if not os.path.exists(error_video_path):
                os.makedirs(error_video_path)
            video_name = os.path.basename(video_path)
            destination = os.path.join(error_video_path, video_name)
            shutil.copy(video_path, destination)

    else:
        # save the images
        save_path = os.path.join(save_root_directory, info_dic["directory_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, image in enumerate(video_information['aligned_faces']):
            save_image(image, os.path.join(save_path, f"frame_{i}.png"))

        # save the information dictionary
        df = pd.DataFrame([info_dic])

        csv_path = os.path.join(save_path, csv_name)

        # overwrite any saved csv
        if os.path.exists(csv_path):
            os.remove(csv_path)

        df.to_csv(csv_path, index=False)

        del csv_path
        del save_path
        del df

    # clean up variables
    del video_information
    del info_dic


def combine_csvs(csv_files, dataset_root, save_name):
    if len(csv_files) <=0:
        error_name = f"ERROR_COMBINING_CSVS.txt"
        with open(os.path.join(dataset_root, error_name), 'w') as file:
            message = f"The csv files provided was empty."
            file.write(message)
            print(message)
        return

    csvs = []
    for csv_path in csv_files:
        csvs.append(pd.read_csv(csv_path))

    frame = pd.concat(csvs, axis=0, ignore_index=True)
    path = os.path.join(dataset_root, save_name)

    # remove any existing csv file
    if os.path.exists(path):
        os.remove(path)

    # save the csv file
    frame.to_csv(path, index=False)


def create_dataset(dataset_root, save_root, dataset_csv_name, video_name_pattern, produce_info_csv_func,
                   detect_video_face_func, tune_cpu, tune_gpu, must_redo=False, verbose=True):
    # place the tune directory by the saved dataset directory
    if ".csv" in dataset_csv_name:
        dataset_name = os.path.splitext(dataset_csv_name)[0]
    else:
        dataset_name = dataset_csv_name

    save_root_pred = os.path.dirname(save_root)
    tune_root = os.path.join(save_root_pred, f"{dataset_name}_processed_tune")
    tune_kf_csv = f"{dataset_name}_processor.csv"
    tune_experiment_name = f"{dataset_name}_dataset_creator_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

    # create the directories
    if not os.path.exists(tune_root):
        os.makedirs(tune_root)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # move any error files found
    move_error_files(save_root)

    # get all the video file paths
    all_video_files = obtain_file_paths(dataset_root, video_name_pattern)
    video_files_remaining = []
    # if we do not want to re-process a video
    if must_redo == False:
        for video_path in all_video_files:
            # create a temp info dic to obtain the directory path this video would be saved in
            info_dic = produce_info_csv_func(video_path, video_frames=-1, frames_present=-1)
            csv_path = os.path.join(save_root, info_dic['directory_path'], VIDEO_CSV_NAME)
            # if the csv file does not exist
            if not os.path.exists(csv_path):
                # this video still needs to be processed
                video_files_remaining.append(video_path)
    else:
        video_files_remaining = all_video_files

    print("~~~~~~~~~~~~~~~~~~ PASS 1: Face Detection ~~~~~~~~~~~~~~~~~~")
    if len(video_files_remaining) > 0:
        # begin_processing({
        #     "video_path": "/home/jarred/Documents/Datasets/SIW_ERROR_Videos/Train/020-2-3-3-2.mov",
        #     "save_root": save_root,
        #     "csv_name": VIDEO_CSV_NAME,
        #     "verbose": verbose,
        #     "detect_video_face_func": detect_video_face_func,
        #     "produce_info_csv_func": produce_info_csv_func,
        # })
        # create the tune config
        tune_config = {
            "video_path": tune.grid_search(video_files_remaining),
            "save_root": save_root,
            "csv_name": VIDEO_CSV_NAME,
            "verbose": verbose,
            "detect_video_face_func": detect_video_face_func,
            "produce_info_csv_func": produce_info_csv_func,
        }

        # start ray tune
        analysis = tune.run(begin_processing, config=tune_config, local_dir=tune_root, name=tune_experiment_name,
                            resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
        df = analysis.results_df
        df.to_csv(os.path.join(tune_root, tune_kf_csv))

    print("~~~~~~~~~~~~~~~~~~ PASS 2: CSV Aggregation ~~~~~~~~~~~~~~~~~~")
    # loop through directories and get all csv files
    csv_files = obtain_file_paths(save_root, VIDEO_CSV_NAME)
    # aggregate csvs
    combine_csvs(csv_files, save_root, dataset_csv_name)
