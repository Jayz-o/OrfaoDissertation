import os

import numpy as np

from Helpers.image_helper import obtain_file_paths

MODE = None
# MODE = 0
# os.environ["CUDA_VISIBLE_DEVICES"]=f"{MODE}"
#os.environ["CUDA_VISIBLE_DEVICES"]=f"0"

import pandas as pd

from DatasetProcessing.DatasetCreators.SIWCreator.ray_siw_dataset_creator import DATASET_CSV_NAME
from DatasetProcessing.KeyFrameExtraction.KFHelpers.keyframe_helper import create_kf_dataset, determine_variational_k

if __name__ == "__main__":
    # dataset root is the path to the siw dataset containing the extracted face frames
    dataset_root = "/home/jarred/Documents/Datasets/SIW"
    save_root = "/home/jarred/Documents/temp_datasets/SIW_KF/"

    # place the tune directory by the saved dataset directory
    dataset_name = os.path.splitext(DATASET_CSV_NAME)[0]
    save_root_pred = os.path.dirname(save_root)
    tune_root = os.path.join(save_root_pred, f"{dataset_name}_kf_siw_tune")

    # feature extractor settings. VggFace uses 10G of space
    # tune_extraction_gpu = 1.0
    tune_extraction_gpu = 1.0
    tune_extraction_cpu = 3
    must_redo_feature_extraction = False
    must_use_separate_extraction_process = False

    # centroid frames settings. When k>200 gpu mem used is 4517 MiB. 0<k<40= 933, 40<k<100 =1445, 280<k<300=7440
    # 100<k<150=2469, 150<k<200=4517
    # tune_kf_gpu = 0.125  # 8 processes

    if MODE == 0:
        tune_kf_gpu = 0.2 # 6 processes
    elif MODE == 1:
        tune_kf_gpu = 0.25 # 6 processes
    else:
        tune_kf_gpu = 0.2
    # tune_kf_gpu = 0.125  # 6 processes
    # tune_kf_gpu = 1.0 /5 # 4 processes
    tune_kf_cpu = 3
    must_redo_kf_extraction = False
    must_use_gpu = True
    min_kf_threshold = None
    max_k = None
    enable_early_stop = True
    early_stop_check_k = None
    must_use_separate_kf_process = False
    # SIW 90 is in the test set so we cannot use the training information
    use_only_training_information=False

    min_thresh_function = None#determine_variational_k
    mean_col_name = "medium_name"

    df = pd.read_csv(os.path.join(dataset_root, "siw.csv"))
    all_videos=None

    def split_df(df):
        if len(df) % 2 != 0:
            df = df.iloc[:-1, :]
        df1, df2 = np.array_split(df, 2)
        return df1, df2
    df1, df2 = split_df(df)
    if MODE == 1:
        all_videos = df1['directory_path'].tolist()
    elif MODE == 0:
        all_videos = df2['directory_path'].tolist()
    else:
        all_videos = None #df.sort_values('frames_present', ascending=True)['directory_path'].tolist()
    # all_videos = df.sort_values('frames_present', ascending=False)['directory_path'].tolist()

    create_kf_dataset(dataset_root, save_root, tune_root, DATASET_CSV_NAME, mean_col_name, min_thresh_function, tune_extraction_gpu,
                      tune_extraction_cpu, must_redo_feature_extraction,
                      must_use_separate_extraction_process,
                      tune_kf_gpu, tune_kf_cpu, must_redo_kf_extraction, must_use_gpu,
                      min_kf_threshold, max_k, enable_early_stop, early_stop_check_k,
                      must_use_separate_kf_process, is_ray=True, all_video_files=all_videos,use_only_training_information=use_only_training_information)
