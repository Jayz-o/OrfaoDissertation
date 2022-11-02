import os

import pandas as pd

from DatasetProcessing.DatasetCreators.CASIACreator.ray_casia_dataset_creator import DATASET_CSV_NAME
from DatasetProcessing.KeyFrameExtraction.KFHelpers.keyframe_helper import create_kf_dataset, PATH_FEATURE_DIC_NAME, \
    find_max_k, determine_variational_k

if __name__ == "__main__":
    # dataset root is the path to the siw dataset containing the extracted face frames
    dataset_root = "/home/jarred/Documents/Datasets/CASIA"
    save_root = "/home/jarred/Documents/Datasets/CASIA_KF"

    # place the tune directory by the saved dataset directory
    dataset_name = os.path.splitext(DATASET_CSV_NAME)[0]
    save_root_pred = os.path.dirname(save_root)
    tune_root = os.path.join(save_root_pred, f"{dataset_name}_kf_casia_tune")

    # feature extractor settings. VggFace uses 10G of space
    tune_extraction_gpu = 1.0
    tune_extraction_cpu = 5
    must_redo_feature_extraction = False
    must_use_separate_extraction_process = False

    # centroid frames settings. You can start with 6 processes for the first run, then reduce to 4 to finish
    # tune_kf_gpu = 0.1667   #   6 processes
    tune_kf_gpu = 0.25  # 4 processes
    tune_kf_cpu = 3
    must_redo_kf_extraction = False
    must_use_gpu = True
    min_kf_threshold = None
    max_k = None
    enable_early_stop = True
    early_stop_check_k = None
    must_use_separate_kf_process = False
    mean_col_name = "attack_category"
    min_thresh_function = None#determine_variational_k
    ALL_VIDEOS = None#["train/real/1/1/"]

    create_kf_dataset(dataset_root, save_root, tune_root, DATASET_CSV_NAME, mean_col_name, min_thresh_function, tune_extraction_gpu,
                      tune_extraction_cpu, must_redo_feature_extraction,
                      must_use_separate_extraction_process,
                      tune_kf_gpu, tune_kf_cpu, must_redo_kf_extraction, must_use_gpu,
                      min_kf_threshold, max_k, enable_early_stop, early_stop_check_k,
                      must_use_separate_kf_process, is_ray=True, all_video_files=ALL_VIDEOS, use_only_training_information=True)
