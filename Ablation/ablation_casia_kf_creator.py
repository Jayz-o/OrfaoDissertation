import os

from DatasetProcessing.DatasetCreators.CASIACreator.casia_helper import CASIA_VIDEO_TO_ATTACK_CATEGORY_DIC
from DatasetProcessing.DatasetCreators.CASIACreator.ray_casia_dataset_creator import DATASET_CSV_NAME
from DatasetProcessing.KeyFrameExtraction.KFHelpers.keyframe_helper import create_kf_dataset
from DatasetProcessing.KeyFrameExtraction.KFHelpers.trend_viewer import view_frames_for_folder

if __name__ == "__main__":
    #changes
    folder_name = "test/real/18/1"
    min_kf_threshold = 8
    save_root = "/home/jarred/Downloads/Temp/"

    # dataset root is the path to the siw dataset containing the extracted face frames
    dataset_root = "/home/jarred/Documents/Datasets/CASIA"

    # place the tune directory by the saved dataset directory
    dataset_name = os.path.splitext(DATASET_CSV_NAME)[0]
    save_root_pred = os.path.dirname(save_root)
    tune_root = os.path.join(save_root_pred, f"{dataset_name}_kf_casia_tune")

    # feature extractor settings. VggFace uses 10G of space
    tune_extraction_gpu = 0.5
    tune_extraction_cpu = 5
    must_redo_feature_extraction = False
    must_use_separate_extraction_process = False

    # centroid frames settings. You can start with 6 processes for the first run, then reduce to 4 to finish
    # tune_kf_gpu = 0.1667   #   6 processes
    tune_kf_gpu = 0.25  # 4 processes
    tune_kf_cpu = 2
    must_redo_kf_extraction = False
    must_use_gpu = True
    max_k = None
    enable_early_stop = True
    early_stop_check_k = None
    must_use_separate_kf_process = False

    subject = os.path.basename(os.path.dirname(folder_name))
    video = os.path.basename(folder_name)

    grid_name = f"S{subject}_V{video}_k{min_kf_threshold}_grid.png"
    title = f"Subject {subject} {CASIA_VIDEO_TO_ATTACK_CATEGORY_DIC[video+'.avi']} Min K = {min_kf_threshold}"
    create_kf_dataset(dataset_root, save_root, tune_root, DATASET_CSV_NAME, tune_extraction_gpu,
                      tune_extraction_cpu, must_redo_feature_extraction,
                      must_use_separate_extraction_process,
                      tune_kf_gpu, tune_kf_cpu, must_redo_kf_extraction, must_use_gpu,
                      min_kf_threshold, max_k, enable_early_stop, early_stop_check_k,
                      must_use_separate_kf_process, all_video_files=[folder_name])#all_video_files[:1])
    save_location = os.path.join(save_root, folder_name)
    look_location = os.path.join(dataset_root, folder_name)
    view_frames_for_folder(look_location, os.path.join(save_location, grid_name), None, 8)
