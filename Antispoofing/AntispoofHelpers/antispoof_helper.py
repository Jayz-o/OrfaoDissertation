import os
import shutil

from Antispoofing.AntispoofHelpers.hyper_perameter_helper import start_antispoofing
from Antispoofing.process_antispoofing import copy_csv_files
from constants import ANTISPOOFING_SAVED_FOLDER_ROOT, DATASETS_FOLDER_ROOT, \
    GENERATED_IMAGE_FOLDER_ROOT, SAVED_CHECKPOINTS_FOLDER, SAVED_SPOOF_METRICS_FOLDER, SAVED_TENSOR_BOARD_FOLDER, \
    SAVED_TUNE_FOLDER, TRADITIONAL_IMAGE_FOLDER_ROOT, GENERATED_TRADITIONAL_IMAGE_FOLDER_ROOT, DOCUMENTS


def antispoof_setup(dataset_name, dataset_csv_name, get_train_frame_func, get_protocol_frame_dic_func, get_stratified_name_col_func,process_dataset_metrics_func,
                     aug_folder_combinations, tune_cpu, tune_gpu, epochs=20, train_subject_number=None, test_subject_number=None,
                    tune_experiment_name=None, aug_percentages=None, num_run_repeats=1, is_ray=True, original_dataset_name=None,
                    stratified_name_list_func=None, is_traditional=False, use_last_only=False, aug_dataset_name=None, save_path=None,
                    is_single_folder=True, include_traditional_aug=False, aug_after_split=False, must_remove_normal=False, is_gen_with_trad_aug=False, mode_info=None, must_use_normal_only=False, error_only=False, use_hsv=False):
    save_aug_folder_name =""
    if aug_dataset_name is None:
        aug_dataset_name = dataset_name
    else:
        save_aug_folder_name = dataset_name +"_"

    if train_subject_number is not None:
        aug_dataset_name += f"_{train_subject_number}"
    aug_csv = f"{aug_dataset_name}.csv"
    save_aug_folder_name += aug_dataset_name



    # documents/Datasets/CASIA
    dataset_root = os.path.join(DATASETS_FOLDER_ROOT, dataset_name)
    original_dataset_root = None
    if original_dataset_name is not None:
        original_dataset_root = os.path.join(DATASETS_FOLDER_ROOT, original_dataset_name)

    if is_gen_with_trad_aug and aug_percentages is not None:
        aug_root = os.path.join(GENERATED_TRADITIONAL_IMAGE_FOLDER_ROOT, aug_dataset_name)
    elif is_traditional and aug_percentages is not None:
        # documents/TraditionalAugmentation/CASIA
        aug_root = os.path.join(TRADITIONAL_IMAGE_FOLDER_ROOT, aug_dataset_name)
    else:
        # documents/SavedAntispoofing/CASIA
        aug_root = os.path.join(GENERATED_IMAGE_FOLDER_ROOT, aug_dataset_name)


    if mode_info is not None:
        save_aug_folder_name+=mode_info

    when_aug = ""
    if aug_after_split:
        when_aug = "AfterSplit"
    else:
        when_aug = "BeforeSplit"


    if save_path is None:
        save_root = os.path.join(ANTISPOOFING_SAVED_FOLDER_ROOT, save_aug_folder_name)
    else:
        save_root = os.path.join(save_path, save_aug_folder_name)

    save_checkpoints_root = os.path.join(save_root, when_aug, SAVED_CHECKPOINTS_FOLDER)
    save_metrics_root = os.path.join(save_root, when_aug, SAVED_SPOOF_METRICS_FOLDER)
    save_tb_root = os.path.join(save_root, when_aug, SAVED_TENSOR_BOARD_FOLDER)
    save_tune_root = os.path.join(save_root, when_aug, SAVED_TUNE_FOLDER)

    repeat_run_list = [i for i in range(1, num_run_repeats+1)]

    start_antispoofing(dataset_root, dataset_csv_name, aug_root, aug_csv, save_metrics_root, save_checkpoints_root,
                       save_tb_root, save_tune_root, aug_folder_combinations, tune_gpu, tune_cpu, epochs,
                       get_train_frame_func, get_protocol_frame_dic_func,get_stratified_name_col_func,process_dataset_metrics_func, tune_experiment_name,
                       aug_percentages, repeat_run_list, train_subject_number,
                       test_subject_number,is_ray=is_ray, original_dataset_root=original_dataset_root, stratified_name_list_func=stratified_name_list_func,
                       is_traditional=is_traditional, use_last_only=use_last_only, is_single_folder=is_single_folder,
                       include_traditional_aug=include_traditional_aug, aug_after_split=aug_after_split, must_remove_normal=must_remove_normal, mode_info=mode_info,
                       must_use_normal_only=must_use_normal_only, error_only=error_only, use_hsv=use_hsv)


    copy_csv_files(ANTISPOOFING_SAVED_FOLDER_ROOT, DOCUMENTS)
    shutil.make_archive(os.path.join(DOCUMENTS, "DownloadResults"), 'zip', os.path.join(DOCUMENTS,"MasterMetrics"))
