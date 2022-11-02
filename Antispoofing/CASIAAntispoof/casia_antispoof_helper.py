import os
import shutil

import pandas as pd

from Antispoofing.AntispoofHelpers.dataset_helper import make_attack_category_frame
from DatasetProcessing.DatasetCreators.CASIACreator.casia_helper import convert_attack_category_to_medium_list

LOW_QUALITY_CATEGORIES = ['N1', 'R1', 'W1', 'C1']
NORMAL_QUALITY_CATEGORIES = ['N2', 'R2', 'W2', 'C2']
HIGH_QUALITY_CATEGORIES = ['HR_N', 'HR_R', 'HR_W', 'HR_C']
WARPED_PHOTO_CATEGORIES = ['N1', 'N2', 'HR_N', 'W1', 'W2', 'HR_W']
CUT_PHOTO_CATEGORIES = ['N1', 'N2', 'HR_N', 'C1', 'C2', 'HR_C']
VIDEO_PHOTO_CATEGORIES = ['N1', 'N2', 'HR_N', 'R1', 'R2', 'HR_R']
ALL_CATEGORIES = ['HR_N', 'HR_W', 'HR_C', 'HR_R', 'N1', 'N2', 'R1', 'R2', 'W1', 'W2', 'C1', 'C2']

ALL_CATEGORIES_LIST = [LOW_QUALITY_CATEGORIES, NORMAL_QUALITY_CATEGORIES, HIGH_QUALITY_CATEGORIES,
                    WARPED_PHOTO_CATEGORIES, CUT_PHOTO_CATEGORIES, VIDEO_PHOTO_CATEGORIES, ALL_CATEGORIES]


def get_casia_stratified_name_col(combinations):
    return None


def get_train_casia_frame_func(dataset_root, dataset_csv_name, combinations, train_subject_number):
    frame = pd.read_csv(os.path.join(dataset_root, dataset_csv_name))
    if train_subject_number is not None:
        frame = frame.query(f"subject_number == {train_subject_number}")
    else:
        frame = frame.query("usage_type == 'train'")
    attack_frame = make_attack_category_frame(frame, combinations)
    return attack_frame



def get_casia_test_frame(dataset_name, dataset_csv_name, test_subject_number):

    frame = pd.read_csv(os.path.join(dataset_name, dataset_csv_name))
    return frame.query("usage_type == 'test'")


def get_casia_protocol_frame_dic(dataset_name, dataset_csv_name, combinations, test_subject_number):
    test_frame = get_casia_test_frame(dataset_name, dataset_csv_name, test_subject_number)
    protocol_frame_dic = {
        tuple(LOW_QUALITY_CATEGORIES): {'Low_Quality': {"protocol_number": 1,
                                                 "frame": make_attack_category_frame(test_frame,
                                                                                     LOW_QUALITY_CATEGORIES)}},
       tuple(NORMAL_QUALITY_CATEGORIES): {'Normal_Quality': {"protocol_number": 2,
                                                       "frame": make_attack_category_frame(test_frame,
                                                                                           NORMAL_QUALITY_CATEGORIES)}},
        tuple(HIGH_QUALITY_CATEGORIES): {'High_Quality': {"protocol_number": 3,
                                                   "frame": make_attack_category_frame(test_frame,
                                                                                       HIGH_QUALITY_CATEGORIES)}},
        tuple(WARPED_PHOTO_CATEGORIES): {'Warped_Photo_Attack': {"protocol_number": 4,
                                                          "frame": make_attack_category_frame(test_frame,
                                                                                              WARPED_PHOTO_CATEGORIES)}},
        tuple(CUT_PHOTO_CATEGORIES): {'Cut_Photo_Attack': {"protocol_number": 5,
                                                    "frame": make_attack_category_frame(test_frame,
                                                                                        CUT_PHOTO_CATEGORIES)}},
        tuple(VIDEO_PHOTO_CATEGORIES): {'Video_Replay_Attack': {"protocol_number": 6,
                                                         "frame": make_attack_category_frame(test_frame,
                                                                                             VIDEO_PHOTO_CATEGORIES)}},
        tuple(ALL_CATEGORIES): {'All_Categories': {"protocol_number": 7,
                                            "frame": make_attack_category_frame(test_frame, ALL_CATEGORIES)}},
    }
    return protocol_frame_dic[tuple(combinations)]


def process_casia_dataset_metrics_func(df, save_root):
    selected_df = df[
        ["experiment_tag", "protocol", "protocol_number", "fold_number", "config/HP_AUG_PER", "config/HP_REPEAT",
         "APCER", "BPCER", "ACER", "AUC", "EER", "TN","time_total_s",
         "TP", "FN", "FP"]].sort_values(by=["protocol_number", "ACER"])
    selected_df.to_csv(os.path.join(save_root, "selected_metrics.csv"))
    descriptive_metrics = selected_df.groupby(["config/HP_AUG_PER", "protocol_number"]).describe(percentiles=[])
    descriptive_metrics.to_csv(os.path.join(save_root, "descriptive_metrics.csv"))
    quick_metrics = descriptive_metrics[
        [("APCER", "mean"), ("APCER", "std"), ("BPCER", "mean"), ("BPCER", "std"), ("ACER", "mean"), ("ACER", "std"),
         ("EER", "mean"), ("EER", "std"), ("AUC", "mean"), ("AUC", "std"), ("time_total_s", "mean")]]
    quick_metrics.to_csv(os.path.join(save_root, "quick_metrics.csv"))


def get_casia_stratified_name_col(combinations):
    return "attack_category"

def get_casia_stratified_name_list_func(attack_category):
    return convert_attack_category_to_medium_list(attack_category)


if __name__ == "__main__":
    save_root = "/home/jarred/Documents/SavedAntispoofing/SIW_90/Tune/Baseline/SIW_antispoof_06_30_2022_14_46_55"
    process_casia_dataset_metrics_func(pd.read_csv(os.path.join(save_root, "SIW_antispoof_tune.csv")),save_root)
    pass
    # dataset_root = "/home/jarred/Documents/Datasets/CASIA_KF"
    # dataset_csv = "casia.csv"
    # output_root = "/home/jarred/Documents/Datasets/GAN"
    # copy_for_gan(dataset_root, dataset_csv, output_root)
