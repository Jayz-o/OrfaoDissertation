import os.path
import pickle
import shutil

import numpy as np
from tqdm import tqdm
import pandas as pd

from DatasetProcessing.KeyFrameExtraction.KFHelpers.keyframe_helper import K_VS_SIL_PKL, PATH_FEATURE_DIC_NAME
from Helpers.image_helper import obtain_file_paths



def _print(m):
    print(m)

def _outer(func, a):
    func(a)

def copy_keyframes_across(dataset_root, save_root, dataset_csv):
    dataset_frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    paths = dataset_frame['directory_path'].tolist()
    for path in paths:
        save_path = os.path.join(save_root, path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle_path = os.path.join(dataset_root, path, K_VS_SIL_PKL)
        feature_path = os.path.join(dataset_root, path, PATH_FEATURE_DIC_NAME)
        new_pickle = os.path.join(save_path, K_VS_SIL_PKL)
        new_feature = os.path.join(save_path, PATH_FEATURE_DIC_NAME)
        shutil.copy(pickle_path, new_pickle)
        shutil.copy(feature_path, new_feature)

def fix_path_feature_name(old_dataset_root,dataset_root, dataset_csv, replace_name_old, replace_name_new):
    dataset_frame = pd.read_csv(os.path.join(old_dataset_root, dataset_csv))
    paths = dataset_frame['directory_path'].tolist()
    for path in paths:
        feature_path = os.path.join(dataset_root, path, PATH_FEATURE_DIC_NAME)
        with open(feature_path, 'rb') as f:
            dic = pickle.load(f)
        paths = list(dic.keys())
        features = list(dic.values())
        fixed_dic = {}
        for i in range(len(paths)):
            if replace_name_old in paths[i]:
                fixed_dic[paths[i].replace(replace_name_old, replace_name_new)] = features[i]
            else:
                fixed_dic[paths[i]] = features[i]
        with open(feature_path, 'wb') as f:
            pickle.dump(fixed_dic, f)

def fix_category_name(old_dataset_root, dataset_csv):
    dataset_frame = pd.read_csv(os.path.join(old_dataset_root, dataset_csv))
    paths = dataset_frame['directory_path'].tolist()
    for path in paths:
        csv_path = os.path.join(old_dataset_root, path, "info.csv")
        info_csv = pd.read_csv(csv_path)
        if info_csv["attack_category"][0] == "HR_w":
            info_csv["attack_category"] = "HR_W"
            info_csv.to_csv(csv_path, index=False)


if __name__ == "__main__":
    dataset_root = "/home/jarred/Documents/Datasets/CASIA_KF"
    save_root = "/home/jarred/Documents/Datasets/TEMP"
    dataset_csv = "casia.csv"
    copy_keyframes_across(dataset_root,save_root,dataset_csv)
    # dataset_root = "/home/jarred/Documents/Datasets/TEMP"
    # old_dataset_root = "/home/jarred/Documents/Datasets/CASIA"
    # save_root = "/home/jarred/Documents/Datasets/TEMP"
    # dataset_csv = "casia.csv"
    # fix_path_feature_name(old_dataset_root,dataset_root, dataset_csv, "CAISA_Processed", "CASIA")
    # fix_category_name("/home/jarred/Documents/Datasets/CASIA_KF","casia.csv")




