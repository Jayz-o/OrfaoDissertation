
# protocol 2, attack category R, sensor id unique, leave one out. Gan for each 3 combinations
import os

import pandas as pd

from Antispoofing.AntispoofHelpers.dataset_helper import get_dataframe_by_usage_type
from GAN.GANHelpers.gan_dataset_helper import copy_attack_categories_to_folder, copy_medium_names_to_folder


def get_protocol_2_folders(dataset_root, dataset_csv, output_root, subject_number=None, verbose=False):
    frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is not None:
        frame = frame.query(f'subject_number == {subject_number} and attack_category == "R"')
    else:
        frame = frame.query(f'usage_type == "train" and attack_category == "R"')
    unique_medium_names = frame['medium_name'].unique().tolist()
    unique_medium_names.sort()
    medium_name_combinations = []
    for index, id in enumerate(unique_medium_names):
        # temp = unique_medium_names.copy()
        # temp.pop(index)
        # medium_name_combinations.append(temp)
        medium_name_combinations.append([id])
    for medium_combination in medium_name_combinations:
        copy_medium_names_to_folder(frame, medium_combination, dataset_root, output_root, subject_number, verbose)

#protocol 3. attack category R , P train on 1 test on the other. Gan for R and P
def get_protocol_3_folders(dataset_root, dataset_csv, output_root, subject_number=None, verbose=False):
    frame_main = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is not None:
        frame_r = frame_main.query(f'subject_number == {subject_number} and attack_category == "R"')
        frame_p = frame_main.query(f'subject_number == {subject_number} and attack_category == "P"')
    else:
        frame_r = frame_main.query(f'usage_type == "train" and attack_category == "R"')
        frame_p = frame_main.query(f'usage_type == "train" and attack_category == "P"')


    copy_attack_categories_to_folder(frame_r, ["R"], dataset_root, output_root, subject_number, verbose)
    copy_attack_categories_to_folder(frame_p, ["P"], dataset_root, output_root, subject_number, verbose)

def get_normal_folders(dataset_root, dataset_csv, output_root, subject_number=None, verbose=False):
    frame_main = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    if subject_number is not None:
        frame_n = frame_main.query(f'subject_number == {subject_number} and attack_category == "N"')
    else:
        frame_n = frame_main.query(f'usage_type == "train" and attack_category == "N"')


    copy_attack_categories_to_folder(frame_n, ["N"], dataset_root, output_root, subject_number, verbose)


