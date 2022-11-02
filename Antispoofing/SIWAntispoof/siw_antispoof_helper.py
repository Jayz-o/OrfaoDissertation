import os

import pandas as pd

from Antispoofing.AntispoofHelpers.dataset_helper import make_medium_name_frame, get_dataframe_by_medium_name, \
    get_dataframe_by_attack_category
from DatasetProcessing.DatasetCreators.SIWCreator.siw_helper import convert_attack_category_to_medium_list

ATTACK_TYPE_COMBINATIONS = [
    'ASUS-IP7P-IPP2017', 'ASUS-IP7P-SGS8', 'ASUS-IPP2017-SGS8', 'IP7P-IPP2017-SGS8', 'P', 'R'
]



def get_siw_train_frame_func(dataset_root, dataset_csv_name, combinations, train_subject_number):
    frame = pd.read_csv(os.path.join(dataset_root, dataset_csv_name))
    if train_subject_number is not None:
        frame = frame.query(f"subject_number == {train_subject_number}")
    else:
        frame = frame.query("usage_type == 'train'")
    attack_type_combinations = combinations[0].split("-")
    attack_frames = []
    for attack_type in attack_type_combinations:
        if attack_type == "P" or attack_type == "R":
            attack_frames.append(get_dataframe_by_attack_category(frame, attack_type))
        else:
            attack_frames.append(get_dataframe_by_medium_name(frame, attack_type))

    real_frame = get_dataframe_by_attack_category(frame, "N")
    attack_frame = pd.concat(attack_frames)
    return pd.concat([real_frame, attack_frame])

def get_siw_stratified_name_list_func(attack_category):
    return convert_attack_category_to_medium_list(attack_category)
def get_siw_stratified_name_col(combinations):

    return "medium_name"

def process_siw_dataset_metrics_func(df, save_root):
    selected_df = df[
        ["experiment_tag", "protocol", "protocol_number", "fold_number","config/HP_AUG_PER", "config/HP_REPEAT", "APCER", "BPCER", "ACER", "AUC", "EER", "TN",
         "TP", "FN", "FP"]].sort_values(by=["protocol_number", "ACER"])
    selected_df.to_csv(os.path.join(save_root, "selected_metrics.csv"))
    descriptive_metrics = selected_df.groupby(["config/HP_AUG_PER","protocol_number"]).describe(percentiles=[])
    descriptive_metrics.to_csv(os.path.join(save_root, "descriptive_metrics.csv"))
    quick_metrics = descriptive_metrics[
        [("APCER", "mean"), ("APCER", "std"), ("BPCER", "mean"), ("BPCER", "std"), ("ACER", "mean"), ("ACER", "std"),
         ("EER", "mean"), ("EER", "std"), ("AUC", "mean"), ("AUC", "std")]]
    quick_metrics.to_csv(os.path.join(save_root, "quick_metrics.csv"))
    individual_protocols = df.groupby(['protocol_number','config/HP_AUG_PER', 'protocol', 'config/HP_COMB' ]).mean()
    individual_protocols.to_csv(os.path.join(save_root, "individual_protocols.csv"), index_label=True)
    averaged_protocols = df.groupby(['protocol_number', 'config/HP_AUG_PER', ]).mean()
    averaged_protocols.to_csv(os.path.join(save_root, "averaged_protocols.csv"), index_label=True)
    averaged_metrics = df.groupby(['protocol_number', 'config/HP_AUG_PER', ]).describe()
    averaged_metrics.to_csv(os.path.join(save_root, "averaged_metrics.csv"), index_label=True)

def get_siw_test_frame(dataset_name, dataset_csv_name, test_subject_number):
    frame = pd.read_csv(os.path.join(dataset_name, dataset_csv_name))
    if test_subject_number is None:
        return frame.query("usage_type == 'test'")
    else:
        return frame.query(f'subject_number == {test_subject_number}')

def get_test_medium_frame(test_frame, medium_name, with_normal=True):
    medium_frame = get_dataframe_by_medium_name(test_frame, medium_name)
    if with_normal:
        real_frame = get_dataframe_by_attack_category(test_frame, 'N')
        return pd.concat([real_frame, medium_frame])
    else:
        return medium_frame

def get_test_attack_frame(test_frame, attack_category, with_normal=True):
    attack_frame = get_dataframe_by_attack_category(test_frame, attack_category)
    if with_normal:
        real_frame = get_dataframe_by_attack_category(test_frame, 'N')
        return pd.concat([real_frame, attack_frame])
    return attack_frame

def get_siw_protocol_frame_dic(dataset_name, dataset_csv_name, combinations, test_subject_number,with_normal=True):
    test_frame = get_siw_test_frame(dataset_name, dataset_csv_name, test_subject_number)
    # remove any normal
    tempcategories = []
    for cat in combinations:
        if "N" not in cat:
            tempcategories.append(cat)

    combinations = tempcategories

    protocol_frame_dic = {
            'ASUS-IP7P-IPP2017': {'LOO_SGS8': {"protocol_number":2,
                                               "frame":get_test_medium_frame(test_frame, 'SGS8',with_normal)}},
            'ASUS-IP7P-SGS8': {'LOO_IPP2017': {"protocol_number": 2,
                                               "frame": get_test_medium_frame(test_frame, 'IPP2017',with_normal)}},
            'ASUS-IPP2017-SGS8': {'LOO_IP7P': {"protocol_number": 2,
                                               "frame": get_test_medium_frame(test_frame, 'IP7P',with_normal)}},
            'IP7P-IPP2017-SGS8': {'LOO_ASUS': {"protocol_number": 2,
                                               "frame": get_test_medium_frame(test_frame, 'ASUS',with_normal)}},
            'R': {'UnKnown_P': {"protocol_number": 3, "frame": get_test_attack_frame(test_frame, 'P', with_normal)}},
            'P': {'UnKnown_R': {"protocol_number": 3, "frame": get_test_attack_frame(test_frame, 'R', with_normal)}},
           }

    return protocol_frame_dic[combinations[0]]