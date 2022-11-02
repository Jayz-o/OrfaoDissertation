import pandas as pd
import os
import shutil

from Antispoofing.AntispoofHelpers.spoof_metric import determine_spoof_metrics
from DatasetProcessing.DatasetCreators.FaceHelpers.face_detection_helper import detect_face_in_video
from Helpers.image_helper import obtain_file_paths
from tqdm.auto import tqdm
def combined_metrics(df):
    return df.groupby(['protocol_number', 'when_aug_performed' , 'aug_type','config/HP_AUG_PER']).mean()

def combined_metrics_describe(df):

    return df.groupby(['protocol_number', 'when_aug_performed' , 'aug_type','config/HP_AUG_PER']).describe()

def individual_metrics(df):

    return df.groupby(['protocol_number', 'protocol', 'when_aug_performed' , 'aug_type','config/HP_AUG_PER']).mean()

def load_csv(full_csv_path, aug_type, when_aug_performed):
    df = pd.read_csv(full_csv_path)
    df['aug_type'] = aug_type
    df['when_aug_performed'] = when_aug_performed
    if 'Multi_TP' in df:
        df = df[['when_aug_performed', 'aug_type', 'protocol_number', 'protocol', 'config/HP_AUG_PER',  'TP', 'TN', 'FP', 'FN','APCER','BPCER', 'ACER', "EER", "AUC", 'Multi_TP', 'Multi_TN', 'Multi_FP', 'Multi_FN','Multi_APCER','Multi_BPCER', 'Multi_ACER', "Multi_EER", "Multi_AUC", 'config/HP_REPEAT', 'fold_number', 'config/HP_COMB']].sort_values(["protocol_number", "config/HP_AUG_PER"])
    else:
        df = df[['when_aug_performed', 'aug_type', 'protocol_number', 'protocol', 'config/HP_AUG_PER',  'TP', 'TN', 'FP', 'FN','APCER','BPCER', 'ACER', "EER", "AUC", 'config/HP_REPEAT', 'fold_number', 'config/HP_COMB']].sort_values(["protocol_number", "config/HP_AUG_PER"])
    return df

def video_based_results(csv_file_path, protocol_name, fold_index,protocol_number, fold_save_metrics_root, save_metric_name, window_size=None, start_frame=0, is_casia=True):
    df_predictions = pd.read_csv(csv_file_path)
    # isolate video_name
    def categorise_video(row):
        res =  os.path.basename(os.path.dirname(row['file_paths']))
        if is_casia:
            if "-" in res:
                return res.split("-")[1]
        return res

    def obtain_subjects_for_video(row):
        return os.path.basename(os.path.dirname(os.path.dirname(row['file_paths'])))
    # isolate frame number
    def get_only_frames(row):
        if type(row['file_paths']) is int:
            return
        base = os.path.basename(row['file_paths'])
        split = base.split("_")
        temp1 = split[1]
        split2 = temp1.split(".png")
        return int(split2[0])
    df_predictions['video_name'] = df_predictions.apply(categorise_video, axis=1)
    df_predictions['subject_number'] = df_predictions.apply(obtain_subjects_for_video, axis=1)
    video_names = df_predictions['video_name'].unique()
    subjects = df_predictions['subject_number'].unique()
    video_names.sort()
    subjects.sort()
    video_list = []
    window_name = f"all"
    # try:
    for subject_number in subjects:
        for name in video_names:
            single_vid_df = df_predictions.query(f"video_name == '{name}' and subject_number == '{subject_number}'")
            single_copy = single_vid_df.copy()
            single_copy['frame_number'] = single_vid_df.apply(get_only_frames, axis=1)
            # single_vid_df_copy= single_vid_df.apply(lambda row: get_only_frames(row), axis=0)
            # single_vid_df_copy= single_vid_df.apply(lambda row: get_only_frames(row), axis=0)
            single_copy = single_copy.sort_values(by='frame_number')
            if window_size is not None:
                single_copy = single_copy[start_frame:start_frame+window_size]
                window_name = f"{window_size}"
            real = 0
            spoof = 1
            spoof_pred_count = single_copy[(single_copy.predicted == 1)].count()["predicted"]
            real_pred_count = single_copy[(single_copy.predicted == 0)].count()["predicted"]
            if spoof_pred_count >= real_pred_count:
                predicted = 1
            else:
                predicted = 0
            ground_truth = single_vid_df['ground_truth'].tolist()[0]
            video_list.append({'subject_number': subject_number, "video_name": name, f'spoof({spoof})_pred_count': spoof_pred_count, f'real({real}_pred_count': real_pred_count, "predicted": predicted, "ground_truth": ground_truth, "window":window_size})
            pred_path_namee = os.path.join(fold_save_metrics_root, f"Window_{window_name}", f"{protocol_name}-{fold_index}")
            if not os.path.exists(pred_path_namee):
                os.makedirs(pred_path_namee)

            single_copy.to_csv(os.path.join(pred_path_namee, f"S{subject_number}-V{name}-Predictions.csv"), index=False)

    multi_frame = pd.DataFrame.from_dict(video_list)
    # save_metric_name = None
    if fold_save_metrics_root is not None:
        multi_frame.to_csv(os.path.join(fold_save_metrics_root, f"test_{protocol_name}_multi_frame_results_w_{window_name}.csv"), index=False)
        save_metric_name = os.path.join(fold_save_metrics_root,f"{save_metric_name}_{protocol_name}_Multi_Metrics")
    predicted = multi_frame['predicted'].tolist()
    ground_truth = multi_frame['ground_truth'].tolist()
    metric_dic = determine_spoof_metrics(ground_truth, predicted, protocol_name, fold_index,protocol_number, save_dir= None, must_show=False)
    # metric_dic = determine_spoof_metrics(ground_truth, predicted, protocol_name, fold_index,protocol_number, save_dir= save_metric_name, must_show=False)
    metric_dic = dict(("{}_{}".format("Multi",k),v) for k,v in metric_dic.items())
    # except:
    #     print(csv_file_path)
    return metric_dic
# /home/jarred/Documents/SavedAntispoofing/SIW_90_GenAugNormalAndSpoof/AfterSplit/SpoofMetrics/Single/SIW_antispoof_09_30_2022_13_58_46/A@ASUS-IP7P-IPP2017,A@N_aug_0.1_run_1/0/
# SIW_90_GenAugNormalAndSpoof/AfterSplit/SpoofMetrics/Single/SIW_antispoof_09_30_2022_13_58_46/A@ASUS-IP7P-IPP2017,A@N_aug_0.1_run_1/0/test_LOO_SGS8_results.csv
def copy_multi_csv_files(antispoof_root, save_folder_path=None, window_size = None, save_metric_name="Test", must_save_metrics=False, start_frame=0, is_casia=True):
    # create the save to folder
    save_folder_name = "MasterMetrics"
    if save_folder_path is None:
        save_folder_path = os.path.join(antispoof_root, save_folder_name)
    else:
        save_folder_path = os.path.join(save_folder_path, save_folder_name)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # get the csv files
    csv_files = obtain_file_paths(antispoof_root, r"^test_(.*?)_results\.csv$")

    # loop through the files
    multi_metric_list = []
    for csv_path in tqdm(csv_files):
        if "multi" in csv_path:
            continue
        # load up the csv
        temp_root = antispoof_root
        if temp_root[-1] != "/":
            temp_root += "/"
        temp_path = csv_path.split(temp_root)[1]
        path_bits = temp_path.split(os.path.sep)
        folder_name = path_bits[0]
        when_aug = path_bits[1]
        # SIW_90_GenAugNormalAndSpoof/AfterSplit/SpoofMetrics/Single/SIW_antispoof_09_30_2022_13_58_46/A@ASUS-IP7P-IPP2017,A@N_aug_0.1_run_1/0/test_LOO_SGS8_results.csv
        fold_number = path_bits[-2]
        protocol = path_bits[-1].split(".csv")[0]
        protocol = protocol.split("test_")[1]
        protocol = protocol.split("_results")[0]
        combination = path_bits[-3].split("_aug_")[1]
        run_aug_bits = combination.split("_run_")

        aug = run_aug_bits[0]

        run = run_aug_bits[1]
        if "SIW" in folder_name:
            if "LOO" in protocol:
                protocol_number = 2
            else:
                protocol_number = 3
        else:
            if "Low_Quality" in protocol:
                protocol_number = 1
            elif "Normal_Quality" in protocol:
                protocol_number = 2
            elif "High_Quality" in protocol:
                protocol_number = 3
            elif "Warped" in protocol:
                protocol_number = 4
            elif "Cut" in protocol:
                protocol_number = 5
            elif "Video" in protocol:
                protocol_number = 6
            elif "All" in protocol:
                protocol_number = 7
            else:
                raise TypeError("Unaccounted for protocol")

        folder_bits =folder_name.split("_")
        aug_type = folder_bits[-1]
        if "Baseline" in aug_type:
            when_aug = "None"
        if "KF" in folder_name:
            aug_type += "_KF"
        # create save folder
        save_path = os.path.join(save_folder_path, folder_name, when_aug)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        predictions_folder = os.path.join(save_path, "predictions", f"prot_num_{protocol_number}-prot_{protocol}-rep_{run}-aug_{aug}-fold_{fold_number}" )

        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        # if os.path.exists(os.path.join(predictions_folder, f"predictions.csv")):
        #     continue
        save_metrics_path = None
        if must_save_metrics:
            save_metrics_path = predictions_folder
        multi_metric = video_based_results(csv_path, protocol, fold_number, protocol_number, save_metrics_path,
                            save_metric_name, window_size=window_size,start_frame=start_frame,is_casia=is_casia)
        multi_metric["when_aug_performed"] = when_aug
        multi_metric["aug_type"] = aug_type
        multi_metric['window_size']= window_size
        multi_metric['config/HP_AUG_PER']= aug
        multi_metric['config/HP_REPEAT']=run
        multi_metric['protocol_number']=protocol_number
        multi_metric['protocol']=protocol
        multi_metric['fold_number']=fold_number

        multi_metric_list.append(multi_metric)

        # copy the file over
        source_csv_name = csv_path
        destination = os.path.join(predictions_folder, f"predictions.csv")
        shutil.copy(source_csv_name, destination)

    multi_df = pd.DataFrame.from_dict(multi_metric_list)
    multi_df.to_csv(os.path.join(save_folder_path, f"multi_window_{window_size}.csv"), index=True)
    # multi_df = pd.read_csv(os.path.join(save_folder_path, "multi_window_30.csv"))
    selected_multi_df = multi_df[['when_aug_performed', 'aug_type', 'protocol_number', 'protocol', 'config/HP_AUG_PER',  'Multi_TP', 'Multi_TN', 'Multi_FP', 'Multi_FN','Multi_APCER','Multi_BPCER', 'Multi_ACER', "Multi_EER", "Multi_AUC", 'config/HP_REPEAT', 'fold_number']].sort_values(["protocol_number", "config/HP_AUG_PER"])
    df_combined = combined_metrics(selected_multi_df)
    df_combined.to_csv(os.path.join(save_folder_path, f"multi_window_{window_size}_combined.csv"), index=True)
    df_combined_describe = combined_metrics_describe(selected_multi_df)
    df_combined_describe.to_csv(os.path.join(save_folder_path,f"multi_window_{window_size}_combined_describe.csv"), index=True)
    df_individual = individual_metrics(selected_multi_df)
    df_individual.to_csv(os.path.join(save_folder_path, f"multi_window_{window_size}_individual.csv"), index=True)

# /home/jarred/Documents/SavedAntispoofing/SIW_90_GenAugNormalAndSpoof/AfterSplit/SpoofMetrics/Single/SIW_antispoof_09_30_2022_13_58_46/A@ASUS-IP7P-IPP2017,A@N_aug_0.1_run_1/0/
# SIW_90_GenAugNormalAndSpoof/AfterSplit/SpoofMetrics/Single/SIW_antispoof_09_30_2022_13_58_46/A@ASUS-IP7P-IPP2017,A@N_aug_0.1_run_1/0/
def copy_csv_files(antispoof_root, save_folder_path=None):
    # create the save to folder
    save_folder_name = "MasterMetrics"
    if save_folder_path is None:
        save_folder_path = os.path.join(antispoof_root, save_folder_name)
    else:
        save_folder_path = os.path.join(save_folder_path, save_folder_name)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # get the csv files
    csv_files = obtain_file_paths(antispoof_root, r"_antispoof_tune.csv$")

    # loop through the files
    combined_frames =[]
    describe_combined_frames =[]
    individual_frames =[]
    for csv in csv_files:
        # load up the csv
        temp_root = antispoof_root
        if temp_root[-1] != "/":
            temp_root += "/"
        temp_path = csv.split(temp_root)[1]
        path_bits = temp_path.split(os.path.sep)
        folder_name = path_bits[0]
        when_aug = path_bits[1]
        folder_bits =folder_name.split("_")
        aug_type = folder_bits[-1]
        if "Baseline" in aug_type:
            when_aug = "None"
        # create save folder
        save_path = os.path.join(save_folder_path, folder_name, when_aug)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df = load_csv(csv,aug_type, when_aug)
        df_combined = combined_metrics(df)
        df_combined.to_csv(os.path.join(save_path, "combined.csv"), index=True)
        df_combined_describe = combined_metrics_describe(df)
        df_combined_describe.to_csv(os.path.join(save_path, "combined_describe.csv"), index=True)
        df_individual = individual_metrics(df)
        df_individual.to_csv(os.path.join(save_path, "individual.csv"), index=True)

        combined_frames.append(df_combined)
        describe_combined_frames.append(df_combined_describe)
        individual_frames.append(df_individual)

        source_csv_name = csv
        destination = os.path.join(save_path, f"{folder_name}-{when_aug}.csv")
        shutil.copy(source_csv_name, destination)

    combined_frames_df = pd.concat(combined_frames).groupby(['protocol_number', 'when_aug_performed', 'aug_type','config/HP_AUG_PER']).mean()
    combined_frames_df.to_csv(os.path.join(save_folder_path, "combined.csv"), index=True)
    describe_combined_frames_df = pd.concat(describe_combined_frames).groupby(['protocol_number', 'when_aug_performed', 'aug_type','config/HP_AUG_PER' ]).describe()
    describe_combined_frames_df.to_csv(os.path.join(save_folder_path, "combined_describe.csv"), index=True)
    individual_frames_df = pd.concat(individual_frames).groupby(['protocol_number','protocol', 'config/HP_AUG_PER', 'aug_type', 'when_aug_performed']).mean()
    individual_frames_df.to_csv(os.path.join(save_folder_path, "individual.csv"), index=True)




if __name__ == '__main__':
    # windows = [None]
    windows = [3, 5, 7, 9,10, 11, 15, None]
    # save_path = "/media/jarred/ssd/SIW_Metrics"
    save_path = "/home/jarred/Desktop/Metrics/CASIA"
    save_name = "SIW_Test_"
    is_casia = False
    antispoof_root = "/home/jarred/Desktop/SavedAntispoofing/CASIA"

    # antispoof_root = "/home/jarred/Documents/SavedAntispoofing/CASIA"
    # save_path = "/media/jarred/ssd/CASIA_Metrics"
    # is_casia = True
    # save_name = "CASIA_Test_"
    # save_path = "/home/jarred/Dropbox/CASIA_test"
    # save_path = "/home/jarred/Dropbox/SIW"
    # for window in windows:
    #     copy_multi_csv_files(antispoof_root, save_path, window_size=window, save_metric_name=save_name, start_frame=0, must_save_metrics=True, is_casia=is_casia)
    # antispoof_root = "/home/jarred/Desktop/NewSIWBL/"
    # save_path = "/home/jarred/Dropbox/SIW"
    copy_csv_files(antispoof_root, save_path)
    # # antispoof_root = "/home/jarred/Documents/SavedAntispoofing"
    # save_path = "/home/jarred/Dropbox"
    # copy_csv_files(antispoof_root, save_path)