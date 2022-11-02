import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from Helpers.image_helper import create_image_grid_from_paths, obtain_file_paths
plt.style.use("default")

def view_frames_for_folder(folder_root, file_name, title, cols):
    files = obtain_file_paths(folder_root, r"^frame")
    create_image_grid_from_paths(files,file_name=file_name, title=title, num_cols=cols, dpi=1000)

def create_k_vs_sil_grid(dataset_root, dataset_csv, subject_number, save_path_root):
    frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    filtered = frame.query(f"subject_number == {subject_number}")
    names = (filtered['attack_category'] + '|' + filtered['medium_name'] + '|' + filtered['session_name'] + '|' +
             filtered['sensor_id'].astype(str)).tolist()
    directories = filtered['directory_path'].tolist()
    full_paths = [os.path.join(dataset_root, dir_path, "k_vs_sil.png") for dir_path in directories]
    create_image_grid_from_paths(full_paths,
                                 file_name=os.path.join(save_path_root, f"subject_{subject_number}_trends.pdf"),
                                 title=f"Subject {subject_number}", dpi=1000, scale=2, desired_image_shape=1000,
                                 class_names=names)


def create_scatter_plot(x, y, title, mean, min_ratio, max_ratio, out_root, dataset_name):
    # https://towardsdatascience.com/legend-outside-the-plot-matplotlib-5d9c1caa9d31
    with plt.ioff():
        fig, ax = plt.subplots()
        plt.scatter(x, y)
        plt.title(None)
        plt.ylabel("Keframes")
        plt.xlabel("Video Frames")

        # Add a legend
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.35),
            title=f"Keyframe Reduction (%):\nMean = {round(mean * 100, 4)}\nMin = {round(min_ratio * 100, 4)}\nMax = {round(max_ratio * 100, 4)}"
        )

        folder_name = os.path.join(out_root, f"{dataset_name}_IndividualTrends")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, f"{title}.png")
        fig.savefig(file_path, dpi=1000)

    return file_path

    # fig.savefig(os.path.join(k_vs_silscore_save_path, "k_vs_sil.png"), dpi=300)


def get_siw_dic_key(row_dict):
    return f"{row_dict['attack_category']}|Sen:{row_dict['sensor_id']}|M:{row_dict['medium_id']}|Ses:{row_dict['session_id']}"


def get_casia_dic_key(row_dict):
    # return f"{row_dict['ground_truth']}"
    return f"{row_dict['attack_category']}"


def determine_trend_for_files(dataset_root, dataset_csv, save_root, get_dataset_dic_key_func, must_redo=False,
                              file_name="trends.pdf", num_cols=None, sorted_key_order=None, scale=2):
    dataset_name = os.path.basename(dataset_root)
    pickle_path = os.path.join(save_root, f"{dataset_name}_trend.pkl")
    if must_redo is False and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as file:
            info_dic = pickle.load(file)
    else:
        frame = pd.read_csv(os.path.join(dataset_root, dataset_csv))
        info_dic = dict()
        for row_dict in frame.to_dict(orient="records"):
            folder_key = get_dataset_dic_key_func(row_dict)
            k_score = row_dict['frames_present']
            video_length = row_dict['video_frames']
            k_length_ratio = k_score / video_length

            if folder_key in info_dic:
                info_dic[folder_key]['total_ratio'] += k_length_ratio
                info_dic[folder_key]['ratios'].append(k_length_ratio)
                info_dic[folder_key]['k_scores'].append(k_score)
                info_dic[folder_key]['video_lengths'].append(video_length)
            else:
                info_dic[folder_key] = {'total_ratio': k_length_ratio, 'ratios': [k_length_ratio],
                                        'k_scores': [k_score],
                                        'video_lengths': [video_length]}
        copy = info_dic.copy()
        for key, value in copy.items():
            average = value['total_ratio'] / len(value['ratios'])
            min_frames = min(value['video_lengths'])
            max_frames = max(value['video_lengths'])
            min_k_score = min(value['k_scores'])
            max_k_score = max(value['k_scores'])
            min_ratio = min(value['ratios'])
            max_ratio = max(value['ratios'])
            info_dic[key]['average'] = average
            info_dic[key]['min_frames'] = min_frames
            info_dic[key]['max_frames'] = max_frames
            info_dic[key]['min_k_score'] = min_k_score
            info_dic[key]['max_k_score'] = max_k_score
            info_dic[key]['min_ratio'] = min_ratio
            info_dic[key]['max_ratio'] = max_ratio
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        with open(pickle_path, 'wb') as file:
            pickle.dump(info_dic, file)
    grid_Images = []
    labels = []
    if sorted_key_order is None:
        sorted_key_order = sorted(info_dic)
    for key in sorted_key_order:
        value = info_dic[key]
        labels.append(key)
        grid_Images.append(
            create_scatter_plot(value['video_lengths'], value['k_scores'], key, value['average'], value['min_ratio'],
                                value['max_ratio'], save_root, dataset_name))
    extension = os.path.splitext(file_name)[1]
    if len(extension) <= 0:
        file_name += ".pdf"
    save_path = os.path.join(save_root, file_name)
    create_image_grid_from_paths(grid_Images, file_name=save_path, scale=scale, dpi=1000, title=None,#f"{dataset_name} Reductions",
                                 desired_image_shape=1000, class_names=labels, num_cols=num_cols)


def determine_trend_for_siw_files(dataset_root, dataset_csv, save_root, must_redo=False, file_name="siw_trends.pdf", num_cols=None):
    determine_trend_for_files(dataset_root, dataset_csv, save_root, get_siw_dic_key, must_redo, file_name, num_cols, scale=3)


def determine_trend_for_casia_files(dataset_root, dataset_csv, save_root, must_redo=False,
                                    file_name="casia_trends.pdf", num_cols=None):
    sorted_keys = [
        'N1', 'N2', 'HR_N',
        'C1', 'C2', 'HR_C',
        'R1', 'R2', 'HR_R',
        'W1', 'W2', 'HR_W',
        # "real", "spoof"

                   ]
    determine_trend_for_files(dataset_root, dataset_csv, save_root, get_casia_dic_key, must_redo, file_name, num_cols,
                              sorted_key_order=sorted_keys)

def determine_casia_trend():
    # dataset_root = "/home/jarred/Documents/Datasets/CASIA_KF_OLD"
    dataset_root = "/home/jarred/Documents/Datasets/CASIA_KF"
    dataset_csv = "casia.csv"
    save_root = os.path.join(dataset_root, "Trends")
    determine_trend_for_casia_files(dataset_root, dataset_csv, save_root)

if __name__ == "__main__":
    determine_casia_trend()
    dataset_root = "/home/jarred/Documents/Datasets/SIW_KF"
    dataset_csv = "siw.csv"
    save_root = os.path.join("/home/jarred/Documents/Datasets/SIW_KF_Trends")
    # subject_number = 100
    # determine_trend_for_siw_files(dataset_root, dataset_csv)
    # # image = create_scatter_plot([1,5,10], [10, 5, 7], f"file", 0.21, 0, 1)
    # # create_image_grid([np.array(image)], file_name="out.png", dpi=1000)
    determine_trend_for_siw_files(dataset_root, dataset_csv, save_root, must_redo=True, num_cols=5)

    # view_frames_for_folder("/home/jarred/Documents/Datasets/TEMP/train/spoof/1/6", "/home/jarred/Documents/Datasets/TEMP/train/spoof/1/6/1_grid_k2.png", "Subject 1 C2 Min K=2")
