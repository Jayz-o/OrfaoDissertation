import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"


from DatasetProcessing.DatasetCreators.SIWCreator.ray_siw_dataset_creator import DATASET_CSV_NAME

import os.path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from datetime import datetime

from ray import tune
from sklearn.metrics import silhouette_score
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from DatasetProcessing.helper import combine_csvs, move_error_files
from Helpers.image_helper import load_image_from_file
from Helpers.image_helper import obtain_file_paths
from DatasetProcessing.KeyFrameExtraction.KFHelpers.processify import processify

K_VS_SIL_PKL = "k_vs_sil.pkl"

os.environ['TUNE_RESULT_DELIM'] = '/'
# set the plot style
# plt.style.use('dark_background')
# plt.style.use('default')

# name of the path_feature_dictionary
INFO_CSV_NAME = 'info.csv'
PATH_FEATURE_DIC_NAME = "path_feature_dic.pkl"

def determine_variational_k(info_frame, mean_k):
    if info_frame['optimal_k'][0] < mean_k:
        return mean_k
    else:
        return int(round(info_frame['optimal_k'][0]))

def get_category_mean_dic(save_root, dataset_name, mean_col, use_only_training_information=True):
    pickle_name = K_VS_SIL_PKL
    dataset_csv_path = os.path.join(save_root, f"{dataset_name}.csv")
    dataset_frame = pd.read_csv(dataset_csv_path)
    # only use training set information
    if use_only_training_information:
        dataset_frame = dataset_frame.query("usage_type == train")
    metrics = dataset_frame.groupby([mean_col])
    metric_frame = metrics.describe()

    metric_frame = metric_frame.iloc[:, 24:]
    metric_frame.to_csv(os.path.join(save_root, f"{dataset_name}_kf_stats.csv"))

    return metric_frame.to_dict()[('optimal_k', 'mean')]


def find_max_k(dataset_root, directory_path, pickle_name):
    folder_path = os.path.join(dataset_root, directory_path)
    with open(os.path.join(folder_path, pickle_name), "rb") as f:
        k_vs_sil = pickle.load(f)
    max_score = max(k_vs_sil[1])
    max_index = k_vs_sil[1].index(max_score)
    return k_vs_sil[0][max_index]

def view_keyframe_reduction(kf_csv, save_root=None, must_show=False, is_pdf=False, dpi=300):
    """
    visualise the frame reduction through keyframe extraction.
    :param kf_csv: The csv file belonging to the keyframe dataset.
    :param save_root: The root directory to save the plot to.
    :param must_show: True to show display the plot
    :param is_pdf: True to save the plot as a pdf
    :return: None
    """
    # plt.style.use('default')
    # plt.style.use('dark_background')
    df = pd.read_csv(kf_csv)
    df.sort_values(['video_frames'], inplace=True)
    original_frames = df['video_frames'].tolist()
    key_frames = df['frames_present'].tolist()
    video_numbers = [i for i in range(1, len(original_frames) + 1)]
    with plt.ioff():
        f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        # for setting different fonts
        # font = {'family': 'normal', 'weight': 'bold', 'size': 15}
        # plt.rc('font', **font)
        bar_width = 0.8
        ax1.bar(video_numbers, original_frames, label='Original Frames', alpha=0.5, color='b', width=bar_width)
        ax1.bar(video_numbers, key_frames, label='Keyframes', alpha=0.5, color='r', width=bar_width)
        plt.sca(ax1)
        ax1.set_ylabel("Number of Frames")
        ax1.set_xlabel("Ordered Videos (By Frame Count)")
        plt.tight_layout()
        plt.legend(loc='upper left')

        if not save_root is None:
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            if is_pdf:
                plot_name = "keyframe_effect.pdf"
            else:
                plot_name = "keyframe_effect.png"
            plt.savefig(os.path.join(save_root, plot_name), dpi=dpi)

        if not must_show:
            plt.ioff()
        else:
            plt.ion()
            plt.show()


def initialise_tf():
    import tensorflow as tf
    try:
        # fix memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


def load_path_feature_dic(save_path):
    try:
        path_feature_dic_file = os.path.join(save_path, PATH_FEATURE_DIC_NAME)
        if os.path.exists(path_feature_dic_file):
            with open(path_feature_dic_file, 'rb') as file:
                path_feature_dic = pickle.load(file)
            return path_feature_dic
        else:
            return None
    except Exception as e:
        print(f"ERROR reading pickle file. This could be due to the process extracting the features being "
              f"unexpectedly stopped while saving the features.. Removing pickle for extraction to occur "
              f"again:\n{e}")
        if os.path.exists(path_feature_dic_file):
            os.remove(path_feature_dic)


def extract_features_from_paths(image_files, save_root=None, frame_dir=None, force_redo=False,
                                must_use_separate_process=False):
    if must_use_separate_process:
        return _extract_features_from_paths_process(image_files, save_root, frame_dir, force_redo)
    else:
        return _extract_features_from_paths(image_files, save_root, frame_dir, force_redo)


@processify
def _extract_features_from_paths_process(image_files, save_root=None, frame_dir=None, force_redo=False):
    return _extract_features_from_paths(image_files, save_root, frame_dir, force_redo)


def _extract_features_from_paths(image_files, save_root=None, frame_dir=None, force_redo=False):
    """
    Extract VGG Face features from an image using the image file path
    :param image_files: a list of image file paths
    :param save_path_root: The save root directory to save the file_path_feature dictionary
    :param frame_dir: The directory applied to the save root in which to save the file_path_feature dictionary
    :return: The file_path - Feature dictionary
    """
    initialise_tf()

    # determine the save path
    save_path = None
    if not save_root is None:
        if frame_dir is None:
            save_path = save_root
        else:
            save_path = os.path.join(save_root, frame_dir)

    path_feature_dic_file = os.path.join(save_path, PATH_FEATURE_DIC_NAME)

    # check if the features have already been extracted
    if force_redo == False and os.path.exists(path_feature_dic_file):
        return load_path_feature_dic(save_path)

    from keras_vggface.vggface import VGGFace
    from tensorflow.keras.models import Model
    from keras_vggface.utils import preprocess_input
    # use vggface2
    base_model = VGGFace(model='resnet50')


    # do not include the fully connected layer
    vgg_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    # load the images from their paths
    temp_images = []
    for image_path in image_files:
        image = load_image_from_file(image_path, None, desired_shape=(224, 224))
        temp_images.append(preprocess_input(image))

    # extract the features
    features = vgg_extractor.predict(np.stack(temp_images))

    # create dictionary containing the file name as the key and the normalised features as the value.
    path_feature_dic = {}
    for i in range(len(image_files)):
        path_feature_dic[image_files[i]] = features[i] #/ np.linalg.norm(features[i])

    # save the file feature dictionary
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(path_feature_dic_file, "wb") as file:
            pickle.dump(path_feature_dic, file)

    # clean local variables
    del base_model
    del vgg_extractor
    del features
    del temp_images

    return path_feature_dic


@processify
def process_get_sil_for_k_tf(k, feature_array):
    try:
        initialise_tf()
        from kmeanstf import TunnelKMeansTF
        print(k)
        kmeans = TunnelKMeansTF(n_clusters=k).fit(feature_array)
        labels = kmeans.labels_
        score = silhouette_score(feature_array, labels, metric='euclidean')
        centroids = kmeans.cluster_centers_
        del kmeans
        del labels
    except Exception as e:
        e
        return -1, -1, -1
    return k, score, centroids


def get_sil_for_k_tf(k, feature_array):
    try:
        import faiss
        print(k)

        kmeans = faiss.Kmeans(d=feature_array.shape[1], k=k, gpu=True, verbose=False)
        kmeans.train(feature_array)
        cluster_centers = kmeans.centroids
        labels = kmeans.index.search(x=feature_array, k=1)[1].reshape(-1)
        score = silhouette_score(feature_array, labels, metric='euclidean')

        del kmeans
        del labels
    except Exception as e:
        print(e)
        return -1, -1, -1
    return k, score, cluster_centers



@processify
def process_get_sil_for_k(k, feature_array):
    try:
        print(k)
        kmeans = KMeans(n_clusters=k, init='k-means++').fit(feature_array)
        score = silhouette_score(feature_array, kmeans.labels_, metric='euclidean')
        centroids = kmeans.cluster_centers_
        del kmeans

    except:
        return -1, -1, -1
    return k, score, centroids


def get_sil_for_k(k, feature_array):
    try:
        print(k)
        kmeans = KMeans(n_clusters=k, init='k-means++').fit(feature_array)
        score = silhouette_score(feature_array, kmeans.labels_, metric='euclidean')
        centroids = kmeans.cluster_centers_
        del kmeans

    except Exception as e:
        return -1, -1, -1
    return k, score, centroids


def determine_optimal_k(path_feature_dic, max_k=None, k_vs_silscore_save_path=None,
                        enable_early_stop=False, early_stop_check_k=None, force_redo=False, must_use_gpu=True,
                        must_use_separate_process=False, save_frequency = 5):
    """
    determine the K that produces the maximum silhouette score
    :param path_feature_dic: a dictionary containing the file paths as keys and the vgg face features as values.
    :param min_kf_threshold: the minimum images to be present.
    :param max_k: the maximum k to check
    :param k_vs_silscore_save_path: the path to save the k vs silhouette score graph
    :param enable_early_stop: True to stop checking silhouette scores if the silhouette trend begins to decrease.
    :param force_redo: True to ignore any saved files
    :param must_use_gpu: True to use the tensorflow implementation (Note at k=600, 22 700 GB of GPU memory is used).
    False to use CPU and RAM
    :param must_use_separate_process: True to to run the KMeans in a separate process. This will allow for memory to
    free when using tensorflow but may take longer
    :return: the k that maximises the silhouette score. Returns -1 if the optimisation failed
    """

    # test if the silhouette score has already been done
    print(f"Working with: {k_vs_silscore_save_path}")
    # create the directories for saving
    if k_vs_silscore_save_path is not None:
        # create the path if it does not exist
        if not os.path.exists(k_vs_silscore_save_path):
            os.makedirs(k_vs_silscore_save_path)

    # maintain the silhouette score
    k_sil_dic = dict()
    # k needs to be a minimum of 2 for clustering
    start_k = 2
    # get the features
    feature_array = np.stack(list(path_feature_dic.values()))
    # determine the stopping point
    total_k = len(feature_array)
    if max_k is None:
        max_k = int(0.8*total_k)

    # try load the ks and sils
    if force_redo == False:
        k_vs_sil_save_path = os.path.join(k_vs_silscore_save_path, K_VS_SIL_PKL)
        if os.path.exists(k_vs_sil_save_path):
            with open(k_vs_sil_save_path, "rb") as file:
                k_vs_sil = pickle.load(file)
                k_sil_dic = dict(zip(k_vs_sil[0], k_vs_sil[1]))
                start_k = k_vs_sil[0][-1]+1


    ks = [i for i in range(start_k, max_k)]

    # determine a possible cutoff threshold
    if early_stop_check_k is None:
        possible_cut_threshold = int(0.6 * max_k)
    else:
        possible_cut_threshold = early_stop_check_k
    early_stop_occurred = False
    # determine if we should stop checking
    if enable_early_stop and (start_k-1== possible_cut_threshold):
        temp_sil = list(k_sil_dic.values())
        temp_ks = list(k_sil_dic.keys())
        silhouette_trend = np.corrcoef(temp_ks, temp_sil)[0][1]
        # if the trend is decreasing. i.e. silhouette score is getting worse
        if silhouette_trend < 0:
            print(
                f"Silhouette trend is {silhouette_trend}. Breaking loop at {possible_cut_threshold} instead of {max_k}")
            # gradient is decreasing, stop testing
            del temp_ks
            del silhouette_trend
            del temp_sil
            early_stop_occurred = True

    save_counter = 1
    if not early_stop_occurred:
        for k in ks:
            if must_use_gpu:
                if must_use_separate_process:
                    current_k, score, _ = process_get_sil_for_k_tf(k, feature_array)
                else:
                    current_k, score, _ = get_sil_for_k_tf(k, feature_array)

            else:
                if must_use_separate_process:
                    current_k, score, _ = process_get_sil_for_k(k, feature_array)
                else:
                    current_k, score, _ = get_sil_for_k(k, feature_array)
            # We ran into an error
            if current_k < 0:
                return -1
            # save the state
            k_sil_dic[k] = score
            # test if we should save the progress
            if save_counter % save_frequency == 0:
                ks = list(k_sil_dic.keys())
                sils = list(k_sil_dic.values())
                with open(os.path.join(k_vs_silscore_save_path, K_VS_SIL_PKL), "wb") as file:
                    pickle.dump([ks, sils], file)
                print(f"Saved K VS SIL: {k} /{max_k}")

            # determine if we should stop checking
            if enable_early_stop and (k == possible_cut_threshold):
                temp_sil = list(k_sil_dic.values())
                temp_ks = list(k_sil_dic.keys())
                silhouette_trend = np.corrcoef(temp_ks, temp_sil)[0][1]
                # if the trend is decreasing. i.e. silhouette score is getting worse
                if silhouette_trend < 0:
                    print(
                        f"Silhouette trend is {silhouette_trend}. Breaking loop at {possible_cut_threshold} instead of {max_k}")
                    # gradient is decreasing, stop testing
                    del temp_ks
                    del silhouette_trend
                    del temp_sil
                    break
                else:
                    del temp_ks
                    del silhouette_trend
                    del temp_sil
            save_counter += 1
    # obtain the K's and silhouette scores
    ks = list(k_sil_dic.keys())
    sils = list(k_sil_dic.values())

    # save the silhouette score graph and features
    if k_vs_silscore_save_path is not None:
        fig = plt.figure()
        plt.plot(ks, sils)
        plt.title("Optimal K")
        plt.ylabel("Silhouette Score")
        plt.xlabel("K")
        plt.ioff()
        fig.savefig(os.path.join(k_vs_silscore_save_path, "k_vs_sil.png"), dpi=300)

        with open(os.path.join(k_vs_silscore_save_path, K_VS_SIL_PKL), "wb") as file:
            pickle.dump([ks, sils], file)
    del feature_array

    # determine the k that produced the highest silhouette score
    if len(sils)>0:
        highest_sil = max(sils)
        highest_sil_index = sils.index(highest_sil)
        optimal_k = ks[highest_sil_index]
        del highest_sil
        del highest_sil_index
    else:
        optimal_k = -1

    # clean up
    del ks
    del sils

    return optimal_k


def get_centroid_frames(optimal_k, path_feature_dic, must_use_gpu=True):
    # get the optimal k


    # if no optimal can be determine, let the user know
    if optimal_k < 0:
        return None
    # get the file paths and features
    file_paths = list(path_feature_dic.keys())
    feature_array = np.stack(list(path_feature_dic.values()))

    # get the centroids for the optimal k
    if must_use_gpu:
        _, _, centroids = get_sil_for_k_tf(optimal_k, feature_array)
    else:
        _, _, centroids = get_sil_for_k(optimal_k, feature_array)

    closest_indices, _ = vq(centroids, feature_array)
    closest_images = [file_paths[i] for i in closest_indices]

    return closest_images


def find_optimal_k(frame_directory, save_root, dataset_root, max_k=None,
                      enable_early_stop=False, early_stop_check_k=None, force_redo=False, must_use_gpu=True,
                      must_use_separate_process=False):
    save_path = os.path.join(save_root, frame_directory)
    new_info_csv_path = os.path.join(save_path, INFO_CSV_NAME)
    if not force_redo and os.path.exists(new_info_csv_path):
        print(f"Keyframe csv found. Skipping: {frame_directory}")
        return

    path_feature_dic_file = load_path_feature_dic(save_path)
    if path_feature_dic_file is None:
        with open(os.path.join(save_root, f"ERROR_MISSING_PATH_FEATURE_PICKLE_{frame_directory.replace('/', '_')}.txt"),
                  'w') as file:
            message = f"Could not find path_feature pickle for: {frame_directory}.\nlocation checked: {save_path}"
            file.write(message)
            print(message)
        return
    else:
        optimal_k = determine_optimal_k(path_feature_dic_file, max_k, save_path,
                                        enable_early_stop, early_stop_check_k, force_redo, must_use_gpu,
                                        must_use_separate_process)


        if optimal_k < 0:
            with open(os.path.join(save_root, f"ERROR_FINDING_OPTIMAL_K_{frame_directory.replace('/', '_')}.txt"),
                      "w") as file:
                message = f"Could not Optimal K for: {frame_directory}"
                file.write(message)
                print(message)
            return None
        else:
            info_dic = {
                "video_frames": len(path_feature_dic_file),
                "frames_present": -1 ,
                "optimal_k": optimal_k,
                        }
            info_csv = pd.DataFrame.from_dict([info_dic])

            info_csv.to_csv(new_info_csv_path, index=False)

def extract_keyframes(frame_directory, save_root, category_mean_dic, category_col, min_thresh_function, force_redo=False, must_use_gpu=True):
    save_path = os.path.join(save_root, frame_directory)
    new_info_csv_path = os.path.join(save_path, INFO_CSV_NAME)
    if not os.path.exists(new_info_csv_path):
        with open(os.path.join(save_root, f"ERROR_MISSING_INFO_CSV_{frame_directory.replace('/', '_')}.txt"),
                  'w') as file:
            message = f"Could not find info csv for: {frame_directory}.\nlocation checked: {save_path}"
            file.write(message)
            print(message)
        return
    info_frame = pd.read_csv(new_info_csv_path)
    if not force_redo:
        if info_frame["frames_present"][0] >= 0:
            print(f"Keyframes already found. Skipping: {frame_directory}")
            return

    path_feature_dic_file = load_path_feature_dic(save_path)
    if path_feature_dic_file is None:
        with open(os.path.join(save_root, f"ERROR_MISSING_PATH_FEATURE_PICKLE_{frame_directory.replace('/', '_')}.txt"),
                  'w') as file:
            message = f"Could not find path_feature pickle for: {frame_directory}.\nlocation checked: {save_path}"
            file.write(message)
            print(message)
        return
    else:
        # remove any frames present in the directory
        image_files = obtain_file_paths(save_path, r"^frame")
        for image in image_files:
            os.remove(image)
        category = info_frame[category_col].tolist()
        if type(category) is list:
            category = category[0]
        mean_k = int(round(category_mean_dic[category]))

        if min_thresh_function is not None:
            mean_k = min_thresh_function(info_frame, mean_k)

        # find the keyframes
        closest_images = get_centroid_frames(mean_k,path_feature_dic_file, must_use_gpu)
        if closest_images is None:
            with open(os.path.join(save_root, f"ERROR_FINDING_CENTROIDS_{frame_directory.replace('/', '_')}.txt"),
                      "w") as file:
                message = f"Could not find centroids for: {frame_directory}"
                file.write(message)
                print(message)
            return None
        else:

            for image_path in closest_images:
                # get the file name
                file_name = os.path.basename(image_path)
                new_path = os.path.join(save_path, file_name)
                # copy the frame to the new location
                shutil.copy(image_path, new_path)

            info_frame['frames_present'] = len(closest_images) # number of KF
            info_frame['average_k'] = mean_k # number of KF
            info_frame.to_csv(new_info_csv_path, index=False)


def extract_vgg_features(config):
    file_directory = config['file_path']
    frames_location = os.path.join(config['dataset_root'], file_directory)
    file_paths = obtain_file_paths(frames_location, r"^frame_")
    if len(file_paths) <= 0:
        with open(os.path.join(config['save_root'], f"ERROR_MISSING_FRAMES_{file_directory.replace('/', '_')}.txt"),
                  "w") as file:
            message = f"Could not find frames for {file_directory}.\nLocation checked: {frames_location}"
            file.write(message)
            print(message)
        return
    if len(file_paths) < 3:
        with open(os.path.join(config['save_root'], f"ERROR_NOT_ENOUGH_FRAMES_{file_directory.replace('/', '_')}.txt"),
                  "w") as file:
            message = f"There are not enough frames to cluster. At least 3 frames are required.\nLocation checked: {frames_location}"
            file.write(message)
            print(message)
        return

    extract_features_from_paths(image_files=file_paths[:3000], save_root=config['save_root'], frame_dir=file_directory, force_redo=config['force_redo'],
                                must_use_separate_process=config["must_use_separate_process"])


def find_keyframes(config):
    frame_directory = config['frame_directory']
    save_root = config['save_root']
    dataset_root = config['dataset_root']
    max_k = config['max_k']
    enable_early_stop = config['enable_early_stop']
    early_stop_check_k = config['early_stop_check_k']
    force_redo = config['force_redo']
    must_use_gpu = config['must_use_gpu']
    must_use_separate_process = config['must_use_separate_process']

    find_optimal_k(frame_directory, save_root, dataset_root, max_k=max_k,
                   enable_early_stop=enable_early_stop, early_stop_check_k=early_stop_check_k, force_redo=force_redo,
                   must_use_gpu=must_use_gpu, must_use_separate_process=must_use_separate_process)

def find_mean_optimal_keyframes(config):
    frame_directory = config['frame_directory']
    save_root = config['save_root']
    force_redo = config['force_redo']
    must_use_gpu = config['must_use_gpu']
    category_mean_dic = config['category_mean_dic']
    category_col = config["category_col"]
    min_thresh_function = config['min_thresh_function']
    extract_keyframes(frame_directory, save_root, category_mean_dic, category_col,min_thresh_function,
                      force_redo=force_redo, must_use_gpu=must_use_gpu)


def create_kf_dataset(dataset_root, save_root, tune_root, dataset_csv_name, mean_col_name, min_thresh_function, tune_extraction_gpu=0.5,
                      tune_extraction_cpu=5, must_redo_feature_extraction=False,
                      must_use_separate_extraction_process=False,
                      tune_kf_gpu=0.25, tune_kf_cpu=5, must_redo_kf_extraction=False, must_use_gpu=True,
                      min_kf_threshold=None, max_k=None, enable_early_stop=True, early_stop_check_k=None,
                      must_use_separate_kf_process=False, all_video_files=None, is_ray=True,use_only_training_information=True, ds_name=None):
    if dataset_csv_name is not None:
        if ".csv" in dataset_csv_name:
            dataset_name = os.path.splitext(dataset_csv_name)[0]
        else:
            dataset_name = dataset_csv_name
    else:
        dataset_name = ds_name
    tune_extraction_csv = f"{dataset_name}_feature_extractor.csv"
    tune_extraction_experiment_name = f"{dataset_name}_extraction_creator_" + datetime.now().strftime(
        "%m_%d_%Y_%H_%M_%S")

    tune_kf_csv = f"{dataset_name}_optimal_k.csv"
    tune_kf_experiment_name = f"{dataset_name}_optimal_k" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    tune_extract_kf_csv = f"{dataset_name}_kf_extractor.csv"
    tune_extract_kf_experiment_name = f"{dataset_name}_kf_extractor_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    # create the directories
    if not os.path.exists(tune_root):
        os.makedirs(tune_root)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # move any error files found
    move_error_files(save_root)


    print("~~~~~~~~~~~~~~~~~~ PASS 1: Feature Extraction ~~~~~~~~~~~~~~~~~~")
    video_files_remaining = []
    # if we do not want to re-process a video
    if not must_redo_feature_extraction:
        for video_path in all_video_files:
            # see if the vgg face features have already been extracted
            path_feature_dic_file = os.path.join(save_root, video_path, PATH_FEATURE_DIC_NAME)
            if not os.path.exists(path_feature_dic_file):
                # this video still needs to be processed
                video_files_remaining.append(video_path)
    else:
        video_files_remaining = all_video_files

    if len(video_files_remaining) > 0:
        if is_ray:
            # create the tune config
            tune_extraction_config = {
                "file_path": tune.grid_search(video_files_remaining),
                "save_root": save_root,
                "dataset_root": dataset_root,
                "force_redo": must_redo_feature_extraction,
                "must_use_separate_process": must_use_separate_extraction_process
            }

            # start ray tune
            analysis = tune.run(extract_vgg_features, config=tune_extraction_config, local_dir=tune_root,
                                name=tune_extraction_experiment_name,
                                resources_per_trial={"cpu": tune_extraction_cpu, "gpu": tune_extraction_gpu},
                                resume="AUTO")
            df = analysis.results_df
            df.to_csv(os.path.join(tune_root, tune_extraction_csv))
        else:
            for file in video_files_remaining:
                extract_vgg_features({
                    "file_path": file,
                    "save_root": save_root,
                    "dataset_root": dataset_root,
                    "force_redo": must_redo_feature_extraction,
                    "must_use_separate_process": must_use_separate_extraction_process
                })



    print("~~~~~~~~~~~~~~~~~~ PASS 2: Find Optimal Keyframes ~~~~~~~~~~~~~~~~~~")
    video_files_remaining = []
    # if we do not want to re-process a video
    if must_redo_kf_extraction == False:
        for video_path in all_video_files:
            # see if the vgg face features have already been extracted
            temp_csv_file_path = os.path.join(save_root, video_path, INFO_CSV_NAME)
            if not os.path.exists(temp_csv_file_path):
                # this video still needs to be processed
                video_files_remaining.append(video_path)
    else:
        video_files_remaining = all_video_files

    if len(video_files_remaining) > 0:
        if is_ray:
            tune_kf_config = {
                'frame_directory': tune.grid_search(video_files_remaining),
                'save_root': save_root,
                'dataset_root': dataset_root,
                'max_k': max_k,
                'enable_early_stop': enable_early_stop,
                'early_stop_check_k': early_stop_check_k,
                'force_redo': must_redo_kf_extraction,
                'must_use_gpu': must_use_gpu,
                'must_use_separate_process': must_use_separate_kf_process,
            }

            # start ray tune
            analysis = tune.run(find_keyframes, config=tune_kf_config, local_dir=tune_root, name=tune_kf_experiment_name,
                                resources_per_trial={"cpu": tune_kf_cpu, "gpu": tune_kf_gpu}, resume="AUTO")
            df = analysis.results_df
            df.to_csv(os.path.join(tune_root, tune_kf_csv))
        else:
            for file in video_files_remaining:
                find_keyframes({
                    'frame_directory': file,
                    'save_root': save_root,
                    'dataset_root': dataset_root,
                    'min_kf_threshold': min_kf_threshold,
                    'max_k': max_k,
                    'enable_early_stop': enable_early_stop,
                    'early_stop_check_k': early_stop_check_k,
                    'force_redo': must_redo_kf_extraction,
                    'must_use_gpu': must_use_gpu,
                    'must_use_separate_process': must_use_separate_kf_process,
                })
    print("~~~~~~~~~~~~~~~~~~ PASS 3: CSV Aggregation Phase 1 ~~~~~~~~~~~~~~~~~~")
    # loop through directories and get all csv files
    csv_files = obtain_file_paths(save_root, INFO_CSV_NAME)
    combine_csvs(csv_files, save_root, dataset_csv_name)


    print("~~~~~~~~~~~~~~~~~~ PASS 4: Extract Optimal Keyframes ~~~~~~~~~~~~~~~~~~")
    video_files_remaining = []
    # if we do not want to re-process a video
    if must_redo_feature_extraction == False:
        for video_path in all_video_files:
            # see if the vgg face features have already been extracted
            temp_csv_file_path = os.path.join(save_root, video_path, INFO_CSV_NAME)
            if not os.path.exists(temp_csv_file_path):
                # this video still needs to be processed
                video_files_remaining.append(video_path)
            else:
                temp_df = pd.read_csv(temp_csv_file_path)
                if temp_df["frames_present"][0] <0:
                    video_files_remaining.append(video_path)

    else:
        video_files_remaining = all_video_files

    print("~~~~~~~~~~~~~~~~~~ PASS 5: CSV Aggregation ~~~~~~~~~~~~~~~~~~")
    # loop through directories and get all csv files
    csv_files = obtain_file_paths(save_root, INFO_CSV_NAME)
    combine_csvs(csv_files, save_root, dataset_csv_name)

    #   produce keyframe reduction
    view_keyframe_reduction(os.path.join(save_root, dataset_csv_name), save_root, is_pdf=True)




if __name__ == "__main__":
    # dataset root is the path to the siw dataset containing the extracted face frames
    dataset_root = "/home/jarred/Documents/Generated/Generated/SIW_90"
    save_root = "/home/jarred/Documents/Ablation/SIW_GAN3"

    # place the tune directory by the saved dataset directory
    dataset_name = "SIW_KF_GAN3"
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
    tune_kf_gpu = 0.2  # 6 processes
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


    all_videos=[
                './ASUS',
                './IP7P',
                './IPP2017',
                './P',
                './SGS8',

                ]


    ds_name = "GAN3"

    create_kf_dataset(dataset_root, save_root, tune_root, DATASET_CSV_NAME, None, min_thresh_function, tune_extraction_gpu,
                      tune_extraction_cpu, must_redo_feature_extraction,
                      must_use_separate_extraction_process,
                      tune_kf_gpu, tune_kf_cpu, must_redo_kf_extraction, must_use_gpu,
                      min_kf_threshold, max_k, enable_early_stop, early_stop_check_k,
                      must_use_separate_kf_process, is_ray=True, all_video_files=all_videos,use_only_training_information=use_only_training_information, ds_name=ds_name)
