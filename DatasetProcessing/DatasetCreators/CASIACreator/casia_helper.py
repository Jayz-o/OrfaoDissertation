import os
import re
from DatasetProcessing.DatasetCreators.FaceHelpers.face_detection_helper import detect_face_in_video
# Globals
CASIA_VIDEO_PATTERN = r"^([1-8]|HR_[1-4]).avi"

CASIA_VIDEO_TO_ATTACK_CATEGORY_DIC = {
        "1.avi": "N1",
        "2.avi": "N2",
        "3.avi": "W1",
        "4.avi": "W2",
        "5.avi": "C1",
        "6.avi": "C2",
        "7.avi": "R1",
        "8.avi": "R2",
        "HR_1.avi": "HR_N",
        "HR_2.avi": "HR_W",
        "HR_3.avi": "HR_C",
        "HR_4.avi": "HR_R",
    }

def produce_info_csv(video_file_path, video_frames, frames_present):
    """
    CASIA-FA/train_release/1/1.avi
    :param video_file_path: The path to the video
    :param video_frames: The number of frames in the original video
    :param frames_present: The frames present after processing
    :return: a dictionary containing the csv information
    """



    ground_truth_dic = {
        "1.avi": "real",
        "2.avi": "real",
        "3.avi": "spoof",
        "4.avi": "spoof",
        "5.avi": "spoof",
        "6.avi": "spoof",
        "7.avi": "spoof",
        "8.avi": "spoof",
        "HR_1.avi": "real",
        "HR_2.avi": "spoof",
        "HR_3.avi": "spoof",
        "HR_4.avi": "spoof",
    }

    # get the video name
    video_name_with_extention = os.path.basename(video_file_path)

    # test the video file path is in the correct format
    matches = re.search(CASIA_VIDEO_PATTERN, video_name_with_extention)
    if matches is None:
        raise TypeError("The video file path is not in the form SubjectID_SensorID_TypeID_MediumID_SessionID.mov. "
                        "e.g.: [1,8].avi or HR_[1,4].avi")
    # get the video name without the extension
    video_name = os.path.splitext(video_name_with_extention)[0]
    # get the subject number
    subject_number = os.path.basename(os.path.dirname(video_file_path))
    # get the release type
    release_type = os.path.basename(os.path.dirname(os.path.dirname(video_file_path)))

    if "train" in release_type:
        usage_type = "train"
    elif "test" in release_type:
        usage_type = "test"
    else:
        raise TypeError(f"Could not determine release type: {release_type}")

    attack_category = CASIA_VIDEO_TO_ATTACK_CATEGORY_DIC[video_name_with_extention]
    ground_truth = ground_truth_dic[video_name_with_extention]

    info_dic = dict()
    info_dic["usage_type"] = usage_type
    info_dic["ground_truth"] = ground_truth
    info_dic["subject_number"] = subject_number
    info_dic["video_name"] = video_name_with_extention
    info_dic["video_frames"] = video_frames
    info_dic["frames_present"] = frames_present
    info_dic["attack_category"] = attack_category
    info_dic["directory_path"] = os.path.join(usage_type, ground_truth, subject_number, f"{subject_number}-{video_name}")

    return info_dic


def detect_faces_casia(video_path):
    video_name = os.path.basename(video_path)
    if "HR_" in video_name:
        # higher resolution videos require more of a vertical shift to include the mouth
        video_information = detect_face_in_video(video_path, zoom_factor=0.28, vertical_face_location_shift=100)
    else:
        video_information = detect_face_in_video(video_path, zoom_factor=0.28, vertical_face_location_shift=50)
    return video_information

def convert_attack_category_to_groundtruth(attack_category):
    attack_category_to_groundtruth_dic = {
        "N1": "real",
        "N2": "real",
        "W1": "spoof",
        "W2": "spoof",
        "C1": "spoof",
        "C2": "spoof",
        "R1": "spoof",
        "R2": "spoof",
        "HR_N": "real",
        "HR_W": "spoof",
        "HR_C": "spoof",
        "HR_R": "spoof",
    }
    return attack_category_to_groundtruth_dic[attack_category]

def convert_attack_category_to_medium_list(attack_category):

    return [attack_category]

if __name__ == "__main__":
    print(produce_info_csv("CASIA-FA/test_release/30/3.avi", 486, 454))


