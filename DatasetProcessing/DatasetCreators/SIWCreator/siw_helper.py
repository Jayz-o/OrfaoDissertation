import os
import re

from DatasetProcessing.DatasetCreators.FaceHelpers.face_detection_helper import detect_face_in_video

SIW_VIDEO_PATTERN = r"^\d{3}-[1,2]-([1,2]-[1-2]|3-[1-4])-[1,2]\.mov"



def produce_info_csv(video_file_path, video_frames, frames_present):
    """
    Produce the csv info for 1 video file. The video name '001-1-1-1-1.mov' relates to
    SubjectID_SensorID_TypeID_MediumID_SessionID.mov
    :param video_file_path: The path to the video
    :param video_frames: The number of frames in the original video
    :param frames_present: The frames present after processing
    :return: a dictionary containing the csv information
    """

    # setup the spoof in the wild naming dictionaries
    sensor_id_dic = {
            "1": "Canon_EOS_T6",
            "2": "Logitech_C920_Webcam"
        }
    type_id_dic = {
        "1": "Live",
        "2": "Print",
        "3": "Replay"
    }
    medium_id_lighting_dic = {
        "1": "NLV",
        "2": "ELV",
    }
    medium_id_resolution_dic = {
        "1": "HR", # HR: High res (5184 x 3456).
        "2": "LR", # LR: Low Res (1920 x 1080).
    }
    medium_id_device_dic = {
        "1": "IPP2017", # IPP2017: iPad Pro 2017.
        "2": "IP7P",    # IP7P: iPhone 7 plus.
        "3": "ASUS",    # ASUS: Asus MBI68B.
        "4": "SGS8",    # SGS8: Samsung Galaxy S8
    }

    session_id_papertype_dic = {
        "1": "GP",  # GP: Glossy Paper
        "2": "MP"  # MP: Matt Paper
    }
    session_id_motion_dic = {
        "1": "MBF",  # MBF: Move Backward and forward.
        "2": "YARFEC"  # YARFEC: Yaw-angle rotation & facial expression change
    }
    session_id_selection_dic ={
        "1": "RS",  # random selection
        "2": "RS",
    }

    # get the video name
    video_name = os.path.basename(video_file_path)
    # test the video file path is in the correct format
    matches = re.search(SIW_VIDEO_PATTERN, video_name)
    if matches is None:
        raise TypeError("The video file path is not in the form SubjectID_SensorID_TypeID_MediumID_SessionID.mov. "
                        "e.g.: 001-1-1-1-1.mov")
    # get the video name without the extension
    video_name = os.path.splitext(video_name)[0]
    # obtain the IDS from the video name: SubjectID-SensorID-TypeID-MediumID-SessionID
    video_bits = video_name.split('-')
    subject_number = int(video_bits[0])
    sensor_id = video_bits[1]
    type_id = video_bits[2]
    medium_id = video_bits[3]
    session_id = video_bits[4]

    # sensor ID and Type ID are independent
    sensor_name = sensor_id_dic[sensor_id]
    type_name = type_id_dic[type_id]

    # medium ID and Session ID are dependent
    # Live attack
    if type_id == "1":
        attack_category = "N"
        ground_truth = "real"
        medium_name = medium_id_lighting_dic[medium_id]
        session_name = session_id_motion_dic[session_id]

    elif type_id == "2":
        attack_category = "P"
        ground_truth = "spoof"
        medium_name = medium_id_resolution_dic[medium_id]
        session_name = session_id_papertype_dic[session_id]

    elif type_id == "3":
        attack_category = "R"
        ground_truth = "spoof"
        medium_name = medium_id_device_dic[medium_id]
        session_name = session_id_selection_dic[session_id]
    else:
        raise TypeError(f"typed id is not handled: {type_id}")


    if "Train" in video_file_path:
        usage_type = "train"
    elif "Test" in video_file_path:
        usage_type = "test"
    else:
        raise TypeError("Cannot determine usage type. Train or Test is not present in the video path")

    info_dic = dict()
    info_dic["usage_type"] = usage_type
    info_dic["ground_truth"] = ground_truth
    info_dic["subject_number"] = subject_number
    info_dic["video_name"] = f"{video_name}.mov"
    info_dic["video_frames"] = video_frames
    info_dic["frames_present"] = frames_present
    info_dic["attack_category"] = attack_category
    info_dic["directory_path"] = os.path.join(usage_type, ground_truth, str(subject_number), video_name.split(".")[0])
    info_dic["sensor_id"] = sensor_id
    info_dic["sensor_name"] = sensor_name
    info_dic["type_id"] = type_id
    info_dic["type_name"] = type_name
    info_dic["medium_id"] = medium_id
    info_dic["medium_name"] = medium_name
    info_dic["session_id"] = session_id
    info_dic["session_name"] = session_name

    return info_dic

def detect_faces_siw(video_path):
    video_information = detect_face_in_video(video_path, zoom_factor=0.31, vertical_face_location_shift=50)
    return video_information

def convert_attack_category_to_groundtruth(attack_category):
    attack_category_to_groundtruth_dic = {
                    'ASUS-IP7P-IPP2017':"spoof",
                   'ASUS-IP7P-SGS8':"spoof",
                   'ASUS-IPP2017-SGS8':"spoof",
                   'IP7P-IPP2017-SGS8':"spoof",
                   'ASUS':"spoof",
                   'IP7P':"spoof",
                   'IPP2017':"spoof",
                   'SGS8':"spoof",
                   'P':"spoof",
                   'R':"spoof",
                    'N': "real"
    }

    return attack_category_to_groundtruth_dic[attack_category]

def convert_attack_category_to_medium_list(attack_category):
    attack_category_to_medium_list_dic = {
                    'ASUS-IP7P-IPP2017':["ASUS","IP7P","IPP2017"],
                   'ASUS-IP7P-SGS8': ['ASUS','IP7P','SGS8'],
                   'ASUS-IPP2017-SGS8': ['ASUS','IPP2017','SGS8'],
                   'IP7P-IPP2017-SGS8': ['IP7P','IPP2017','SGS8'],
                    'ASUS': ["ASUS"],
                    'IP7P': [ 'IP7P'],
                    'IPP2017': ['IPP2017'],
                    'SGS8': ['SGS8'],
                   'P':["HR", "LR"],
                   'R':['ASUS','IPP2017','SGS8', 'IP7P'],
                    'N': ["ELV", "NLV"],
    }

    return attack_category_to_medium_list_dic[attack_category]

if __name__ == "__main__":
    print(produce_info_csv("Train/017-1-3-3-1.mov", 486, 454))