import os.path
import cv2

from DatasetProcessing.DatasetCreators.FaceHelpers.face_align_helper import FaceAligner
from Helpers.image_helper import create_image_grid, obtain_file_paths, load_image_from_file


def try_initialise_gpu():
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e.args)
def detect_face_in_video(video_path, zoom_factor, vertical_face_location_shift, desired_width_height=256):
    """
    detect faces in a video
    :param video_path: The full path to the video
    :param zoom_factor: increase to remove more background
    :param vertical_face_location_shift: increase to move the crop region towards the chin
    :param desired_width_height: The desired cropped image dimension
    :return: a dictionary with the following structure {"aligned_faces" : aligned_faces,
                                                        "video_frames" : video_frames,
                                                        "aligned_faces_count": len(aligned_faces)
                                                        }
    """
    if not os.path.exists(video_path):
        raise TypeError(f"Could not locate the video_path: {video_path}")

    try_initialise_gpu()

    # create the aligner
    face_aligner = FaceAligner(desiredFaceWidth=desired_width_height, zoom_out_factor=zoom_factor)

    # streaming variables
    vs = cv2.VideoCapture(video_path)
    frame_available = True
    aligned_faces = list()
    video_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)

    # while the video is streamable
    while frame_available:
        # read the next frame from the file
        (frame_available, frame) = vs.read()
        if frame_available is False or frame is None:
            break
        temp_frame = frame.copy()
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        aligned_face = face_aligner.align(temp_frame, shift=vertical_face_location_shift)

        if aligned_face is not None:
            aligned_faces.append(aligned_face)

        # clean local variables
        del temp_frame
        del aligned_face

    info_dic = {
        "aligned_faces" : aligned_faces,
        "video_frames" : video_frames,
        "aligned_faces_count": len(aligned_faces)
    }
    # clean variables
    del vs
    del aligned_faces
    del face_aligner
    del frame_available
    del video_frames

    return info_dic

def detect_face_in_folder(folder_path, zoom_factor, vertical_face_location_shift, desired_width_height=256, frame_pattern=r"^seed", ignore_detection=False):
    """
    detect faces in a video
    :param video_path: The full path to the video
    :param zoom_factor: increase to remove more background
    :param vertical_face_location_shift: increase to move the crop region towards the chin
    :param desired_width_height: The desired cropped image dimension
    :return: a dictionary with the following structure {"aligned_faces" : aligned_faces,
                                                        "video_frames" : video_frames,
                                                        "aligned_faces_count": len(aligned_faces)
                                                        }
    """
    if not os.path.exists(folder_path):
        raise TypeError(f"Could not locate the video_path: {folder_path}")

    try_initialise_gpu()

    # create the aligner
    face_aligner = FaceAligner(desiredFaceWidth=desired_width_height, zoom_out_factor=zoom_factor)

    aligned_faces = list()
    images = obtain_file_paths(folder_path, frame_pattern)

    error_frames = []
    # while the video is streamable
    original_non_errors = []
    for frame_path in images:
        frame = load_image_from_file(frame_path)

        if ignore_detection:
            aligned_face = frame.numpy()
        else:
            aligned_face = face_aligner.align(frame.numpy(), shift=vertical_face_location_shift)

        if aligned_face is not None:
            aligned_faces.append(aligned_face)
            original_non_errors.append(frame.numpy())
        else:
            error_frames.append(frame_path)

        # clean local variables
        # del aligned_face

    info_dic = {
        "aligned_faces" : aligned_faces,
        "non_error_frames" : original_non_errors,
        "video_frames" : len(images),
        "aligned_faces_count": len(aligned_faces),
        "error_frames": error_frames
    }
    # clean variables
    del aligned_faces
    del face_aligner

    return info_dic

def visualise_face_detection_variables(video_path, zoom_factor_list, vertical_shift_factor_list, desired_width_height=256, must_show=True, save_path=None):
    if not os.path.exists(video_path):
        raise TypeError(f"Could not locate the video_path: {video_path}")

    # streaming variables
    vs = cv2.VideoCapture(video_path)
    frame_available = True
    aligned_faces = list()
    # sort lists
    zoom_factor_list.sort()
    vertical_shift_factor_list.sort()

    total_iterations = len(zoom_factor_list) * len(vertical_shift_factor_list)
    labels = []
    # while the video is streamable
    while frame_available:
        # read the next frame from the file
        (frame_available, frame) = vs.read()
        if frame_available is False or frame is None:
            break
        temp_frame = frame.copy()
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        must_skip_frame = False

        for zoom_factor in zoom_factor_list:
            # create the aligner
            face_aligner = FaceAligner(desiredFaceWidth=desired_width_height, zoom_out_factor=zoom_factor)
            for vertical_shift in vertical_shift_factor_list:
                aligned_face = face_aligner.align(temp_frame, shift=vertical_shift)
                # test if a face was found
                if aligned_face is None:
                    # skip to the next frame
                    must_skip_frame = True
                    break
                else:
                    # keep the aligned face
                    aligned_faces.append(aligned_face)
                    labels.append(f"Z: {zoom_factor} | VS: {vertical_shift}")
            if must_skip_frame:
                # skip to the next frame
                break

        # we only need 1 successful frame for the visualisation
        if len(aligned_faces) == total_iterations:
            break

    create_image_grid(images=aligned_faces, num_cols=len(vertical_shift_factor_list),class_names=labels, file_name=save_path, must_show=must_show, dpi=100, scale=4)

def visualise_preprocessing(video_path, save_path, zoom_factor, vertical_face_location_shift):
    # streaming variables
    vs = cv2.VideoCapture(video_path)
    frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_available = True
    aligned_faces = list()
    i = 0
    # while the video is streamable
    while frame_available:


        # read the next frame from the file
        (frame_available, frame) = vs.read()
        if frame_available is False or frame is None:
            break
        if i < int(frames/2):
            i+=1
            continue
        # plt.imshow(frame)
        # plt.show()
        temp_frame = frame.copy()
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        must_skip_frame = False

        # create the aligner
        # video_path, zoom_factor = 0.28, vertical_face_location_shift = 100
        face_aligner = FaceAligner(desiredFaceWidth=256, zoom_out_factor=zoom_factor)

        aligned_faces = face_aligner.align(temp_frame, shift=vertical_face_location_shift, visualise_process=True)
        # test if a face was found
        if aligned_faces is None:
            # skip to the next frame
            continue
        else:
            # keep the aligned face
            import tensorflow as tf

            # aligned_faces  = [tf.image.resize(image, (256, 256)) for image in aligned_faces]
            create_image_grid(images=aligned_faces[:2], num_cols=2, class_names=["1", "2"],
                      file_name=save_path+ "processed1.png", must_show=False, dpi=1000)
            create_image_grid(images=aligned_faces[2:], num_cols=2, class_names=["3", "4"],
                              file_name=save_path+ "processed2.png", must_show=False, dpi=1000)
            break

if __name__ == "__main__":
    visualise_preprocessing("/home/jarred/Downloads/HR_2.avi", "/home/jarred/Downloads/", 0.28, 100)
    # visualise_face_detection_variables("/home/jarred/Downloads/HR_2.avi", [0.1,0.2, 0.3], [0,100,200])
