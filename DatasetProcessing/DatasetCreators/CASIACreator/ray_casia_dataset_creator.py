from DatasetProcessing.DatasetCreators.FaceHelpers.dataset_creator_helper import create_dataset
from DatasetProcessing.DatasetCreators.CASIACreator.casia_helper import produce_info_csv, CASIA_VIDEO_PATTERN, detect_faces_casia


# globals to be used by other scripts
DATASET_CSV_NAME = "casia.csv"

if __name__ == "__main__":
    # adjust these variables
    dataset_root = "/home/jarred/Documents/Datasets/CASIA_RAW/CASIA-FA"
    save_root = "/home/jarred/Documents/Datasets/TEST"
    #   10 processes uses 23 946 GB of GPU mem. Thus use 8
    tune_gpu = 0.125
    tune_cpu = 3
    must_redo = False
    verbose = True

    create_dataset(dataset_root, save_root, DATASET_CSV_NAME, CASIA_VIDEO_PATTERN, produce_info_csv, detect_faces_casia,
                   tune_cpu, tune_gpu)

