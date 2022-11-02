
from DatasetProcessing.DatasetCreators.FaceHelpers.dataset_creator_helper import create_dataset
from DatasetProcessing.DatasetCreators.SIWCreator.siw_helper import produce_info_csv, SIW_VIDEO_PATTERN, detect_faces_siw


"""
Videos with errors: 
041-2-3-3-1.mov
041-2-3-2-1.mov
101-2-3-3-1.mov
081-2-3-2-1.mov
121-2-3-2-1.mov
121-2-3-3-1.mov
020-2-3-3-2.mov
161-2-3-3-1.mov
141-2-3-3-1.mov
061-2-3-3-1.mov
021-2-3-2-1.mov
101-2-3-2-1.mov
161-2-3-2-1.mov
141-2-3-2-1.mov
061-2-3-2-1.mov

error_files = [
     "train/spoof/41/041-2-3-3-1",
     "train/spoof/41/041-2-3-2-1",
     "train/spoof/101/101-2-3-3-1",
     "train/spoof/81/081-2-3-2-1",
     "train/spoof/121/121-2-3-2-1",
     "train/spoof/121/121-2-3-3-1",
     "test/spoof/20/020-2-3-3-2",
     "test/spoof/161/161-2-3-3-1",
     "test/spoof/141/141-2-3-3-1",
     "test/spoof/61/061-2-3-3-1",
     "test/spoof/21/021-2-3-2-1",
     "train/spoof/101/101-2-3-2-1",
     "test/spoof/161/161-2-3-2-1",
     "test/spoof/141/141-2-3-2-1",
     "test/spoof/61/061-2-3-2-1",
     ]

 error_files = [
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/041/041-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/041/041-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/101/101-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/081/081-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/121/121-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/121/121-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/020/020-2-3-3-2.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/161/161-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/141/141-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/061/061-2-3-3-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/021/021-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Train/spoof/101/101-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/161/161-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/141/141-2-3-2-1.mov",
        "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release/Test/spoof/061/061-2-3-2-1.mov",
    ]

"""
# globals to be used by other scripts
DATASET_CSV_NAME = "siw.csv"


if __name__ == "__main__":
    # adjust these variables
    dataset_root = "/home/jarred/Documents/Datasets/SIW_Combined/SiW_release"
    save_root = "/home/jarred/Documents/Datasets/TEMP3"
    # max GPU memory used was 3819 MiB
    tune_gpu = 0.2
    tune_cpu = 4
    must_redo = False
    verbose = False

    create_dataset(dataset_root, save_root, DATASET_CSV_NAME, SIW_VIDEO_PATTERN, produce_info_csv, detect_faces_siw,
                   tune_cpu, tune_gpu)
