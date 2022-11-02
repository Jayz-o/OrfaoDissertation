import os
import shutil

from tqdm import tqdm

from Antispoofing.AntispoofHelpers.dataset_helper import get_dataframe_by_usage_type, make_attack_category_frame, \
    make_medium_name_frame
from Helpers.image_helper import obtain_file_paths
from NVIDIA_STYLEGAN3.dataset_tool import convert_dataset


def _create_dataset(output_root, new_destination):
    try:
        convert_dataset([f"--source={output_root}",
                         f"--dest={new_destination}",
                         f"--resolution=256x256"])
    except:
        pass
def copy_to_folder(output_root, dataset_root, directory_path, verbose=False):
    directory_bits = directory_path.split("/")
    new_name = f"{directory_bits[-2]}_{directory_bits[-1]}"
    image_files = obtain_file_paths(os.path.join(dataset_root, directory_path), r"^frame")
    if verbose:
        for file in tqdm(image_files):
            new_destination = os.path.join(output_root,f"{new_name}_{os.path.basename(file)}")
            shutil.copy(file, new_destination)
    else:
        for file in image_files:
            new_destination = os.path.join(output_root,f"{new_name}_{os.path.basename(file)}")
            shutil.copy(file, new_destination)



def copy_attack_categories_to_folder(train_frame, categories, dataset_root, output_root, subject_number=None, verbose=False, must_redo=False, copy_root=None):
    copy_dataset_files_to_folder(train_frame, categories, dataset_root, output_root, make_attack_category_frame,
                                 subject_number, verbose, must_redo, copy_root)

def copy_dataset_files_to_folder(train_frame, categories, dataset_root, output_root, make_frame_func, subject_number=None, verbose=False, must_redo=False, copy_root=None):
    if not os.path.exists(dataset_root):
        raise TypeError(f"Could not locate dataset root: {dataset_root}")
    # create an output folder based on the categories
    dataset_name = os.path.basename(dataset_root)
    folder_name = ""

    for i, cat in enumerate(categories):
        if i == len(categories) -1:
            folder_name += f"{cat}"
        else:
            folder_name += f"{cat}-"

    if subject_number is None:
        copy_output_root = os.path.join(output_root, dataset_name, folder_name)
        new_destination = os.path.join(output_root, dataset_name, f"{folder_name}.zip")
    else:
        dataset_name += f"_{subject_number}"
        copy_output_root = os.path.join(output_root, dataset_name, folder_name)
        new_destination = os.path.join(output_root, dataset_name, f"{folder_name}.zip")
    if os.path.exists(new_destination) and must_redo is False:
        print("Folder already exists.... skipping file")
    else:
        if not os.path.exists(copy_output_root):
            os.makedirs(copy_output_root)
        # filter the frame to the provided categories
        category_frame = make_frame_func(train_frame, categories)
        paths = category_frame['directory_path'].tolist()
        if verbose:
            for path in tqdm(paths):
                copy_to_folder(copy_output_root, dataset_root, path)
        else:
            for path in paths:
                copy_to_folder(copy_output_root, dataset_root, path)

        _create_dataset(copy_output_root, new_destination)
        # remove the folder
        shutil.rmtree(copy_output_root)
        if copy_root is not None:
            if subject_number is None:
                copy_destination = os.path.join(copy_root, dataset_name, f"{folder_name}.zip")
            else:
                copy_destination = os.path.join(copy_root, dataset_name, f"{folder_name}.zip")
            shutil.copy(new_destination, copy_destination)

def copy_medium_names_to_folder(train_frame, medium_names, dataset_root, output_root, subject_number=None, verbose=False, must_redo=False, copy_root=None):
    copy_dataset_files_to_folder(train_frame, medium_names, dataset_root, output_root, make_medium_name_frame,
                                 subject_number, verbose, must_redo, copy_root)


def copy_for_gan(dataset_root, dataset_csv, output_root, categories, verbose=True):
    train_frame = get_dataframe_by_usage_type(os.path.join(dataset_root, dataset_csv), "train")
    for cat in categories:
        copy_attack_categories_to_folder(train_frame, [cat], dataset_root, output_root, verbose=verbose)