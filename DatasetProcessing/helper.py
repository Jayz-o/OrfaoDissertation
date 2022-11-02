import os
import re
import shutil

import pandas as pd


def combine_csvs(csv_files, dataset_root, save_name):
    if len(csv_files) <= 0:
        with open(os.path.join(dataset_root, f"ERROR_NO_CSV_FILES"), "w") as file:
            message = f"Could not find any csv files to combine. location: {dataset_root}"
            file.write(message)
            print(message)
        return
    csvs = []
    for csv_path in csv_files:
        try:
            csvs.append(pd.read_csv(csv_path))
        except:
            print(csv_path)

    frame = pd.concat(csvs, axis=0, ignore_index=True)
    path = os.path.join(dataset_root, save_name)

    # remove any existing csv file
    if os.path.exists(path):
        os.remove(path)

    # save the csv file
    frame.to_csv(path, index=False)

def move_error_files(directory_containing_error_files):
    # move any previous error files to an error dir
    error_files = []
    current_dir, sub_dirs, files = next(os.walk(directory_containing_error_files))
    for file in files:
        if re.match(r"ERROR_", file):
            error_files.append(os.path.join(current_dir, file))
    if len(error_files) > 0:
        #   see if an error dir has been created
        error_dirs = []
        for dir in sub_dirs:
            if "ERROR_" in dir:
                error_dirs.append(dir)
        # if there are, we need to get the last dir created and increment the counter
        if len(error_dirs) > 0:
            error_dirs.sort()
            last_error_file = error_dirs[-1]
            error_bits = last_error_file.split("_")
            next_increment = int(error_bits[-1]) + 1
            move_error_path = os.path.join(directory_containing_error_files, f"ERROR_{next_increment}")
            del last_error_file
            del error_bits
            del next_increment
        else:
            # else, we create the first 1
            move_error_path = os.path.join(directory_containing_error_files, "ERROR_0")
        if not os.path.exists(move_error_path):
            os.makedirs(move_error_path)
        for ef in error_files:
            file_name = os.path.basename(ef)
            destination = os.path.join(move_error_path, file_name)
            shutil.move(ef, destination)
        print(f"~~~~~~~~~~~~~~~~~~ FOUND ERRORS: MOVING TO '{move_error_path}' ~~~~~~~~~~~~~~~~~~")
        del error_files
        del error_dirs
        del move_error_path
        del file_name
        del destination
    del current_dir
    del sub_dirs
    del files