import json
import os
import shutil
import subprocess

import pandas as pd
from matplotlib import pyplot as plt

from Helpers.image_helper import obtain_file_paths, create_image_grid_from_paths
from constants import BEST_GAN_FOLDER_NAME


def copy_best_to_folder(source_root, destination_root, convert_category_to_gt_func, best_position=None):
    print(f"Processing: {source_root}")
    # go into the folder and find any training folders
    metric_file_name = "metric-fid50k_full.jsonl"
    metrics = list(set(obtain_file_paths(source_root, metric_file_name)))
    metrics.sort()
    best_dic = dict()
    frames = []
    last_kimg = 0
    for metric in metrics:
        folder_name = os.path.basename(os.path.dirname(metric))
        with open(metric, 'r') as f:
            lines = f.readlines()
            results = []
            for line in lines:
                if len(line)<=1:
                    continue
                result = json.loads(line)
                kimg = int(os.path.splitext(result['snapshot_pkl'])[0].split("-")[-1])

                results.append({'fid': result["results"]["fid50k_full"], "cumulative_kimg":last_kimg + kimg, "model": result["snapshot_pkl"],
                                'fake_image': f"fakes{kimg:06d}.png", 'directory': folder_name, 'kimg': kimg,  'fake_image_path': f"{folder_name}/fakes{kimg:06d}.png"})
            last_kimg += kimg
            df = pd.DataFrame.from_dict(results)
            df.to_csv("info.csv", index=False)
            frames.append(df)
    combined_df = pd.concat(frames)
    # draw graph
    graph_df = combined_df.sort_values('cumulative_kimg')
    x_values = graph_df["cumulative_kimg"].tolist()
    y_values = graph_df['fid'].tolist()
    with  plt.ioff():
        fig = plt.figure()
        plt.plot(x_values, y_values, marker=".")
        temp_title = os.path.basename(source_root)
        if "HR_" in temp_title:
            bits = temp_title.split("_")
            temp_title = f"{bits[1]}_{bits[0]}"

        plt.ylabel("FID")
        plt.xlabel("Kimg")

        fig.savefig(os.path.join(source_root, "kimg_vs_fid.png"), dpi=300)


    combined_df=  combined_df.sort_values('fid')
    combined_df.to_csv(os.path.join(source_root, "training_info.csv"), index=False)
    best_list = list(combined_df.to_dict(orient='records'))
    if len(best_list) > 0:
        if best_position is not None:
            best_info = best_list[best_position]
        else:
            best_info = best_list[0]

        source_folder = os.path.join(source_root, best_info['directory'])

        if not os.path.exists(destination_root):
            os.makedirs(destination_root)
        fakes_name = best_info['fake_image']
        save_real_source =os.path.join(source_folder, "reals.png")
        save_real_destination =os.path.join(destination_root, "reals.png")
        save_fakes_source = os.path.join(source_folder, fakes_name)
        save_fakes_destination = os.path.join(destination_root, fakes_name)
        save_metrics_source = os.path.join(source_folder, metric_file_name)
        save_metrics_destination = os.path.join(destination_root, metric_file_name)
        save_model_source = os.path.join(source_folder, best_info['model'])
        save_model_name = "best_model.pkl"
        save_model_destination = os.path.join(destination_root, save_model_name)
        shutil.copy(save_fakes_source, save_fakes_destination)
        shutil.copy(save_model_source, save_model_destination)
        shutil.copy(save_metrics_source, save_metrics_destination)
        shutil.copy(save_real_source, save_real_destination)
        attack_category = os.path.basename(destination_root)
        ground_truth = convert_category_to_gt_func(attack_category)
        df_dic = {"fid": best_info['fid'], 'cumulative_kimg': best_info['cumulative_kimg'], "attack_category": attack_category,
                  "model_path": os.path.join(attack_category, save_model_name), 'ground_truth': ground_truth}
        df = pd.DataFrame.from_dict([df_dic])
        df.to_csv(os.path.join(destination_root, "info.csv"), index=False)

        folder_name = os.path.basename(source_root)
        cmd = f"convert {save_real_destination} {destination_root}/fakes*.png {destination_root}/output.pdf && gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/default  -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages -dCompressFonts=true -r150 -sOutputFile={destination_root}/{folder_name}_Real_Spoof.pdf {destination_root}/output.pdf && rm -f {destination_root}/output.pdf"
        os.system(cmd)
        return temp_title
    else:
        print(f"Error: Could not find metrics in {source_root}")

def process_gan_results(project_root, dataset_name,source_root, convert_category_to_gt_func, save_folder_name=BEST_GAN_FOLDER_NAME, folder_vs_position=None, order_list=None):
    # documents
    destination_root = os.path.dirname(project_root)
    print(f"Project root: {project_root}")
    if dataset_name not in os.path.basename(source_root):
        source_root = os.path.join(source_root, dataset_name)
    save_path = os.path.join(destination_root, save_folder_name, dataset_name)

    # see if folders already exist
    model_folders = next(os.walk(source_root))[1]
    for model_folder in model_folders:
        best_position = None
        if folder_vs_position is not None:
            if model_folder in folder_vs_position:
                best_position = folder_vs_position[model_folder]
        if not os.path.exists(os.path.join(save_path, model_folder)):
            copy_best_to_folder(os.path.join(source_root, model_folder), os.path.join(save_path, model_folder),
                            convert_category_to_gt_func, best_position)
    csv_files = obtain_file_paths(save_path, "^info.csv")
    all_files = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_files.append(df)
    if len(all_files) > 0:
        combined = pd.concat(all_files)
        combined.to_csv(os.path.join(save_path, f"{dataset_name}.csv"), index=False)
        fid_graphs = obtain_file_paths(source_root, r"^kimg_vs_fid.png")
        labels = None
        if order_list is not None:
            labels = []
            temp_list = []
            for item in order_list:
                for file in fid_graphs:
                    if item in file:
                        temp_list.append(file)
                        if "HR_" in item:
                            bits = item.split("_")
                            item = f"{bits[1]}_{bits[0]}"
                        labels.append(item)
                        break

            fid_graphs = temp_list

        create_image_grid_from_paths(fid_graphs,
                                     file_name=os.path.join(save_path, f"spoof_medium_fid_trends.pdf"),
                                     title=None, dpi=1000, scale=2, desired_image_shape=1000,
                                     class_names=labels)
    else:
        print("No csv files to combine")






