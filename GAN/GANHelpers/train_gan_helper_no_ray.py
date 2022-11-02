import os.path
from datetime import datetime


from Helpers.image_helper import obtain_file_paths
from NVIDIA_STYLEGAN3.train import main
from constants import TRAIN_GAN_FOLDER_NAME

os.environ['TUNE_RESULT_DELIM'] = '/'


def train_gan(project_root, dataset_name, training_folder_name, duration_kimg, num_gpu=1):

    try:
        supported_gpus = [1, 2, 4, 8]
        if num_gpu not in supported_gpus:
            raise TypeError(f"Unsupported num gpu: {num_gpu}. Supported {supported_gpus}")
        # documents
        project_root_pred = os.path.dirname(project_root)
        save_path = os.path.join(project_root_pred, TRAIN_GAN_FOLDER_NAME, dataset_name, training_folder_name)
        temp_files = obtain_file_paths(save_path, f"network-snapshot-{duration_kimg:06d}.pkl")
        if len(temp_files)>0:
            print(f"Found network-snapshot-{duration_kimg:06d}.pkl indicating duration {duration_kimg} has been reached... returning")
            return

        dataset_root = os.path.join(project_root_pred, "Datasets/GAN", dataset_name)
        training_folder_zip = training_folder_name
        if ".zip" not in training_folder_zip:
            training_folder_zip = f"{training_folder_zip}.zip"

        data_path = os.path.join(dataset_root, training_folder_zip)
        if not os.path.exists(data_path):
            raise TypeError(f"Could not find training zip file: {data_path}")

        args = [f"--outdir={save_path}", f"--data={data_path}", "--mirror=1", "--cfg=stylegan3-r", "--batch=32",
                "--gamma=32"]
        if num_gpu == 1:
            args.append("--gpus=1")
            args.append("--batch-gpu=8")
            args.append("--snap=10")
        elif num_gpu == 2:
            args.append("--gpus=2")
            args.append("--batch-gpu=8")
            args.append("--snap=10")
        elif num_gpu == 4:
            args.append("--gpus=4")
            args.append("--batch-gpu=4")
            args.append("--snap=20")
        elif num_gpu == 8:
            args.append("--gpus=8")
        else:
            raise TypeError(f"Unsupported num gpu: {num_gpu}. Supported {supported_gpus}")
        # see if folders already exit
        existing_dirs = []
        if os.path.exists(save_path):
            existing_dirs = os.listdir(save_path)


        if len(existing_dirs) > 0:
            #     run from the last directory
            existing_dirs.sort(reverse=True)
            interested_files = None
            interested_dir = None
            kimg_remaining = duration_kimg
            latest_pickle = None
            for dir in existing_dirs:
                pickle_files = obtain_file_paths(os.path.join(save_path, dir), r"network-snapshot-\d{6}.pkl")
                if len(pickle_files) > 0:
                    pickle_files.sort(reverse=True)
                    if latest_pickle is None:
                        latest_pickle = pickle_files[0]
                    pickle_name = os.path.basename(pickle_files[0])
                    kimg_number = os.path.splitext(pickle_name)[0].split("-")[-1]
                    kimg_number = int(kimg_number)
                    kimg_remaining -= kimg_number
            if latest_pickle is not None:
                if kimg_remaining <= 0:
                    print(f"No kimg remaining: {latest_pickle}")
                else:
                    args.append(f"--kimg={kimg_remaining}")
                    args.append(f"--resume={latest_pickle}")
                    main(args)
            else:
                args.append(f"--kimg={duration_kimg}")
                main(args)
        else:
            args.append(f"--kimg={duration_kimg}")
            main(args)
    except Exception as e:
        print(e)
        pass
    except:
        print("Catch all exception")

def begin_training_gan(config):

    train_gan(config['project_root'], config['dataset_name'], config["training_folder_name"], config['duration_kimg'], config['num_gpus'])

def ray_train_gan(dataset_name, project_root, attack_folders, duration_kimg, num_stylegan_gpu,tune_cpu, tune_gpu, is_ray=True):
    # documents
    project_root_pred = os.path.dirname(project_root)
    tune_root = os.path.join(project_root_pred, "Tune_GAN_Training", dataset_name)
    tune_kf_csv = f"{dataset_name}_gan_training.csv"
    tune_experiment_name = f"{dataset_name}_gan_training_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    if is_ray:
        # tune_config = {
        #     "training_folder_name": tune.grid_search(attack_folders),
        #     "dataset_name": dataset_name,
        #     "duration_kimg": duration_kimg,
        #     "num_gpus": num_stylegan_gpu,
        #     "project_root": project_root
        # }
        #
        # analysis = tune.run(begin_training_gan, config=tune_config, local_dir=tune_root, name=tune_experiment_name,
        #                     resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
        #                     # resources_per_trial={"cpu": 2.0, "gpu": 1.0}, resume="AUTO")
        # df = analysis.results_df
        # df.to_csv(os.path.join(tune_root, tune_kf_csv))
        pass
    else:
        for folder in attack_folders:
            begin_training_gan( {
                "training_folder_name": folder,
                "dataset_name": dataset_name,
                "duration_kimg": duration_kimg,
                "num_gpus": num_stylegan_gpu,
                "project_root": project_root
            })

