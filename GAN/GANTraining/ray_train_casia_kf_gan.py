from GAN.GANHelpers.train_gan_helper import ray_train_gan
from constants import PROJECT_ROOT
import argparse
CASIA_FOLDERS = [
    "R1","W1","C1",
    "R2","W2","C2",
   "HR_R","HR_W","HR_C", "N1", "N2", "HR_N"
    # "N1-N2-HR_N-W1-W2-HR_W",
    # "N1-N2-HR_N-C1-C2-HR_C",
    # "N1-N2-HR_N-R1-R2-HR_R",
    # "HR_N-HR_W-HR_C-HR_R-N1-N2-R1-R2-W1-W2-C1-C2",
]

def train(folders, ngpu, kimg):
    dataset_name = "CASIA_KF"
    project_root = PROJECT_ROOT
    duration_kimg = kimg
    tune_cpu = 10,
    tune_gpu = 1.0,
    num_stylegan_gpus = ngpu
    ray_train_gan(dataset_name, project_root, folders, duration_kimg, num_stylegan_gpus, tune_cpu, tune_gpu, is_ray=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", default=None)
    parser.add_argument("--ngpu", type=int, default=2)
    parser.add_argument("--kimg", type=int, default=2000)

    cmd_args = parser.parse_args()
    train_folders = cmd_args.folders
    if train_folders is None:
        train_folders = CASIA_FOLDERS
        print("Training with ALL FOLDERS")
    else:
        print("Training with Command Line args")
    train(train_folders, cmd_args.ngpu, cmd_args.kimg)