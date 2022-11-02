import argparse

from GAN.GANHelpers.train_gan_helper import ray_train_gan
from constants import PROJECT_ROOT

SIW_FOLDERS = ['N']#'ASUS', 'SGS8', 'P', 'IP7P', 'IPP2017', 'N'] #[  'ASUS-IPP2017-SGS8', 'IP7P-IPP2017-SGS8', 'P', 'R', 'N']#'ASUS-IP7P-SGS8', 'ASUS-IP7P-IPP2017',

def train(folders, ngpu, kimg):
    dataset_name = "SIW_90"
    project_root = PROJECT_ROOT
    duration_kimg = kimg
    tune_cpu = 11,
    tune_gpu = 4.0,
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
        train_folders = SIW_FOLDERS
        print("Training with ALL FOLDERS")
    else:
        print("Training with Command Line args")

    train(train_folders, cmd_args.ngpu, cmd_args.kimg)