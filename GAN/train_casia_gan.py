from GAN.GANHelpers.train_gan_helper import train_gan
from constants import PROJECT_ROOT

ALL_CATEGORIES = ['HR_N', 'HR_W', 'HR_C', 'HR_R', 'N1', 'N2', 'R1', 'R2', 'W1', 'W2', 'C1', 'C2']

if __name__ == "__main__":
    project_root = PROJECT_ROOT
    dataset_name = "CASIA"
    duration_kimg = 5000
    training_folder_name = ALL_CATEGORIES[0]
    num_gpus = 1
    train_gan(project_root, dataset_name, training_folder_name, duration_kimg, num_gpus)