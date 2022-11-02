from GAN.GANHelpers.train_gan_helper_no_ray import ray_train_gan
from constants import PROJECT_ROOT

CASIA_FOLDERS = [
    "R1","W1","C1",
    "R2","W2","C2",
   "HR_R","HR_W","HR_C", "N1", "N2", "HR_N"
    # "N1-N2-HR_N-W1-W2-HR_W",
    # "N1-N2-HR_N-C1-C2-HR_C",
    # "N1-N2-HR_N-R1-R2-HR_R",
    # "HR_N-HR_W-HR_C-HR_R-N1-N2-R1-R2-W1-W2-C1-C2",
]
if __name__ == "__main__":
    dataset_name = "CASIA"
    project_root = PROJECT_ROOT
    duration_kimg = 2000
    tune_cpu = 10,
    tune_gpu = 4.0,
    num_stylegan_gpus = 4
    ray_train_gan(dataset_name, project_root, CASIA_FOLDERS, duration_kimg, num_stylegan_gpus, tune_cpu, tune_gpu, is_ray=False)
