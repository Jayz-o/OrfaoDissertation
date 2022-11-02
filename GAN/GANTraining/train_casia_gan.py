from GAN.GANHelpers.train_gan_helper_no_ray import ray_train_gan
from constants import PROJECT_ROOT

CASIA_FOLDERS = [
    "R1","W1","C1",
    "R2","W2","C2",
   "HR_R","HR_W","HR_C", "N1", "N2", "HR_N"

]
if __name__ == "__main__":
    dataset_name = "CASIA"
    project_root = PROJECT_ROOT
    duration_kimg = 2000
    tune_cpu = 10,
    tune_gpu = 4.0,
    num_stylegan_gpus = 4
    ray_train_gan(dataset_name, project_root, CASIA_FOLDERS, duration_kimg, num_stylegan_gpus, tune_cpu, tune_gpu, is_ray=False)
