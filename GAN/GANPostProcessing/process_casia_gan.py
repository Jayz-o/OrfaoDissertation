from GAN.GANHelpers.copy_best_gan_helper import process_gan_results
from DatasetProcessing.DatasetCreators.CASIACreator.casia_helper import convert_attack_category_to_groundtruth
from constants import PROJECT_ROOT, TRAIN_GAN_FOLDER_ROOT

if __name__ == "__main__":
    dataset_name = "CASIA"
    source_root = TRAIN_GAN_FOLDER_ROOT
    project_root = PROJECT_ROOT

    order_list = ['W1', 'W2', "HR_W","C1", "C2", "HR_C", "R1", 'R2', "HR_R"]
    process_gan_results(project_root, dataset_name, source_root, convert_attack_category_to_groundtruth, order_list=order_list)