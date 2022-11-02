from GAN.GANHelpers.copy_best_gan_helper import process_gan_results
from DatasetProcessing.DatasetCreators.CASIACreator.casia_helper import convert_attack_category_to_groundtruth
from constants import PROJECT_ROOT, TRAIN_GAN_FOLDER_ROOT

if __name__ == "__main__":
    dataset_name = "CASIA_KF"
    source_root = TRAIN_GAN_FOLDER_ROOT
    project_root = PROJECT_ROOT
    folder_vs_position = {'W1': 1,
                          'HR_W': 4,
                          'HR_R': 6,
                          'R1': 1,
                          'R2': 1,
                          'W2': 1,
                          'C2': 1,
                          }
    order_list = ['W1', 'W2', "HR_W", "C1", "C2", "HR_C", "R1", 'R2', "HR_R"]
    process_gan_results(project_root, dataset_name, source_root, convert_attack_category_to_groundtruth,folder_vs_position=folder_vs_position, order_list=order_list)