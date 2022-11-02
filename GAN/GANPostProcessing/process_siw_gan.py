from GAN.GANHelpers.copy_best_gan_helper import process_gan_results
from DatasetProcessing.DatasetCreators.SIWCreator.siw_helper import convert_attack_category_to_groundtruth
from constants import PROJECT_ROOT, TRAIN_GAN_FOLDER_ROOT

if __name__ == "__main__":
    dataset_name = "SIW_90"
    # dataset_name = "SIW"
    source_root = TRAIN_GAN_FOLDER_ROOT
    project_root = PROJECT_ROOT
    order_list = ["ASUS", "IP7P", "IPP2017", "SGS8", "P"]
    process_gan_results(project_root, dataset_name, source_root, convert_attack_category_to_groundtruth, order_list=order_list)