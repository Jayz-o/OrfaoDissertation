from Antispoofing.CASIAAntispoof.casia_antispoof_helper import ALL_CATEGORIES_LIST, ALL_CATEGORIES
from GAN.GANHelpers.gan_dataset_helper import copy_for_gan


if __name__ == "__main__":
    dataset_root = "/home/jarred/Documents/Datasets/CASIA"
    dataset_csv = "casia.csv"
    output_root = "/home/jarred/Documents/Datasets/GAN"
    copy_for_gan(dataset_root, dataset_csv, output_root, ALL_CATEGORIES,verbose=True)
