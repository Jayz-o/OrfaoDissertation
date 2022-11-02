from GAN.GANHelpers.traditional_generate_helper import generate_casia_images
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    dataset_root = "/home/jarred/Documents/Datasets/CASIA"
    dataset_csv = "casia.csv"
    subject_number = None
    # subject_number = "90"
    generate_casia_images(dataset_root, dataset_csv)