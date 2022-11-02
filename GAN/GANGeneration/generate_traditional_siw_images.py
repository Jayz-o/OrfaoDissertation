from GAN.GANHelpers.traditional_generate_helper import generate_siw_images_for_subject
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == '__main__':
    dataset_root = "/home/jarred/Documents/Datasets/SIW"
    dataset_csv = "siw.csv"
    subject_number = None
    subject_number = "90"
    generate_siw_images_for_subject(dataset_root, dataset_csv, subject_number=subject_number)