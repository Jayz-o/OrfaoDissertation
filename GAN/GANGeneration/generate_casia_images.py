from GAN.GANHelpers.gan_generate_helper import  generate_casia_images
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    dataset_root = "/home/jarred/Documents/Datasets/CASIA_KF"
    # dataset_root = "/home/jarred/Documents/Datasets/CASIA"
    dataset_csv_root = "/home/jarred/Documents/Datasets/CASIA"

    dataset_csv = "casia.csv"
    is_ray = True
    # ignore_detection =True
    ignore_detection =False


    generate_casia_images(dataset_root, dataset_csv,
                                    dataset_csv_root=dataset_csv_root, is_ray=is_ray, ignore_detection=ignore_detection)