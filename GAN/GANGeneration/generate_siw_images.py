from GAN.GANHelpers.gan_generate_helper import generate_siw_images_for_subject

if __name__ == '__main__':
    dataset_root = "/home/jarred/Documents/Datasets/SIW"
    # dataset_root = "/home/jarred/Documents/Datasets/SIW_KF"
    dataset_csv_root = "/home/jarred/Documents/Datasets/SIW"
    dataset_csv = "siw.csv"
    # subject_number = None
    subject_number = "90"
    is_ray = True
    ignore_detection = False
    generate_siw_images_for_subject(dataset_root, dataset_csv, subject_number=subject_number,
                                    dataset_csv_root=dataset_csv_root, is_ray=is_ray, ignore_detection=ignore_detection)