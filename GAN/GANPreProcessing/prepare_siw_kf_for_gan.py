from GAN.GANHelpers.siw_gan_dataset_helper import get_protocol_3_folders, get_protocol_2_folders, get_normal_folders

if __name__ == "__main__":
    dataset_root = "/home/jarred/Documents/Datasets/SIW_KF"
    dataset_csv = "siw.csv"
    output_root = "/home/jarred/Documents/Dissertation/GAN"
    # output_root = "/home/jarred/Documents/Datasets/GAN"
    subject_number = "75"
    verbose = True
    get_protocol_2_folders(dataset_root, dataset_csv, output_root, subject_number,verbose)
    get_protocol_3_folders(dataset_root, dataset_csv, output_root, subject_number,verbose)
    get_normal_folders(dataset_root, dataset_csv, output_root, subject_number,verbose)
