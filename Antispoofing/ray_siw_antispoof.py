import argparse

from Antispoofing.AntispoofHelpers.antispoof_helper import antispoof_setup
from Antispoofing.AntispoofHelpers.hyper_perameter_helper import  AUG_PERCENTAGES

from Antispoofing.SIWAntispoof.siw_antispoof_helper import get_siw_train_frame_func, \
    get_siw_protocol_frame_dic, get_siw_stratified_name_col, process_siw_dataset_metrics_func, \
    get_siw_stratified_name_list_func

ATTACK_TYPE_COMBINATIONS = [
     'A@ASUS-IP7P-IPP2017,A@N',
     'A@ASUS-IP7P-SGS8,A@N',
     'A@ASUS-IPP2017-SGS8,A@N',
     'A@IP7P-IPP2017-SGS8,A@N',
     'A@P,A@N',
     'A@R,A@N',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--ncpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--whensplit", type=int, default=0)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--hsv", type=int, default=0)
    parser.add_argument("--err", type=int, default=0)
    cmd_args = parser.parse_args()
    error_only = cmd_args.err == 1
    use_hsv = cmd_args.hsv == 1
    mode = cmd_args.mode
    # mode = 0
    if cmd_args.whensplit == 0: # 0: aug before split
        aug_after_split = False
    else:                       #  1: aug after split
        aug_after_split = True

    train_subject_number = 90
    test_subject_number = 75
    save_path = None  # "/home/jarred/Documents/TEMP"
    is_single_folder = True

    tune_experiment_name = None#"SIW_antispoof_07_16_2022_10_58_02"
    stratified_name_func = get_siw_stratified_name_list_func

    is_gen_with_trad_aug=False
    is_baseline = False
    is_traditional = False
    use_last_only = False
    must_remove_normal = False
    must_use_normal_only = False

    mode_info = ""
    if mode == 0: # baseline
        is_baseline = True
        mode_info = "_Baseline"
    if mode == 1: # traditional aug with spoof only
        is_baseline = False
        is_traditional = True
        use_last_only = False
        must_remove_normal = True
        mode_info = "_TradAugSpoofOnly"
    if mode == 2: # traditional aug with normal only
        is_baseline = False
        is_traditional = True
        use_last_only = False
        must_use_normal_only = True
        must_remove_normal = False
        mode_info = "_TradAugNormalOnly"
    if mode == 3: # traditional aug with both
        is_baseline = False
        is_traditional = True
        use_last_only = False
        must_remove_normal = False
        mode_info = "_TradAugNormalAndSpoof"
    if mode == 4: # generated spoof only
        is_baseline = False
        is_traditional = False
        use_last_only = False
        must_remove_normal = True
        mode_info = "_GenAugSpoofOnly"
    if mode == 5: # generated with Normal only
        is_baseline = False
        is_traditional = False
        use_last_only = False
        must_use_normal_only = True
        must_remove_normal = False
        mode_info = "_GenAugNormalOnly"

    if mode == 6: # generated with Normal and Spoof
        is_baseline = False
        is_traditional = False
        use_last_only = False
        must_remove_normal = False
        mode_info = "_GenAugNormalAndSpoof"

    if mode == 7:  # generated with traditional aug spoof only
        is_baseline = False
        is_traditional = False
        use_last_only = False
        must_remove_normal = True
        is_gen_with_trad_aug=True
        mode_info = "_GenAugWithTradSpoofOnly"
    if mode == 8:  # generated with traditional aug Normal only
        is_baseline = False
        is_traditional = False
        use_last_only = False
        must_use_normal_only = True
        is_gen_with_trad_aug=True
        mode_info = "_GenAugWithTradNormalOnly"
    if mode == 9:  # generated with traditional aug Normal and Spoof
        is_baseline = False
        is_traditional = False
        use_last_only = False
        is_gen_with_trad_aug=True
        mode_info = "_GenAugWithTradNormalAndSpoof"

    print(f"Antispoofing: {mode_info}")
    dataset_name = "SIW"
    dataset_csv_name = "siw.csv"
    tune_cpu = cmd_args.ncpu
    tune_gpu= 1.0/cmd_args.ngpu
    epochs = cmd_args.epochs
    num_run_repeats = cmd_args.repeat
    if is_baseline:
        aug_percentages = None
    else:
        aug_percentages = AUG_PERCENTAGES

    print("HSV: ", use_hsv)

    antispoof_setup(dataset_name, dataset_csv_name, get_siw_train_frame_func, get_siw_protocol_frame_dic,
                    get_siw_stratified_name_col, process_siw_dataset_metrics_func,
                    ATTACK_TYPE_COMBINATIONS, tune_cpu, tune_gpu, epochs, train_subject_number,
                    test_subject_number, tune_experiment_name, aug_percentages, num_run_repeats, is_ray=True,
                    stratified_name_list_func=stratified_name_func, is_traditional=is_traditional,
                    use_last_only=use_last_only, save_path=save_path, is_single_folder=is_single_folder,
                    aug_after_split=aug_after_split, must_remove_normal=must_remove_normal,
                    is_gen_with_trad_aug=is_gen_with_trad_aug, mode_info=mode_info,
                    must_use_normal_only=must_use_normal_only, use_hsv=use_hsv,  error_only=error_only)

