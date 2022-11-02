import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune
from collections import Counter
# from imblearn.over_sampling import RandomOverSampler
from Antispoofing.AntispoofHelpers.antispoof_model_helper import create_vit_b32
from Antispoofing.AntispoofHelpers.dataset_helper import get_random_selection_on_aug_category, get_antispoof_frame, \
    get_train_validation_generator, get_test_generator, Y_COL, Z_COL, X_COL, test_split
from Antispoofing.AntispoofHelpers.spoof_metric import determine_spoof_metrics, PROTOCOL_COL
import os
os.environ['TUNE_RESULT_DELIM'] = '/'
AUG_PERCENTAGES = [0.05,0.1,0.2, 0.30]

def initialise_tf():
    import tensorflow as tf
    try:
        # fix memory issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

def combine_with_augmentation(train_frame, aug_frame, aug_root, categories, aug_percentage, stratified_name_list_func=None, use_last_only=False, must_remove_normal=True, must_use_normal_only=False):    # determine how many frames are in the dataset
    total_frames = train_frame.shape[0]
    # (Itot / aug %) / (1 - aug %)
    temp = []
    for cat in categories:
        if "-" in cat:
            temp2 = cat.split('-')
            for t in temp2:
                temp.append(t)
        else:
            temp.append(cat)
    categories = temp


    tempcategories = [] #['ASUS', 'IP7P', 'IPP2017', 'SGS8']
    for cat in categories:
        if cat == "R":
            tempcategories.append('ASUS')
            tempcategories.append('IP7P')
            tempcategories.append('IPP2017')
            tempcategories.append('SGS8')
        elif "N" in cat:
            if not must_remove_normal:
                tempcategories.append(cat)
        else:
            tempcategories.append(cat)

    categories = tempcategories

    if must_use_normal_only:
        tempcategories = []
        for cat in categories:
            if "N" in cat:
                tempcategories.append(cat)
        categories = tempcategories

    if use_last_only:
        num_categories = 1
    else:
        num_categories = len(categories)
    num_augmentation_files = round(((total_frames * aug_percentage)/ (1 - aug_percentage))/num_categories)
    # file_path, ground_truth df
    # to augment with only N
    random_aug_frame = get_random_selection_on_aug_category(aug_frame, categories, aug_root, num_augmentation_files, seed=None, stratified_name_list_func=stratified_name_list_func, use_last_only=use_last_only)
    return random_aug_frame

def video_based_results(single_frame, protocol_name, fold_index,protocol_number, fold_save_metrics_root, save_metric_name):
    temp_single = single_frame.copy()
    def categorise_video(row):
        return os.path.basename(os.path.dirname(row['file_paths']))
    temp_single['video_name'] = temp_single.apply(lambda row: categorise_video(row), axis=1)
    video_names = temp_single['video_name'].unique()
    video_list = []
    for name in video_names:
        temp_df = temp_single.query(f"video_name == '{name}'")
        real = 0
        spoof = 1
        spoof_pred_count = temp_df[(temp_df.predicted == 1)].count()["predicted"]
        real_pred_count = temp_df[(temp_df.predicted == 0)].count()["predicted"]
        if spoof_pred_count > real_pred_count:
            predicted = 1
        else:
            predicted = 0
        ground_truth = temp_df['ground_truth'].tolist()[0]
        video_list.append({"video_name": name, f'spoof({spoof})_pred_count': spoof_pred_count, f'real({real}_pred_count': real_pred_count, "predicted": predicted, "ground_truth": ground_truth})
    multi_frame = pd.DataFrame.from_dict(video_list)
    multi_frame.to_csv(os.path.join(fold_save_metrics_root, f"test_{protocol_name}_multi_frame_results.csv"), index=False)
    predicted = multi_frame['predicted'].tolist()
    ground_truth = multi_frame['ground_truth'].tolist()
    metric_dic = determine_spoof_metrics(ground_truth, predicted, protocol_name, fold_index,protocol_number, save_dir=os.path.join(fold_save_metrics_root, f"{save_metric_name}_{protocol_name}_Multi_Metrics"), must_show=False)
    metric_dic = dict(("{}_{}".format("Multi",k),v) for k,v in metric_dic.items())
    return metric_dic


def antispoof(config):
    use_hsv = config['use_hsv']
    must_remove_normal = config['must_remove_normal']
    aug_after_split = config['aug_after_split']
    must_use_normal_only = config['must_use_normal_only']
    initialise_tf()
    import tensorflow as tf
    tf.keras.backend.set_image_data_format('channels_last')
    include_traditional_aug = config['include_traditional_aug']
    stratified_name_list_func=config['stratified_name_list_func']
    n_folds = config['n_folds']
    current_fold = config['current_fold']
    dataset_root = config['dataset_root']
    original_dataset_root = config['original_dataset_root']
    train_subject_number = config["train_subject_number"]
    test_subject_number = config["test_subject_number"]
    get_train_frame_func = config["get_train_frame_func"]
    get_stratified_name_col_func = config["get_stratified_name_col_func"]
    get_protocol_frame_dic_func = config["get_protocol_frame_dic_func"]
    process_dataset_metrics_func = config["process_dataset_metrics_func"]
    repeat_number = config["HP_REPEAT"]
    # get the config variables
    attack_type_combination = config['HP_COMB']
    aug_percentage = config['HP_AUG_PER']
    use_last_only = config['use_last_only']

    run_folder = f"{attack_type_combination}_aug_{aug_percentage}_run_{repeat_number}"
    epochs = config['epochs']
    save_metrics_root = os.path.join(config['save_metrics_root'], run_folder)
    save_checkpoints_root = os.path.join(config['save_checkpoints_root'], run_folder)
    save_tb_root = os.path.join(config['save_tb_root'], run_folder)

    experiment_dirs = [save_metrics_root, save_checkpoints_root, save_tb_root]
    # create the directories
    for _dir in experiment_dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    dataset_name = config['dataset_name']
    dataset_csv_name = config['dataset_csv_name']
    aug_root = config['aug_root']
    aug_csv = config['aug_csv']

    combinations = []
    # split the attack type combination
    if "," in attack_type_combination:
        attack_type_combination = attack_type_combination.split(",")
        for comb in attack_type_combination:
            combinations.append(comb.split("@")[1])
    elif "-" in attack_type_combination:
        combinations.append(attack_type_combination.split("@")[1])

    else:
        combinations.append(attack_type_combination.split("@")[1])
    # get the train dataset frame
    train_frame = get_train_frame_func(dataset_root, dataset_csv_name, combinations, train_subject_number)

    stratified_name = get_stratified_name_col_func(combinations)
    train_frame = get_antispoof_frame(train_frame, dataset_root, stratified_name=stratified_name)



    sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)


    if aug_after_split:
        if Z_COL in train_frame.columns:
            splits = sss.split(train_frame[X_COL], train_frame[Z_COL])
        else:
            splits = sss.split(train_frame[X_COL], train_frame[Y_COL])

        for i in range(n_folds):
            train_index, val_index = next(splits)
            if i == current_fold:
                break



        fold_index = str(current_fold)
        fold_save_metrics_root = os.path.join(save_metrics_root, fold_index)
        fold_save_checkpoints_root = os.path.join(save_checkpoints_root, fold_index)
        fold_save_tb_root = os.path.join(save_tb_root, fold_index)

        experiment_dirs = [fold_save_metrics_root, fold_save_checkpoints_root, fold_save_tb_root]
        # create the directories
        for _dir in experiment_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        fold_train_frame = train_frame.iloc[train_index]
        fold_val_frame = train_frame.iloc[val_index]

        # add the augmentation files to the train frame
        if aug_percentage > 0:
            aug_frame = pd.read_csv(os.path.join(aug_root, aug_csv))
            aug_frame = combine_with_augmentation(fold_train_frame, aug_frame, aug_root, combinations, aug_percentage,
                                                  stratified_name_list_func, use_last_only, must_remove_normal, must_use_normal_only)
            fold_train_frame = pd.concat([fold_train_frame, aug_frame])
    else:
        # add the augmentation files to the train frame
        if aug_percentage > 0:
            aug_frame = pd.read_csv(os.path.join(aug_root, aug_csv))
            aug_frame = combine_with_augmentation(train_frame, aug_frame, aug_root, combinations, aug_percentage,
                                                  stratified_name_list_func, use_last_only, must_remove_normal,must_use_normal_only)
            train_frame = pd.concat([train_frame, aug_frame])

        if Z_COL in train_frame.columns:
            splits = sss.split(train_frame[X_COL], train_frame[Z_COL])
        else:
            splits = sss.split(train_frame[X_COL], train_frame[Y_COL])

        for i in range(n_folds):
            train_index, val_index = next(splits)
            if i == current_fold:
                break

        fold_index = str(current_fold)
        fold_save_metrics_root = os.path.join(save_metrics_root, fold_index)
        fold_save_checkpoints_root = os.path.join(save_checkpoints_root, fold_index)
        fold_save_tb_root = os.path.join(save_tb_root, fold_index)

        experiment_dirs = [fold_save_metrics_root, fold_save_checkpoints_root, fold_save_tb_root]
        # create the directories
        for _dir in experiment_dirs:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        fold_train_frame = train_frame.iloc[train_index]
        fold_val_frame = train_frame.iloc[val_index]



    if save_metrics_root is not None:
        fold_train_frame.to_csv(f"{save_metrics_root}/fold_{fold_index}_train_frame.csv", index=False)
        fold_val_frame.to_csv(f"{save_metrics_root}/fold_{fold_index}_val_frame.csv", index=False)
    test_split(fold_train_frame, fold_val_frame, X_COL)
    # balance classes
    # res_x, res_y = train_frame.iloc[:, [0,2], train_frame.iloc[:, 1]
    # oversample = RandomOverSampler(random_state=None)
    # res_x, res_y = oversample.fit_resample(res_x, res_y)
    # print(Counter(res_y))
    # visualise
    # counter = Counter(train_frame["info"])
    # plt.bar(counter.keys(), counter.values())
    # plt.show()
    # sns.countplot(temp_frame["info"])
    # plt.show()
    train_generator, valid_generator = get_train_validation_generator(fold_train_frame, fold_val_frame, use_hsv=use_hsv)

    # create the model
    model = create_vit_b32(include_traditional=include_traditional_aug)
    learning_rate = 1e-4
    weight_decay = 1e-5

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)#, beta_2=weight_decay)

    model.compile(optimizer=optimiser, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
    #                                                  factor = 0.2,
    #                                                  patience = 2,
    #                                                  verbose = 1,
    #                                                  min_delta = 1e-4,
    #                                                  min_lr = 1e-6,
    #                                                  mode = 'max')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=1e-4,
                                                     patience=15,
                                                     mode='min',
                                                     restore_best_weights=True,
                                                     verbose=1)
    checkpoint_path = os.path.join(fold_save_checkpoints_root, "best.ckpt")
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fold_save_tb_root, histogram_freq=1, update_freq='epoch')
    callbacks = [earlystopping, checkpointer, tensorboard_callback]  # ,reduce_lr ]

    train_step_size = train_generator.n // train_generator.batch_size
    validation_step_size = valid_generator.n // valid_generator.batch_size
    # STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    history = model.fit(x=train_generator,
                        steps_per_epoch=train_step_size,
                        validation_data=valid_generator,
                        validation_steps=validation_step_size,
                        epochs=epochs,
                        callbacks=callbacks)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))
    with plt.ioff():
        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(fold_save_metrics_root, f"fold_{fold_index}_train_history.png"))

    print("restoring top model")
    # load best model
    latest = tf.train.latest_checkpoint(fold_save_checkpoints_root)
    print(latest)
    model = create_vit_b32()
    model.load_weights(latest)

    save_metric_name = ""
    lookup_dataset_root =""
    if original_dataset_root is None:
        protocol_frame_dic = get_protocol_frame_dic_func(dataset_root, dataset_csv_name, combinations, test_subject_number)
        lookup_dataset_root = dataset_root
    else:
        protocol_frame_dic = get_protocol_frame_dic_func(original_dataset_root, dataset_csv_name, combinations, test_subject_number)
        lookup_dataset_root = original_dataset_root
    if test_subject_number is not None:
        save_metric_name = f"S{test_subject_number}"
    for protocol_name, protocol_number_frame in protocol_frame_dic.items():
        protocol_number = protocol_number_frame["protocol_number"]
        protocol_frame = protocol_number_frame["frame"]
        test_frame = get_antispoof_frame(protocol_frame, lookup_dataset_root)
        # test if there is bias
        test_split(fold_train_frame, test_frame, X_COL)

        test_generator = get_test_generator(test_frame, use_hsv=use_hsv)
        test_step_size = test_generator.n // test_generator.batch_size
        predicted = np.argmax(model.predict(test_generator, test_step_size, verbose=1), axis=1)
        ground_truth = test_generator.classes
        temp_dic = {"file_paths": test_generator.filepaths, "ground_truth": ground_truth, "predicted": list(predicted)}
        df = pd.DataFrame.from_dict(temp_dic, orient='index').transpose()
        df.to_csv(os.path.join(fold_save_metrics_root, f"test_{protocol_name}_results.csv"), index=False)
        metric_dic = determine_spoof_metrics(ground_truth, predicted, protocol_name, fold_index,protocol_number, save_dir=os.path.join(fold_save_metrics_root, f"{save_metric_name}_{protocol_name}_Metrics"), must_show=False)
        multi_metric_dic = video_based_results(df, protocol_name, fold_index,protocol_number, fold_save_metrics_root, save_metric_name)
        metric_dic.update(multi_metric_dic)
        if config['is_ray']:
            tune.report(**metric_dic)


def start_antispoofing(dataset_root, dataset_csv_name, aug_root, aug_csv, save_metrics_root, save_checkpoints_root,
                       save_tb_root, save_tune_root, aug_folder_combinations, tune_gpu, tune_cpu, epochs,
                       get_train_frame_func, get_protocol_frame_dic_func, get_stratified_name_col_func, process_dataset_metrics_func,tune_experiment_name=None,
                       aug_percentages=None, repeat_run_list=None, train_subject_number=None,
                       test_subject_number=None, must_resume_from_last_experiment=True, is_ray=True, n_k_folds=3,
                       original_dataset_root=None, stratified_name_list_func=None, is_traditional=False,
                       use_last_only=False,is_single_folder=True, include_traditional_aug=False, aug_after_split=False
                       , must_remove_normal=False, mode_info="", must_use_normal_only=False, error_only=False, use_hsv=False):
    if aug_percentages is None and repeat_run_list is None:
        raise TypeError("Please specify either the aug_percentages or repeat_run_list")
        return

        # get the dataset name
    if ".csv" not in dataset_csv_name:
        dataset_csv_name += ".csv"

    dataset_name = os.path.basename(dataset_root)


    # test if the dataset creator csv file is present in the dataset root
    dataset_csv_location = os.path.join(dataset_root, dataset_csv_name)
    aug_csv_location = os.path.join(aug_root, aug_csv)
    if not os.path.exists(dataset_csv_location):
        raise TypeError(f"Could not find the dataset csv file: {dataset_csv_location}")

    if not os.path.exists(aug_csv_location) and aug_percentages is not None:
        raise TypeError(f"Could not find the aug csv file: {aug_csv_location}")

    tune_antispoof_csv = f"{dataset_name}_antispoof_tune.csv"

    # if is_traditional:
    #     training_type = "TraditionalAugmentation"
    # else:
    #     # training_type = "GeneratedTraditionalAugmentation"
    #     training_type = "GeneratedAugmentation"
    training_type = ""
    if is_single_folder:
        training_type += "Single"
    else:
        training_type += "Multi"


    save_tune_root = os.path.join(save_tune_root, training_type)
    if tune_experiment_name is None:
        if must_resume_from_last_experiment:
            existing_dirs = []
            if os.path.exists(save_tune_root):
                existing_dirs = os.listdir(save_tune_root)

            if len(existing_dirs) > 0:
                #     run from the last directory
                existing_dirs.sort(reverse=True)
                tune_experiment_name = os.path.basename(existing_dirs[0])

    if tune_experiment_name is None:
        tune_experiment_name = f"{dataset_name}_antispoof_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    save_metrics_root = os.path.join(save_metrics_root,training_type, tune_experiment_name)
    save_checkpoints_root = os.path.join(save_checkpoints_root,training_type, tune_experiment_name)
    save_tb_root = os.path.join(save_tb_root, training_type, tune_experiment_name)



    experiment_dirs = [save_metrics_root, save_checkpoints_root, save_tb_root, save_tune_root]
    # create the directories
    for _dir in experiment_dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    # return
    k_fold_list = [i for i in range(n_k_folds)]
    tune_config = {
        "HP_COMB": tune.grid_search(aug_folder_combinations),
        # "HP_COMB": aug_folder_combinations[0],
        'must_remove_normal' : must_remove_normal,
        "epochs": epochs,
        "save_metrics_root": save_metrics_root,
        "save_checkpoints_root": save_checkpoints_root,
        "save_tb_root": save_tb_root,
        "dataset_root": dataset_root,
        "dataset_name": dataset_name,
        "dataset_csv_name": dataset_csv_name,
        "aug_root": aug_root,
        "aug_csv": aug_csv,
        "train_subject_number" : train_subject_number,
        "test_subject_number" : test_subject_number,
        "get_train_frame_func": get_train_frame_func,
        "get_protocol_frame_dic_func": get_protocol_frame_dic_func,
        'is_ray': is_ray,
        'get_stratified_name_col_func': get_stratified_name_col_func,
        'process_dataset_metrics_func': process_dataset_metrics_func,
        'stratified_name_list_func': stratified_name_list_func,
        'n_folds' : n_k_folds,
        'current_fold': tune.grid_search(k_fold_list),
        'HP_REPEAT': tune.grid_search(repeat_run_list),
        'original_dataset_root': original_dataset_root,
        'use_last_only': use_last_only,
        'include_traditional_aug':include_traditional_aug,
        'aug_after_split':aug_after_split,
        'mode_info': mode_info,
        'must_use_normal_only': must_use_normal_only,
        'use_hsv': use_hsv,
    }

    if aug_percentages is None:
        tune_config["HP_AUG_PER"] = tune.grid_search([0])
    else:
        tune_config["HP_AUG_PER"] = tune.grid_search(aug_percentages)

    if is_ray:
        if error_only:
            analysis = tune.run(antispoof, config=tune_config, local_dir=save_tune_root, name=tune_experiment_name,
                                resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="ERRORED_ONLY")
        else:
            analysis = tune.run(antispoof, config=tune_config, local_dir=save_tune_root, name=tune_experiment_name,
                                resources_per_trial={"cpu": tune_cpu, "gpu": tune_gpu}, resume="AUTO")
        df = analysis.results_df
        df.to_csv(os.path.join(save_tune_root,tune_experiment_name, tune_antispoof_csv))
        process_dataset_metrics_func(df, os.path.join(save_tune_root,tune_experiment_name))
    else:
        if aug_percentages is None:
            aug_per = 0
        else:
            aug_per = aug_percentages[0]
        for combination in aug_folder_combinations:
            antispoof({
                "HP_COMB": combination,
                'must_remove_normal':must_remove_normal,
                "epochs": epochs,
                "save_metrics_root": save_metrics_root,
                "save_checkpoints_root": save_checkpoints_root,
                "save_tb_root": save_tb_root,
                "dataset_root": dataset_root,
                "dataset_name": dataset_name,
                "dataset_csv_name": dataset_csv_name,
                "aug_root": aug_root,
                "aug_csv": aug_csv,
                "train_subject_number" : train_subject_number,
                "test_subject_number" : test_subject_number,
                "get_train_frame_func": get_train_frame_func,
                "get_protocol_frame_dic_func": get_protocol_frame_dic_func,
                "stratified_name_list_func": stratified_name_list_func,
                "HP_AUG_PER": aug_per,
                "HP_REPEAT": 1,
                'is_ray' : is_ray,
                  'get_stratified_name_col_func': get_stratified_name_col_func,
                'process_dataset_metrics_func': process_dataset_metrics_func,
                'n_folds' : n_k_folds,
                'current_fold' : k_fold_list[0],
                'original_dataset_root': original_dataset_root,
                'use_last_only': use_last_only,
                'include_traditional_aug': include_traditional_aug,
                'aug_after_split': aug_after_split,
                'mode_info':mode_info,
                'must_use_normal_only': must_use_normal_only,
                'use_hsv': use_hsv,

            })





