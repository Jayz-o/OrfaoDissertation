import numpy as np

import matplotlib.pyplot as plt

plt.style.use('dark_background')
# plt.style.use('default')

def make_tsne_plot():
    pass
    # get the dataset
    # processed_gen, unprocessed_gen = obtain_datagens(DATASET_ROOT, CSV_NAME, IS_TRAIN, TRANSITION_FOLDER, BATCH_SIZE,
    #                                                  IMAGE_SIZE, AUGMENTATION_PATH, AUGMENTATION_PERCENTAGE)
    # image_features_array, y_values = extract_features(embedding_model, processed_gen, SAVE_FOLDER)
    # for perplexity in [5, 30, 55, 80, 105, 130]:
    #     tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity, learning_rate=10)
    #     tsne_out = tsne.fit_transform(image_features_array)
    #     tsne_frame = pd.DataFrame(np.row_stack(tsne_out), columns=['x', 'y'])
    #     tsne_frame['label'] = y_values  # baseline diagnosis labels
    #     tsne_framesub = tsne_frame.copy()  # tsne_frame[tsne_frame.label.isin(['AD_Dementia','CN_CN','EMCI_MCI','MCI_MCI','MCI_Dementia'])]
    #     # tsne_framesub.label.cat.remove_unused_categories(inplace=True)
    #     sns.set_context("notebook", font_scale=1.1)
    #     sns.set_style("ticks")
    #     sns.lmplot(x='x', y='y',
    #                data=tsne_framesub,
    #                fit_reg=False, legend=True,
    #                height=8,
    #                hue='label',
    #                scatter_kws={"s": 50, "alpha": 0.2})


def plot():
    labels = [str(i) for i in range(1, 8)]

    kf_baseline = [97.73, 95.90, 97.21, 96.87, 96.57, 96.98, 96.66]
    ef_baseline = [97.73, 96.52, 98.05, 97.73, 96.97, 97.48, 97.42]
    kf_augmented = [98.86, 96.71, 98.06, 97.90, 97.93, 97.89, 97.55]
    ef_augmented = [99.50, 97.51, 98.82, 98.26, 98.01, 98.43, 98.02]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots(figsize=(13, 9), dpi=1200)

    rects1 = ax.barh(x + 0.3, kf_baseline, width, label='KF Baseline')
    rects2 = ax.barh(x + 0.1, kf_augmented, width, label='KF Augmented')
    rects3 = ax.barh(x - 0.1, ef_baseline, width, label='EF Baseline')
    rects4 = ax.barh(x - 0.3, ef_augmented, width, label='EF Augmented')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('AUC Percentage')
    ax.set_ylabel('CASIA-FA test protocol')
    ax.set_title('The AUC for each CASIA-FA test protocol')
    # ax.set_yticks(labels,)
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend(bbox_to_anchor=(0.565, -0.1))
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig.tight_layout()

    plt.savefig("test.pdf")


if __name__ == "__main__":
    plot()
