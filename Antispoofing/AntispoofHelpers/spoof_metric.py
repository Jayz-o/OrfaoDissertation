import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import seaborn as sns

plt.style.use("dark_background")
# plt.style.use("default")

SPOOF_METRIC_INFO_CSV = "spoof_metric_info.csv"
TUNE_METRIC_INFO_CSV = "tune_spoof_metric_info.csv"

ROC_PLOT_NAME = "ROC.pdf"

CONFUSION_MATRIX_NAME = "Confusion_Matrix.pdf"

CLASSIFICATION_REPORT = "classification_report.csv"
PROTOCOL_COL = "protocol"
"""
https://sites.google.com/qq.com/face-anti-spoofing/evaluation
* TP: The attacks are recognized as the attacks;
* TN: The real samples are recognized as the real ones;
* FP: The real samples are recognized as the attacks;
* FN: The attacks are recognized as the real samples;
[1] Aghajan, H., Augusto, J. C., & Delgado, R. L. C. (Eds.). (2009). Human-centric interfaces for ambient intelligence. Academic Press
https://www.sciencedirect.com/topics/computer-science/false-rejection-rate
https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
https://chalearnlap.cvc.uab.cat/challenge/33/track/33/metrics/
"""


def calc_tn_fp_fn_tp(y_true, y_pred):
    """
    Obtain the:
    * TP: The attacks are recognized as the attacks.
    * TN: The real samples are recognized as real.
    * FP: The real samples are recognized as the attacks.
    * FN: The attacks are recognized as real.
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :return: tp, tn, fp, fn
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn


def calc_apcer(tp, fn):
    """
    Attack Presentation Classification Error Rate (APCER),
    A.K.A False Rejection Rate for Antispoofing or False Negative Rate:
    APCER = FN / (TP + FN)
    :param tp: The attacks are recognized as the attacks.
    :param fn: The attacks are recognized as real.
    :return: APCER value [0, 1]
    """
    return np.divide(fn, np.add(tp, fn))


def calc_bpcer(tn, fp):
    """
    Normal (Bona Fide) Presentation Classification Error Rate (NPCER/BPCER),
    A.K.A False Acceptance Rate for Antispoofing or False Positive Rate:
    NPCER or BPCER = FP/(FP + TN)
    :param tn: The real samples are recognized as real.
    :param fp: The real samples are recognized as the attacks.
    :return: BPCER value [0, 1]
    """
    return np.divide(fp, np.add(fp, tn))


def calc_acer(apcer, bpcer):
    """
    Average Classification Error Rate (ACER), A.K.A Half Total Error Rate for Antispoofing:
    ACER = (APCER + NPCER) / 2
    :param apcer: The attack presentation classification error.
    :param bpcer: The bonafide or normal presentation classification error.
    :return: ACER value [0, 1]
    """
    return np.divide(np.add(apcer, bpcer), 2)


def calc_auc(y_true, y_pred):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve.
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :return: auc
    """
    return roc_auc_score(y_true, y_pred)


def calc_roc(y_true, y_pred, save_dir=None, must_show=False):
    """
    Receiver Operating Characteristic Curve
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param save_dir: The directory to save the plot to.
    :param must_show: True to show the plot.
    :return: fpr, tpr, threshold, roc_path
    """
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    roc_path = None
    if save_dir is not None or must_show:
        if must_show:
            with plt.ion():
                plt.subplots(1, figsize=(10, 10))
                plt.title('Receiver Operating Characteristic - DecisionTree')
                plt.plot(fpr, tpr)
                plt.plot([0, 1], ls="--")
                plt.plot([0, 0], [1, 0], c=".7")
                plt.plot([1, 1], c=".7")
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.ion()
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    roc_path = os.path.join(save_dir, ROC_PLOT_NAME)
                    plt.savefig(roc_path)
                    roc_path = ROC_PLOT_NAME
        else:
            with plt.ioff():
                plt.subplots(1, figsize=(10, 10))
                plt.title('Receiver Operating Characteristic - DecisionTree')
                plt.plot(fpr, tpr)
                plt.plot([0, 1], ls="--")
                plt.plot([0, 0], [1, 0], c=".7")
                plt.plot([1, 1], c=".7")
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.ion()
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    roc_path = os.path.join(save_dir, ROC_PLOT_NAME)
                    plt.savefig(roc_path)
                    roc_path = ROC_PLOT_NAME

    return fpr, tpr, threshold, roc_path


def calc_eer(fpr, tpr, thresholds):
    """
    Obtain the equal error rate (eer).
    :param fpr: The false positive rate.
    :param tpr: The true positive rate.
    :param thresholds: The thresholds used.
    :return: eer, threshold
    https://gist.github.com/aqzlpm11/9e33a20c5e8347537bec532ae7319ba8
    """
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def plot_cm(y_true, y_pred, save_dir=None, must_show=False, class_names=['Real', 'Spoof']):
    """
    plot a confusion matrix
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param save_dir: The directory to save the plot to.
    :param must_show: True to show the plot.
    :return: confusion plot path
    """
    if save_dir is None and must_show is False:
        print("Warning: plot_cm resulted in no output. Check your args.")
        return
    confusion_path = None
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    indices = np.unique(y_true)
    cm_labels = [class_names[i] for i in indices]
    cm = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    if must_show:
        with plt.ion():
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=annot, fmt='', ax=ax)

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                confusion_path = os.path.join(save_dir, CONFUSION_MATRIX_NAME)
                plt.savefig(confusion_path)
                confusion_path = CONFUSION_MATRIX_NAME
            plt.show()
    else:
        with plt.ioff():
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=annot, fmt='', ax=ax)

            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                confusion_path = os.path.join(save_dir, CONFUSION_MATRIX_NAME)
                plt.savefig(confusion_path)
                confusion_path = CONFUSION_MATRIX_NAME

    return confusion_path


def create_classification_report(y_true, y_pred, save_dir=None, must_show=False, class_names=['Real', 'Spoof']):
    """
    Produce a classification report.
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param save_dir: The directory to save the report to.
    :param must_show: True to print the classification report
    :param class_names: The class names matching the output of the classifier. Default: 0 is Real, 1 is Spoof
    :return: Classification report dataframe, report_save_path
    """
    if save_dir is None and must_show is False:
        print("Warning: the classification report resulted in no output. Check your args.")
    report_save_path = None
    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names, digits=4)
    df_classification_report = pd.DataFrame(report)
    # df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        report_save_path = os.path.join(save_dir, CLASSIFICATION_REPORT)
        df_classification_report.to_csv(report_save_path)
        report_save_path = CLASSIFICATION_REPORT
    if must_show:
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    return df_classification_report, report_save_path

# todo: Look at adding support for saving the file names of the images classified as wrong
def determine_spoof_metrics(y_true, y_pred, protocol_name, fold_number,protocol_number, save_dir=None, must_show=False, return_tune_dic=True):
    """
    Save all the spoof metrics.
    :param y_true: The ground truth labels for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param y_pred: The classifier's predictions for the
    samples. A list containing only 0 - false sample, 1 - true sample.
    :param save_dir: The directory in which to save the spoof metrics csv file.
    :param must_show: True to display all the metric visualisations
    :return: A dictionary containing the spoof metrics
    """

    if type(y_true) is not list:
        y_true = list(y_true)

    if type(y_pred) is not list:
        y_pred = list(y_pred)

    tp, tn, fp, fn = calc_tn_fp_fn_tp(y_true, y_pred)
    apcer = calc_apcer(tp, fn)
    bpcer = calc_bpcer(tn, fp)
    acer = calc_acer(apcer, bpcer)
    # fpr should = bpcer
    unique_classes = set(y_true)
    if len(unique_classes) >=2:
        fpr, tpr, roc_threshold, roc_path = calc_roc(y_true, y_pred, save_dir, must_show)
        eer, eer_threshold = calc_eer(fpr, tpr, roc_threshold)
        auc = calc_auc(y_true, y_pred)

    confusion_path = plot_cm(y_true, y_pred, save_dir, must_show)
    classification_report_df, report_path = create_classification_report(y_true, y_pred, save_dir, must_show)

    metric_dic = {
        'protocol_number': protocol_number,
        'fold_number': fold_number,
        'protocol': protocol_name,
        'TN': tn,
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'APCER': apcer,
        'BPCER': bpcer,
        'ACER': acer,
    }

    report_dic = metric_dic.copy()




    if len(unique_classes) >=2:
        metric_dic['EER'] = eer
        metric_dic['EER_Thresh'] = eer_threshold.item()
        metric_dic['AUC'] = auc
        report_dic = metric_dic.copy()

        metric_dic['ROC'] = roc_path


    metric_dic['CM'] = confusion_path
    metric_dic['REPORT'] = report_path


    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = pd.DataFrame.from_dict([metric_dic])
        df.to_csv(os.path.join(save_dir, SPOOF_METRIC_INFO_CSV), index=False)
        df = pd.DataFrame.from_dict([report_dic])
        df.to_csv(os.path.join(save_dir, TUNE_METRIC_INFO_CSV), index=False)

    if return_tune_dic:
        return report_dic
    else:
        return metric_dic

if __name__ == "__main__":
    # output labels
    spoof_label = 1
    real_label = 0
    # following: https://www.educative.io/edpresso/what-is-precision-and-recall inputs to check correctness.
    ground_truth = np.concatenate((np.array([spoof_label] * 13), np.array([real_label] * 20), np.array([spoof_label] * 3), np.array([real_label] * 4)))
    predictions = np.concatenate((np.array([spoof_label] * 13), np.array([real_label] * 20), np.array([real_label] * 3), np.array([spoof_label] * 4)))

    determine_spoof_metrics(ground_truth, predictions, save_dir="./temp/", must_show=True)



