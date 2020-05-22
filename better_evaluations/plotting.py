import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, relative=True, figsize=(17, 12)):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    if relative:
        cm = (cm / cm.sum(axis=1)).round(2)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True,cmap="OrRd")
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title('Confusion Matrix')
    
    
def plot_precision_recall(y_true, y_pred_proba, figsize=(17, 10)):
    plt.figure(figsize=figsize)
    precision = dict()
    recall = dict()
    n_classes = len(set(y_true))
    for i in range(n_classes):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_true == i, y_pred_proba.values[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()