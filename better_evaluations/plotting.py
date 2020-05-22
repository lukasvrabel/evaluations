import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd


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
    
    
def plot_PR_interactive(y_true, y_pred_proba):
    def yield_PR_df(y_true, y_pred_proba):
        n_classes = len(set(y_true))
        for i in range(n_classes):
            precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y_true == i, y_pred_proba.values[:, i])
            for p, r, t in zip(precisions, recalls, thresholds + [1]):
                yield {
                    'precision': f'{p:.3f}',
                    'recall': f'{r:.3f}',
                    'f1': f'{2 * (p * r) / (p + r):.3f}',
                    'threshold': f'{t:.3f}',
                    'class': i,
                }

    prdf = pd.DataFrame(yield_PR_df(y_true, y_pred_proba))
    fig = px.line(prdf, x="recall", y="precision", color='class', height=600, hover_data=prdf.columns)
    fig.update_traces(mode="markers+lines")
    return fig
