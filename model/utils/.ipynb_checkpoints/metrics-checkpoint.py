import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def get_metrics(predict, target, threshold=.5, predict_b=None):
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    auc = roc_auc_score(target, predict)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn) 
    spe = tn / (tn + fp) 
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Accuracy": np.round(acc, 4),
        "Sensitivity": np.round(sen, 4),
        "Specificity": np.round(spe, 4),
        "Precission": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }
    
def get_metrics_classification(logits, target, classes):
    predict = torch.argmax(logits, dim=1).cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    report = classification_report(target, predict, output_dict=True, zero_division=0.0)
    
    result = {'accuracy': report['accuracy']}
    for class_idx, class_report in report.items():
        if class_idx in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        for metric in ['precision', 'recall', 'f1-score']:
            result[f'{classes[int(class_idx)]}/{metric}'] = class_report[metric]
            
    return result