import torch
from sklearn.metrics import f1_score, confusion_matrix
import wandb

def classification_metrics(logits, targets):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    f1 = f1_score(targets, preds, average='macro')
    confmat = confusion_matrix(targets, preds)

    wandb.log({
        "f1_score": f1,
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                            y_true=targets, preds=preds,
                            class_names=[str(i) for i in range(logits.shape[1])])
    })

    return {"f1_score": f1}
