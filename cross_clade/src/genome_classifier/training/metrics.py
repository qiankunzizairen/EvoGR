import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix

def compute_metrics(y_true, probabilities, num_classes=4):
    y_true=np.asarray(y_true); p=np.asarray(probabilities); pred=p.argmax(1)
    out={"Accuracy":float(accuracy_score(y_true,pred)),"MCC":float(matthews_corrcoef(y_true,pred)),"Macro-F1":float(f1_score(y_true,pred,average="macro",zero_division=0)),"Macro-Precision":float(precision_score(y_true,pred,average="macro",zero_division=0)),"Macro-Recall":float(recall_score(y_true,pred,average="macro",zero_division=0))}
    y=np.eye(num_classes)[y_true]
    try: out["Macro-AUROC"]=float(roc_auc_score(y,p,multi_class="ovr",average="macro"))
    except ValueError: out["Macro-AUROC"]=float("nan")
    try: out["Macro-AUPRC"]=float(average_precision_score(y,p,average="macro"))
    except ValueError: out["Macro-AUPRC"]=float("nan")
    return out, confusion_matrix(y_true,pred,labels=list(range(num_classes)))
