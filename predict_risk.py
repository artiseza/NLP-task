# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 16:59:55 2021

@author: Alan Lin
"""

import torch
import json
import re
import numpy as np
import pandas as pd
import random
import os
import csv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from model import risk_model
from dataset import all_dataset,predict_dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

bout = "_5"

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(8,4))
    #rcParams['figure.figsize'] = (8.0, 4.0)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def readcsv(path: str):
    ans=[]
    with open(path, "r") as risk_ans:
        for i, line in enumerate(csv.reader(risk_ans)):
            if i == 0:
                continue
            ans.append(line[3])
    return ans 

def risk_test(exp_path: str,state):
    # Hyperparameters
    batch_size = 8
    with open(os.path.join(exp_path, "cfg.json"), "r") as f:
        args = json.load(f)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load checkpoints
    ckpts = []
    for ckpt in os.listdir(args["model_path"]):
        match = re.match(r'model-(\d+).pt', ckpt)
        if match is None:
            continue
        ckpts.append(int(match.group(1)))
    ckpts = sorted(ckpts)

    # Log writer
    writer = SummaryWriter(log_dir=args["log_path"])
    
    if state == 0:
        print("evaluate on validation set...")
    else:    
        print("evaluate on testing set...")

    # Data
    
    # data = all_dataset(args["qa_train"], args["risk_train"])
    if state == 0:
        data = all_dataset(args["qa_val"], args["risk_val"],args["high_risk_word"],args["low_risk_word"])
    else:
        data = predict_dataset(args["qa_test"], args["risk_test"])

    for ckpt in ckpts:
        # Model
        model = risk_model(args["d_emb"], args["n_cls_layers"], 0.0)
        model.load_state_dict(torch.load(os.path.join(
            args["model_path"], f"model-{ckpt}.pt")))
        model = model.eval()
        model = model.to(device)

        dataldr = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        answer = {}
        answer["risk"] = data.risk
        pred = {"risk": []}
        pred_ans = []
   
        for batch_data in tqdm_dldr:
            batch_document = []
            for idx in batch_data["article_id"]:
                batch_document.append(data.article[idx])

            batch_document = torch.LongTensor(batch_document).to(device)
            pred_risk = model(batch_document)

            for i, idx in enumerate(batch_data["article_id"]):
                if idx >= len(pred["risk"]):
                    pred["risk"].append(pred_risk[i].tolist())
        if state ==0:
            for i in range(len(pred['risk'])):
                if pred['risk'][i] <= 0.5:
                    pred_ans.append(0)
                else:
                    pred_ans.append(1)
                    
            print("Answer : ",answer['risk'],"\nPredic : ", pred['risk'])
            
            
            # Draw confusion matrix
            cm = confusion_matrix(answer['risk'], pred_ans)
            cm_plot_labels = ['0','1'] #['low', 'high']
            plot_confusion_matrix(cm, cm_plot_labels)
            
            # 2 class
            TP = cm[1,1]
            print('TP :',TP)
            FN = cm[1,0]
            print('FN :',FN)
            FP = cm[0,1]
            print('FP :',FP)
            TN = cm[0,0]
            print('TN :',TN)
            sensitivity = TP/(TP+FN)
            specificity = TN/(FP+TN)
            print('Sensitivity :',sensitivity)
            print('Specificity :',specificity)
            
            print(f"AUROC score: {roc_auc_score(answer['risk'], pred['risk'])}")
        writer.add_scalar("train", ckpt)
    if state == 1:       
        risk_id = [i+1 for i in range(len(pred['risk']))]
        article_id = pd.DataFrame(risk_id, columns=['article_id'])
        risk_result_df = pd.DataFrame(pred['risk'], index = article_id['article_id'], columns=['probability'])          
        print("Predic : \n", risk_result_df)
        # try:
        #     print(f"AUROC score: {roc_auc_score(risk_ans, pred['risk'])}")
        # except ValueError:
        #     pass     
        risk_result_df.to_csv('./predict/decision.csv', encoding='utf8')   
      

if __name__ == "__main__":
    exp_path = os.path.join("exp", "_risk", bout)
    risk_test(exp_path,0)
    risk_test(exp_path,1)