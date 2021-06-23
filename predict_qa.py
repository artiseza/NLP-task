# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 00:51:08 2021

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
from model import qa_model
from dataset import all_dataset,predict_dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

ans=["A","B","C"] #[0,1,2] >> ["A","B","C"]
bout = "_10"
       
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
    with open(path, "r", encoding = "utf-8") as QA_ans:
        for i, line in enumerate(csv.reader(QA_ans)):
            if i == 0:
                continue
            ans.append(line[1])
    return ans 

def qa_test(exp_path: str,state):
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
    if state == 0:
        data = all_dataset(args["qa_val"], args["risk_val"],args["high_risk_word"],args["low_risk_word"])
    else:
        data = predict_dataset(args["qa_test"], args["risk_test"])

    for ckpt in ckpts:
        # Model
        model = qa_model(args["d_emb"], args["n_cls_layers"], 0.0)
        model.load_state_dict(torch.load(os.path.join(
            args["model_path"], f"model-{ckpt}.pt")))
        model = model.eval()
        model = model.to(device)

        dataldr = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        answer = {"qa": []}
        pred = {"qa": []}

        for batch_data in tqdm_dldr:
            batch_document = []

            for idx in batch_data["article_id"]:
                batch_document.append(data.article[idx])

            batch_document = torch.LongTensor(batch_document).to(device)
            batch_question = batch_data["question"].to(device)
            batch_choice = batch_data["choice"].to(device)

            answer["qa"] = answer["qa"] + batch_data["qa_answer"].argmax(dim=-1).tolist()

            pred_qa = model(batch_document, batch_question, batch_choice)

            pred["qa"] = pred["qa"] + pred_qa.argmax(dim=-1).tolist()
            
        if state == 0:
            print("Answer : ",answer['qa'],"\nPredic : ", pred['qa'])       
            # Draw confusion matrix
            cm = confusion_matrix(answer['qa'], pred['qa'])
            cm_plot_labels = ['A', 'B','C'] #['low', 'medium', 'high']
            plot_confusion_matrix(cm, cm_plot_labels)
            
            # 3 class
            # True positive rate(Sensitivity)
            diagonal = [cm[0,0],cm[1,1],cm[2,2]]
            print('diagonal(預測為Class且實際為Class數量) : ',diagonal)
            sum_of_row = [cm[0,0]+cm[0,1]+cm[0,2],cm[1,0]+cm[1,1]+cm[1,2],cm[2,0]+cm[2,1]+cm[2,2]]
            print('sum_of_row(Class實際總數) : ',sum_of_row)
            sensitivity = [diagonal[0]/sum_of_row[0],diagonal[1]/sum_of_row[1],diagonal[2]/sum_of_row[2]]
            print('Sensitivity :',sensitivity)
        
            # True negative rate (Specificity)
            diagonal_not_class = [cm[1,1]+cm[2,2],cm[0,0]+cm[2,2],cm[0,0]+cm[1,1]]
            print('diagonal_not_class(預測為Class且實際不為Class數量) : ',diagonal_not_class)
            sum_of_row_not_class = [sum_of_row[1]+sum_of_row[2],sum_of_row[0]+sum_of_row[2],sum_of_row[0]+sum_of_row[1]]
            print('sum_of_ro_not_classw(不為Class實際總數) : ',sum_of_row_not_class)
            specificity =[diagonal_not_class[0]/sum_of_row_not_class[0],diagonal_not_class[1]/sum_of_row_not_class[1],diagonal_not_class[2]/sum_of_row_not_class[2]]
            print('Specificity :',specificity)
            print(f"qa score: {accuracy_score(answer['qa'], pred['qa'])}")
        writer.add_scalar("train", accuracy_score(answer['qa'], pred['qa']), ckpt)
        # writer.add_scalar("train", pred['qa'], ckpt)
    if state == 1:    
        qa_id = [i+1 for i in range(len(pred['qa']))]
        article_id = pd.DataFrame(qa_id, columns=['id'])
        answer = [ans[i] for i in pred['qa']]
        qa_result_df = pd.DataFrame(answer, index = article_id['id'], columns=['answer'])          
        print("Predic : \n", qa_result_df)
        # qa_result_df.to_csv('./predict/qa.csv', encoding='utf8')
        qa_result_df.to_csv('./predict/final/qa.csv', encoding='utf8')

if __name__ == "__main__":
    exp_path = os.path.join("exp", "_qa", bout)
    qa_test(exp_path,0)
    qa_test(exp_path,1)