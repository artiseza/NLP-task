import torch
import json
import re
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from model import qa_model
from dataset import all_dataset,predict_dataset
from torch.utils.tensorboard import SummaryWriter

'''
Here is the part of training the model, you need to set hyperparameters by yourself.
'''
ans=["A","B","C"] #[0,1,2] >> ["A","B","C"]
bout = "_10"
def qa_train():
    # Hyperparameters
    args = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "n_epoch": 5000,
        "log_step": 500,
        "save_step": 500,
        "d_emb": 300,
        "n_cls_layers": 2,
        "p_drop": 0.25,
        "weight_decay": 0.1,
        "model_path": os.path.join("exp", "_qa", bout),
        "log_path": os.path.join("log", "_qa", bout),
        
        "qa_train": os.path.join("data", "Train_qa_ans.json"),
        "risk_train": os.path.join("data", "Train_risk_classification_ans.csv"),
        "qa_val": os.path.join("data", "SampleData_QA.json"),
        "risk_val": os.path.join("data", "SampleData_RiskClassification.csv"),
              
        "high_risk_word": os.path.join("data", "high_risk_words.txt"),
        "low_risk_word": os.path.join("data", "low_risk_words.txt"),
    }

    # Save training configuration
    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])
    with open(os.path.join(args["model_path"], "cfg.json"), "w") as f:
        json.dump(args, f)

    # Random seed
    random_seed = args["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Model
    model = qa_model(args["d_emb"], args["n_cls_layers"], args["p_drop"])
    # model = torch.load('./model-pre.pt') # load Pre-Trained Weight on Data reconstruction 
    model = model.train()
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args["weight_decay"],
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args["learning_rate"])

    # Data
    data = all_dataset(args["qa_val"], args["risk_val"],args["high_risk_word"],args["low_risk_word"])
    dataldr = torch.utils.data.DataLoader(data, batch_size=args["batch_size"], shuffle=True)

    # Log writer
    if not os.path.isdir(args["log_path"]):
        os.makedirs(args["log_path"])
    writer = SummaryWriter(log_dir=args["log_path"])

    # Train loop
    step = 0
    avg_loss = 0
    for epoch in range(args["n_epoch"]):
        tqdm_dldr = tqdm(dataldr)

        for batch_data in tqdm_dldr:
            optimizer.zero_grad()
            batch_document = []

            for idx in batch_data["article_id"]:
                batch_document.append(data.article[idx])

            batch_document = torch.LongTensor(batch_document).to(device)
            batch_question = batch_data["question"].to(device)
            batch_choice = batch_data["choice"].to(device)
            batch_qa_answer = batch_data["qa_answer"].float().to(device)

            loss = model.loss_fn(batch_document, batch_question,
                                 batch_choice, batch_qa_answer)
            loss.backward()
            optimizer.step()

            step = step + 1
            avg_loss = avg_loss + loss

            if step % args["log_step"] == 0:
                avg_loss = avg_loss/args["log_step"]
                tqdm_dldr.set_description(f"epoch:{epoch}, loss:{avg_loss}")
                writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0

            if step % args["save_step"] == 0:
                if not os.path.isdir(args["model_path"]):
                    os.makedirs(args["model_path"])
                torch.save(model.state_dict(), os.path.join(
                    args["model_path"], f"model-{step}.pt"))

    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])

    torch.save(model.state_dict(), os.path.join(args["model_path"], f"model-{step}.pt"))


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
            print(f"qa: {accuracy_score(answer['qa'], pred['qa'])}")
        writer.add_scalar("train", accuracy_score(answer['qa'], pred['qa']), ckpt)
        # writer.add_scalar("train", pred['qa'], ckpt)
    if state == 1:    
        qa_id = [i+1 for i in range(len(pred['qa']))]
        article_id = pd.DataFrame(qa_id, columns=['id'])
        answer = [ans[i] for i in pred['qa']]
        qa_result_df = pd.DataFrame(answer, index = article_id['id'], columns=['answer'])          
        print("Predic : \n", qa_result_df)
        qa_result_df.to_csv('./predict/qa.csv', encoding='utf8')

if __name__ == "__main__":
    qa_train()
    exp_path = os.path.join("exp", "_qa", bout)
    qa_test(exp_path,0)
    qa_test(exp_path,1)