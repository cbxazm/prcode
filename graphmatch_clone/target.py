import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from createclone_java import createast,creategmndata,createseparategraph
import models
from torch_geometric.data import Data, DataLoader
import ssl
import javalang
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=False)
parser.add_argument("--ifedge", default=False)
parser.add_argument("--whileedge", default=False)
parser.add_argument("--foredge", default=False)
parser.add_argument("--blockedge", default=False)
parser.add_argument("--nexttoken", default=False)
parser.add_argument("--nextuse", default=False)
parser.add_argument("--data_setting", default='0')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_known_args()
threshold = args[0].__getattribute__("threshold")
device=torch.device('cpu')
astdict,vocablen,vocabdict=createast()
treedict=createseparategraph(astdict, vocablen, vocabdict,device,mode="astandnext",nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
traindata,validdata,testdata=creategmndata("0",treedict,vocablen,vocabdict,device)
# model=models.GMNnet(vocablen,embedding_dim=100,num_layers=4,device=device).to(device)
# model = torch.load("models/gmngcj9",map_location=torch.device('cpu'))
def test(dataset):
    # model.eval()
    with open("gcjresult/astandnext_epoch_9") as f:
    # with open("ggnngcjresult/astandnext_epoch_6") as f:
        arr = f.readlines()
        arr = [i.replace("\n", "") for i in arr]
        print(arr)
    # print(arr)
    count=0
    correct=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    index = 0
    for data,label in dataset:
        label=torch.tensor(label, dtype=torch.float, device=device)
        # x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
        # x1=torch.tensor(x1, dtype=torch.long, device=device)
        # x2=torch.tensor(x2, dtype=torch.long, device=device)
        # edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        # edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
        # if edge_attr1!=None:
        #     edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
        #     edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
        # data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        # prediction=model(data)
        # output=F.cosine_similarity(prediction[0],prediction[1])
        # results.append(output.item())
        # prediction = torch.sign(output).item()
        # if prediction>args.threshold and label.item()==1:
        prediction = float(arr[index][0:8])
        index = index + 1
        if prediction>threshold and label.item()==1:
            tp+=1
            #print('tp')
        # if prediction<=args.threshold and label.item()==-1:
        if prediction<=threshold and label.item()==-1:
            tn+=1
            #print('tn')
        # if prediction>args.threshold and label.item()==-1:
        if prediction>threshold and label.item()==-1:
            fp+=1
            #print('fp')
        # if prediction<=args.threshold and label.item()==1:
        if prediction<=threshold and label.item()==1:
            fn+=1
            #print('fn')
    print(tp,tn,fp,fn)
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        print('precision is none')
        return
    p=tp/(tp+fp)
    if tp+fn==0:
        print('recall is none')
        return
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results

# test(validdata)
test(testdata)

# GMNnet(
#   (embed): Embedding(8033, 100)
#   (edge_embed): Embedding(20, 100)
#   (gmnlayer): GMNlayer(
#     (fmessage): Linear(in_features=300, out_features=100, bias=True)
#     (fnode): GRUCell(200, 100)
#   )
#   (mlp_gate): Sequential(
#     (0): Linear(in_features=100, out_features=1, bias=True)
#     (1): Sigmoid()
#   )
#   (pool): GlobalAttention(gate_nn=Sequential(
#     (0): Linear(in_features=100, out_features=1, bias=True)
#     (1): Sigmoid()
#   ), nn=None)
# )
