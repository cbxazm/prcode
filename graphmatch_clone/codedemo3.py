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
import models
from torch_geometric.data import Data, DataLoader
import ssl
import os
import random
import javalang
import javalang.tree
import javalang.ast
import javalang.util
import t22 as F
from javalang.ast import Node
import torch
from anytree import AnyNode, RenderTree
#import treelib
from anytree import find
from edge_index import edges
import t22

print(F)

device=torch.device('cpu')
model = torch.load("models/gmngcj9")
model.eval()

codetype = "type1"

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node,nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)

class Queue():
    def __init__(self):
        self.__list = list()

    def isEmpty(self):
        return self.__list == []

    def push(self, data):
        self.__list.append(data)

    def pop(self):
        if self.isEmpty():
            return False
        return self.__list.pop(0)
def traverse(node,index):
    queue = Queue()
    queue.push(node)
    result = []
    while not queue.isEmpty():
        node = queue.pop()
        result.append(get_token(node))
        result.append(index)
        index+=1
        for (child_name, child) in node.children():
            #print(get_token(child),index)
            queue.push(child)
    return result

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)
def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)
def getedge_flow(node,vocabdict,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])
            '''if len(node.children[1].children)!=0:
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[0].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[-1].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopend'])
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[-1].id)
                edgetype.append(edges['For_loopend'])'''
    #if token=='ForControl':
        #print(token,len(node.children))
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['If']])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,edgetype,ifedge,whileedge,foredge)
def getedge_nextstmt(node,vocabdict,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append([edges['Nextstmt']])
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
            edgetype.append([edges['Prevstmt']])
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt,edgetype)
def getedge_nexttoken(node,vocabdict,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,vocabdict,edgetype,tokenlist):
        token=node.token
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,vocabdict,edgetype,tokenlist)
    gettokenlist(node,vocabdict,edgetype,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append([edges['Nexttoken']])
            src.append(tokenlist[i+1])
            tgt.append(tokenlist[i])
            edgetype.append([edges['Prevtoken']])
def getedge_nextuse(node,vocabdict,src,tgt,edgetype,variabledict):
    def getvariables(node,vocabdict,edgetype,variabledict):
        token=node.token
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if not variabledict.__contains__(variable):
                variabledict[variable]=[variablenode.id]
            else:
                variabledict[variable].append(variablenode.id)
        for child in node.children:
            getvariables(child,vocabdict,edgetype,variabledict)
    getvariables(node,vocabdict,edgetype,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append([edges['Nextuse']])
                src.append(variabledict[v][i+1])
                tgt.append(variabledict[v][i])
                edgetype.append([edges['Prevuse']])
def createast(codetype):
    asts=[]
    paths=[]
    alltokens=[]
    dirname = 'demosrc/type1/t1.txt'
    programfile=open(dirname,encoding='utf-8')
    programtext=programfile.read()
    #将代码token化
    programtokens=javalang.tokenizer.tokenize(programtext)
    #解析成ast抽象语法树
    programast=javalang.parser.parse(programtokens)
    paths.append(dirname)
    asts.append(programast)
    #获取所有的词汇集合放到alltokens中
    get_sequence(programast,alltokens)
    programfile.close()
    #print(programast)
    #print(alltokens)
    astdict=dict(zip(paths,asts))
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount=0
    switchcount=0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print(ifcount,whilecount,forcount,blockcount,docount,switchcount)
    print('allnodes ',len(alltokens))
    #词汇集合去重
    alltokens=list(set(alltokens))
    #统计词汇量
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    #通过zip建立词汇以及对应的下标字典
    vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    return astdict,[vocabsize,vocabdict],paths

def createseparategraph(astdict,vocablen,vocabdict,device,mode='astonly',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True):
    pathlist=[]
    treelist=[]
    print('nextsib ',nextsib)
    print('ifedge ',ifedge)
    print('whileedge ',whileedge)
    print('foredge ',foredge)
    print('blockedge ',blockedge)
    print('nexttoken', nexttoken)
    print('nextuse ',nextuse)
    print(len(astdict))
    for path,tree in astdict.items():
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)
        x = []  #边的词典索引集合(在之前vocadict中记录的)
        ##双向边表示集合
        edgesrc = [] ##边的id集合
        edgetgt = [] ##双向边id集合
        # 边的属性表示 edges = {'Nexttoken': 2, 'Prevtoken': 3, 'Nextuse': 4, 'Prevuse': 5, 'If': 6, 'Ifelse': 7, 'While': 8, 'For': 9,'Nextstmt': 10, 'Prevstmt': 11, 'Prevsib': 12}
        edge_attr=[]
        if mode=='astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)
            if nextsib==True:
                getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
            if blockedge==True:
                getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            tokenlist=[]
            if nexttoken==True:
                getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
            variabledict={}
            if nextuse==True:
                getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)
        edge_index=[edgesrc, edgetgt]
        astlength=len(x)
        pathlist.append(path)
        treelist.append([[x,edge_index,edge_attr],astlength])
        astdict[path]=[[x,edge_index,edge_attr],astlength]
    return astdict
def creategmndata(pathlist,treedict,vocablen,vocabdict,device):
    testdata=[]
    datalist = []
    print('test data')
    data1 = treedict[pathlist[0]]
    data2 = treedict[pathlist[1]]
    x1, edge_index1, edge_attr1, ast1length = data1[0][0], data1[0][1], data1[0][2], data1[1]
    x2, edge_index2, edge_attr2, ast2length = data2[0][0], data2[0][1], data2[0][2], data2[1]
    if edge_attr1 == []:
        edge_attr1 = None
        edge_attr2 = None
    data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
    datalist.append(data)
    return datalist
def createpairdata(treedict,pathlist,device):
    datalist=[]
    countlines=1
    #726377
    print("总长度",pathlist.__len__())
    for line in pathlist:
        print(countlines)
        countlines += 1
        pairinfo = line.split()
        code1path=pairinfo[0]
        code2path = pairinfo[1]
        label=int(pairinfo[2])
        data1 = treedict[code1path]
        data2 = treedict[code2path]
        x1,edge_index1,edge_attr1,ast1length=data1[0][0],data1[0][1],data1[0][2],data1[1]
        x2,edge_index2,edge_attr2,ast2length=data2[0][0],data2[0][1],data2[0][2],data2[1]
        if edge_attr1==[]:
            edge_attr1 = None
            edge_attr2 = None
        data = [[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2], label]
        datalist.append(data)
    return datalist
def test(dataset):
    model.eval()
    count=0
    correct=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    for data in dataset:
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
        x1=torch.tensor(x1, dtype=torch.long, device=device)
        x2=torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1!=None:
            edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction=model(data)
        output=F.cosine_similarity(prediction[0],prediction[1])
        print(output)
    return results


#创建代码抽象语法树以及词汇量统计
astdict,vocainfo,pathlist=createast(codetype)
#对于创建好的抽象语法树进行数据流控制流边的添加
treedict = createseparategraph(astdict, vocainfo[0], vocainfo[1],device,mode="astandnext",nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
#整合两个文件的所有数据
testdata = creategmndata(pathlist,treedict,vocainfo[0],vocainfo[1],device)
#通过模型进行计算
test(testdata)
