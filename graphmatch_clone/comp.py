import os
import re
import copy
import itertools

from gensim import matutils
from datetime import datetime
from collections import Counter

from util import wordext
from util import localfile

from git import *
from fetch_raw_diff import *
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
from javalang.ast import Node
import torch
from anytree import AnyNode, RenderTree
#import treelib
from anytree import find
from edge_index import edges

# text_sim_type = 'lsi'
text_sim_type = 'tfidf'

code_sim_type = 'tfidf'
# code_sim_type = 'bow'

extract_sim_type = 'ori_and_overlap'

add_timedelta = False
add_conf = False
model_path = '/DATA/newcbx/model/'

model = torch.load("models/gmngcj9")
device=torch.device('cpu')
# ---------------------------------------------------------------------------

def counter_similarity(A_counter, B_counter):
    C = set(A_counter) | set(B_counter)
    tot1, tot2 = 0, 0
    for x in C:
        tot1 += min(A_counter.get(x,0), B_counter.get(x,0))
        tot2 += max(A_counter.get(x,0), B_counter.get(x,0))
    if tot2 == 0:
        return 0
    return 1.0 * tot1 / tot2

def set_similarity(A, B):
    if (A is None) or (B is None):
        return 0
    if (len(A) == 0) or (len(B) == 0):
        return 0
    return len(set(A) & set(B)) / len(set(A) | set(B))
    
def list_similarity(A, B):
    if (A is None) or (B is None):
        return 0
    if (len(A) == 0) or (len(B) == 0):
        return 0
    A_counter = wordext.get_counter(A)
    B_counter = wordext.get_counter(B)
    return counter_similarity(A_counter, B_counter)

def vsm_bow_similarity(A_counter, B_counter):
    return matutils.cossim(list(A_counter.items()), list(B_counter.items()))

# ---------------------------------------------------------------------------

def get_tokens(text):
    return wordext.get_words_from_text(text)

def get_file_list(pull):
    return [x["name"] for x in pull['file_list']]

def get_location(pull):
    location_set = []
    for file in pull["file_list"]:
        for x in file["location"]["add"]:
            location_set.append([file["name"], int(x[0]), int(x[0]) + int(x[1])])
    return location_set

def get_code_from_file_list(pr_info):
    add_code = list(itertools.chain(*[wordext.get_words_from_file(file["name"], file["add_code"]) for file in pr_info]))
    del_code = list(itertools.chain(*[wordext.get_words_from_file(file["name"], file["del_code"]) for file in pr_info]))
    return [add_code, del_code]

def get_code_tokens(pull):
    return get_code_from_file_list(pull["file_list"])

def get_pull_on_overlap(pull, overlap_set):
    new_pull = copy.deepcopy(pull)
    new_pull["file_list"] = list(filter(lambda x: x["name"] in overlap_set, new_pull["file_list"]))
    return new_pull

def get_delta_code_tokens_counter(code_tokens_result):
    add_code_tokens = code_tokens_result[0]
    del_code_tokens = code_tokens_result[1]
    
    add_c = wordext.get_counter(add_code_tokens)
    del_c = wordext.get_counter(del_code_tokens)
    
    changed_c = Counter()
    for t in add_c:
        times = add_c[t] - del_c[t]
        if times > 0:
            changed_c[t] = times
    return changed_c

# ---------------------------------------------------------------------------

def location_similarity(la, lb):

    def cross(x1, y1, x2, y2):
        return not((y1 < x2) or (y2 < x1))

    if (la is None) or (lb is None):
        return 0.0
    
    '''
    # only calc on overlap files
    a_f = [x[0] for x in la]
    b_f = [x[0] for x in lb]
    c_f = set(a_f) & set(b_f)
    
    la = list(filter(lambda x: x[0] in c_f, la))
    lb = list(filter(lambda x: x[0] in c_f, lb))
    '''

    if len(la) + len(lb) == 0:
        return 0.0

    match_a = [False for x in range(len(la))]
    match_b = [False for x in range(len(lb))]
    
    index_b = {}
    for i in range(len(lb)):
        file = lb[i][0]
        if file not in index_b:
            index_b[file] = []
        index_b[file].append(i)
        
    for i in range(len(la)):
        file = la[i][0]
        for j in index_b.get(file,[]):
            if cross(la[i][1], la[i][2], lb[j][1], lb[j][2]):
                match_a[i] = True
                match_b[j] = True
    
    # weigh with code line
    a_match, a_tot = 0, 0
    for i in range(len(la)):
        part_line = la[i][2] - la[i][1]
        a_tot += part_line
        if match_a[i]:
            a_match += part_line
    
    b_match, b_tot = 0, 0
    for i in range(len(lb)):
        part_line = lb[i][2] - lb[i][1]
        b_tot += part_line
        if match_b[i]:
            b_match += part_line
    
    if a_tot + b_tot == 0:
        return 0
    return (a_match + b_match) / (a_tot + b_tot)
    # return (match_a.count(True) + match_b.count(True)) / (len(match_a) + len(match_b))

# ---------------------------------------------------------------------------

import nlp
model = None
def init_model_from_raw_docs(documents, save_id=None):
    global model
    if(os.path.exists(model_path + '%s.dictionary' % save_id)):
        print("model already load done")
        return
    res = []
    index = 1
    for document in documents:
        print(index)
        index = index + 1
        res.append(get_tokens(document))
        pass
    # model = nlp.Model([get_tokens(document) for document in documents], save_id)
    model = nlp.Model(res,save_id)
    print('init nlp model for text successfully!')


def get_text_sim(A, B,repo):
    save_path = repo.replace("/", "_") + '_allpr'
    code_model = nlp.Model("", save_path)
    # return code_model.query_sim_tfidf(counter2list(A), counter2list(B))
    A = get_tokens(A)
    B = get_tokens(B)
    if code_model is None:
        return [list_similarity(A, B)]
    
    if text_sim_type == 'lsi':
        sim = code_model.query_sim_lsi(A, B)

    if text_sim_type == 'tfidf':
        sim = code_model.query_sim_tfidf(A, B)

    # len_mul = model.query_vet_len_mul(A, B)
    len_mul = len(A) * len(B)

    return [sim]
    
    
code_model = None
def init_code_model_from_tokens(documents, save_id=None):
    global code_model
    code_model = nlp.Model(documents, save_id)
    print('init nlp model for code successfully!')

def counter2list(A):
    a_c = []
    for x in A:
        for t in range(A[x]):
            a_c.append(x)
    return a_c

def vsm_tfidf_similarity(A, B,repo):
    save_path = repo.replace("/","_")+'_allpr_code'
    code_model = nlp.Model("",save_path)
    return code_model.query_sim_tfidf(counter2list(A), counter2list(B))

# ---------------------------------------------------------------------------

'''
#detect cases: feat(xxxx)

def special_pattern(a):
    x1 = get_pr_and_issue_numbers(a)
    x2 = re.findall('\((.*?)\)', a)
    x1 = list(filter(lambda x: len(x) > 1, x1))
    return x1 + x2

def title_has_same_pattern(a, b):
    if set(special_pattern(a)) & set(special_pattern(b)):
        return True
    else:
        return False
'''

def check_pattern(A, B):
    ab_num = set([A["number"], B["number"]])
    a_text = str(A["title"]) + ' ' + str(A["body"])
    b_text = str(B["title"]) + ' ' + str(B["body"])

    a_set = set(get_numbers(a_text) + get_version_numbers(a_text)) - ab_num
    b_set = set(get_numbers(b_text) + get_version_numbers(b_text)) - ab_num
    if a_set & b_set:
        return 1
    else:
        def get_reasonable_numbers(x):
            return get_pr_and_issue_numbers(x) + get_version_numbers(x)

        a_set = set(get_reasonable_numbers(a_text)) - ab_num
        b_set = set(get_reasonable_numbers(b_text)) - ab_num
        if a_set and b_set and (a_set != b_set):
            return -1
        return 0

def get_code_sim(A, B,repo):
    A_overlap_code_tokens = get_code_tokens(A)
    B_overlap_code_tokens = get_code_tokens(B)
    
    A_delta_code_counter = get_delta_code_tokens_counter(A_overlap_code_tokens)
    B_delta_code_counter = get_delta_code_tokens_counter(B_overlap_code_tokens)
    
    if code_sim_type == 'bow':
        code_sim = vsm_bow_similarity(A_delta_code_counter, B_delta_code_counter)
        return [code_sim]
    if code_sim_type == 'jac':
        code_sim = counter_similarity(A_delta_code_counter, B_delta_code_counter)
        return [code_sim]
    if code_sim_type == 'tfidf':
        code_sim = vsm_tfidf_similarity(A_delta_code_counter, B_delta_code_counter,repo)
        return [code_sim]
    if code_sim_type == 'bow_two':
        code_sim_add = vsm_bow_similarity(wordext.get_counter(A_overlap_code_tokens[0]),
                                          wordext.get_counter(B_overlap_code_tokens[0]))
        code_sim_del = vsm_bow_similarity(wordext.get_counter(A_overlap_code_tokens[1]),
                                          wordext.get_counter(B_overlap_code_tokens[1]))
        return [code_sim_add, code_sim_del]
    if code_sim_type == 'bow_three':
        code_sim_delta = vsm_bow_similarity(A_delta_code_counter, B_delta_code_counter)
        code_sim_add = vsm_bow_similarity(wordext.get_counter(A_overlap_code_tokens[0]),
                                          wordext.get_counter(B_overlap_code_tokens[0]))
        code_sim_del = vsm_bow_similarity(wordext.get_counter(A_overlap_code_tokens[1]),
                                          wordext.get_counter(B_overlap_code_tokens[1]))
        return [code_sim_delta, code_sim_add, code_sim_del]
# ---------------------------------------------------------------------------

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
def createast(files):
    asts=[]
    paths=[]
    alltokens=[]
    for file in files:
        programfile=open(os.path.join(file),encoding='utf-8')
        programtext=programfile.read()
        #将代码token化
        programtokens=javalang.tokenizer.tokenize(programtext)
        #解析成ast抽象语法树
        programast=javalang.parser.parse(programtokens)
        paths.append(file)
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
        x = []  #节点的id集合(在之前vocadict中记录的)
        ##双向边表示集合
        edgesrc = []
        edgetgt = []
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
    datalist = []
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
def process(data):
    model.eval()
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
    return output.item()

def get_code_sim_new(A, B, repo,overlapset,num1,num2):
   filepath = "/data/funccbx/change_files/"+repo
   codesimlist = []
   for filename in overlapset:
       file1 = filepath+"/"+num1+"/"+filename
       file2 = filepath+"/"+num1+"/"+filename
       astdict, vocainfo, pathlist = createast([file1,file2])
       treedict = createseparategraph(astdict, vocainfo[0], vocainfo[1], device, mode="astandnext", nextsib=True,
                                      ifedge=True, whileedge=True, foredge=True, blockedge=True, nexttoken=True,
                                      nextuse=True)
       filedata = creategmndata(pathlist, treedict, vocainfo[0], vocainfo[1], device)
       codesim = process(filedata)
       codesimlist.append(codesim)
       pass
   count = 0.0
   for sim in codesimlist:
       count+=sim
       pass
   res = count/codesimlist.__len__() if codesimlist.__len__() !=0 else 0.0
   return  res

#未处理的变更文件计算相似性
def get_code_sim_before_new(A, B, repo,overlapset,num1,num2):
   filepath = "/data/utf8files/change_files/"+repo
   codesimlist = []
   for filename in overlapset:
       file1 = filepath+"/"+num1+"/"+filename
       file2 = filepath+"/"+num1+"/"+filename
       astdict, vocainfo, pathlist = createast([file1,file2])
       treedict = createseparategraph(astdict, vocainfo[0], vocainfo[1], device, mode="astandnext", nextsib=True,
                                      ifedge=True, whileedge=True, foredge=True, blockedge=True, nexttoken=True,
                                      nextuse=True)
       filedata = creategmndata(pathlist, treedict, vocainfo[0], vocainfo[1], device)
       codesim = process(filedata)
       codesimlist.append(codesim)
       pass
   count = 0.0
   for sim in codesimlist:
       count+=sim
       pass
   res = count/codesimlist.__len__() if codesimlist.__len__() !=0 else 0.0
   return  res



def calc_sim(A, B,repo):
    pattern = check_pattern(A, B)
    title_sim = get_text_sim(A["title"], B["title"],repo)
    desc_sim = get_text_sim(A["body"], B["body"],repo)
    file_list_sim = list_similarity(get_file_list(A), get_file_list(B))

    if ('merge_commit_flag' in A) and ('merge_commit_flag' in B):
        file_list_sim = set_similarity(get_file_list(A), get_file_list(B))

    overlap_files_set = set(get_file_list(A)) & set(get_file_list(B))

    A_overlap, B_overlap = get_pull_on_overlap(A, overlap_files_set), get_pull_on_overlap(B, overlap_files_set)
    code_sim = get_code_sim(A, B,repo)+get_code_sim(A_overlap, B_overlap,repo)
    commit_sim = get_commit_sim_vector(A,B)
    location_sim = [location_similarity(get_location(A), get_location(B))] + \
                    [location_similarity(get_location(A_overlap), get_location(B_overlap))]

    '''
    common_words = list(set(get_tokens(A["title"])) & set(get_tokens(B["title"])))
    overlap_title_len = len(common_words)
    
    if model is not None:
        title_idf_sum = model.get_idf_sum(common_words)
    else:
        title_idf_sum = 0
    '''

    def get_time(t):
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")

    overlap_files_len = len(overlap_files_set)

    ret = {
            'title': title_sim,
            'desc': desc_sim,
            'code': code_sim,
            'file_list': [file_list_sim, overlap_files_len],
            'location': location_sim,
            'commit':commit_sim
            # 'pattern': [pattern],
           }

    if add_conf:
        conf = 0
        for file in overlap_files_set:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if (file_name in A["title"]) and (file_name in B["title"]):
                conf = 1
                break
        ret['conf'] = [conf]

    if add_timedelta:
        delta_time = abs((get_time(A['created_at']) - get_time(B['created_at'])).days)
        ret['time'] = [delta_time]

    return ret


def calc_sim_new(A,B,repo,num1,num2):
    pattern = check_pattern(A, B)
    title_sim = get_text_sim(A["title"], B["title"], repo)
    desc_sim = get_text_sim(A["body"], B["body"], repo)
    file_list_sim = list_similarity(get_file_list(A), get_file_list(B))
    if ('merge_commit_flag' in A) and ('merge_commit_flag' in B):
        file_list_sim = set_similarity(get_file_list(A), get_file_list(B))

    overlap_files_set = set(get_file_list(A)) & set(get_file_list(B))

    A_overlap, B_overlap = get_pull_on_overlap(A, overlap_files_set), get_pull_on_overlap(B, overlap_files_set)
    code_sim = get_code_sim_new(A, B, repo,overlap_files_set,num1,num2)
    commit_sim = get_commit_sim_vector(A,B)
    location_sim = [location_similarity(get_location(A), get_location(B))] + \
                   [location_similarity(get_location(A_overlap), get_location(B_overlap))]

    '''
    common_words = list(set(get_tokens(A["title"])) & set(get_tokens(B["title"])))
    overlap_title_len = len(common_words)

    if model is not None:
        title_idf_sum = model.get_idf_sum(common_words)
    else:
        title_idf_sum = 0
    '''

    def get_time(t):
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")

    overlap_files_len = len(overlap_files_set)

    ret = {
        'title': title_sim,
        'desc': desc_sim,
        'code': code_sim,
        'file_list': [file_list_sim, overlap_files_len],
        'location': location_sim,
        'commit':commit_sim
        # 'pattern': [pattern],
    }

    if add_conf:
        conf = 0
        for file in overlap_files_set:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if (file_name in A["title"]) and (file_name in B["title"]):
                conf = 1
                break
        ret['conf'] = [conf]

    if add_timedelta:
        delta_time = abs((get_time(A['created_at']) - get_time(B['created_at'])).days)
        ret['time'] = [delta_time]

    return ret

def calc_sim_before_new(A,B,repo,num1,num2):
    pattern = check_pattern(A, B)
    title_sim = get_text_sim(A["title"], B["title"], repo)
    desc_sim = get_text_sim(A["body"], B["body"], repo)
    file_list_sim = list_similarity(get_file_list(A), get_file_list(B))

    if ('merge_commit_flag' in A) and ('merge_commit_flag' in B):
        file_list_sim = set_similarity(get_file_list(A), get_file_list(B))

    overlap_files_set = set(get_file_list(A)) & set(get_file_list(B))

    A_overlap, B_overlap = get_pull_on_overlap(A, overlap_files_set), get_pull_on_overlap(B, overlap_files_set)
    code_sim = get_code_sim_before_new(A, B, repo,overlap_files_set,num1,num2)
    commit_sim = get_commit_sim_vector(A,B)
    location_sim = [location_similarity(get_location(A), get_location(B))] + \
                   [location_similarity(get_location(A_overlap), get_location(B_overlap))]

    '''
    common_words = list(set(get_tokens(A["title"])) & set(get_tokens(B["title"])))
    overlap_title_len = len(common_words)

    if model is not None:
        title_idf_sum = model.get_idf_sum(common_words)
    else:
        title_idf_sum = 0
    '''

    def get_time(t):
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")

    overlap_files_len = len(overlap_files_set)

    ret = {
        'title': title_sim,
        'desc': desc_sim,
        'code': code_sim,
        'file_list': [file_list_sim, overlap_files_len],
        'location': location_sim,
        'commit':commit_sim
        # 'pattern': [pattern],
    }

    if add_conf:
        conf = 0
        for file in overlap_files_set:
            file_name = os.path.splitext(os.path.basename(file))[0]
            if (file_name in A["title"]) and (file_name in B["title"]):
                conf = 1
                break
        ret['conf'] = [conf]

    if add_timedelta:
        delta_time = abs((get_time(A['created_at']) - get_time(B['created_at'])).days)
        ret['time'] = [delta_time]

    return ret

def sim_to_vet(r):
    vet = []
    for v in [r['title'],r['desc'],r['code'],r['file_list'],r['location'], r['pattern']]:
        vet += v

    if add_timedelta:
        vet += r['time']
    
    if add_conf:
        vet += r['conf']
    
    return vet

# pull requests sim
def get_pr_sim(A, B,repo):
    A["file_list"] = fetch_pr_info(A) if not check_large(A) else []
    B["file_list"] = fetch_pr_info(B) if not check_large(B) else []

    A["title"] = str(A["title"] or '')
    A["body"] = str(A["body"] or '')

    B["title"] = str(B["title"] or '')
    B["body"] = str(B["body"] or '')

    return calc_sim(A, B,repo)





def get_pr_sim_new(A, B, repo,num1,num2):
    A["file_list"] = fetch_pr_info(A) if not check_large(A) else []
    B["file_list"] = fetch_pr_info(B) if not check_large(B) else []

    A["title"] = str(A["title"] or '')
    A["body"] = str(A["body"] or '')

    B["title"] = str(B["title"] or '')
    B["body"] = str(B["body"] or '')

    return calc_sim_new(A, B, repo,num1,num2)

def get_pr_sim_before_new(A, B, repo,num1,num2):
    A["file_list"] = fetch_pr_info(A) if not check_large(A) else []
    B["file_list"] = fetch_pr_info(B) if not check_large(B) else []

    A["title"] = str(A["title"] or '')
    A["body"] = str(A["body"] or '')

    B["title"] = str(B["title"] or '')
    B["body"] = str(B["body"] or '')

    return calc_sim_before_new(A, B, repo,num1,num2)
    
def get_pr_sim_vector(A, B,repo):
    return sim_to_vet(get_pr_sim(A, B,repo))

def get_pr_sim_vector_new(A, B,repo,num1,num2):
    return sim_to_vet(get_pr_sim_new(A, B,repo,num1,num2))

def get_pr_sim_vector_before_new(A, B,repo,num1,num2):
    return sim_to_vet(get_pr_sim_before_new(A, B,repo,num1,num2))

def leave_feat(A, B, way):
    r = get_pr_sim(A, B)
    if 'text' in way:
        r['title'] = r['desc'] = []
    elif 'code' in way:
        r['code'] = []
    elif 'file_list' in way:
        r['file_list'] = []
    elif 'location' in way:
        r['location'] = []
    elif 'pattern' in way:
        r['pattern'] = []
    return sim_to_vet(r)
        
    
def old_way(A, B,repo):
    A["title"] = str(A["title"] or '')
    A["body"] = str(A["body"] or '')
    
    B["title"] = str(B["title"] or '')
    B["body"] = str(B["body"] or '')
    save_path = repo.replace("/", "_") + '_allpr'
    code_model = nlp.Model("", save_path)

    return code_model.query_sim_tfidf(get_tokens(A["title"]), get_tokens(B["title"])) + \
           code_model.query_sim_tfidf(get_tokens(A["body"]), get_tokens(B["body"]))

    
# commits sim

def get_commit_sim_vector(A, B):
    Anumber = A['number']
    Bnumber = B['number']
    commitapiA = 'https://api.github.com/repos/elastic/elasticsearch/pulls/%s/commits'%Anumber
    commitapiB = 'https://api.github.com/repos/elastic/elasticsearch/pulls/%s/commits'%Bnumber
    commitA = ''
    commitB = ''
    res = requests.get(commitapiA, headers=headers)
    arr = json.loads(res.text)
    for v in arr:
        commitA += v['commit']['message']
    res = requests.get(commitapiB, headers=headers)
    arr = json.loads(res.text)
    for v in arr:
        commitB += v['commit']['message']
    save_path = 'elastic/elasticsearch'.replace("/", "_") + '_allpr_code'
    code_model = nlp.Model("", save_path)
    sim = code_model.query_sim_tfidf(commitA, commitB)
    return [sim]

    # def commit_to_pull(x):
    #     t = {}
    #     t["number"] = x['sha']
    #     t['title'] = t['body'] = str(x['commit']['message'] or '')
    #     t["file_list"] = fetch_commit(x['url'])
    #     t['commit_flag'] = True
    #     return t
    # ret = calc_sim(commit_to_pull(A), commit_to_pull(B))
    # return sim_to_vet(ret)


