import os
import re
import sys
import random
import copy
import json

from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy import array
from multiprocessing import Pool

from util import localfile


from comp import *
from git import *
from clf import *

#split pull request

dataloc = "data/pyclf/new_first_msr_pairs.txt"


def get_all_dict():
    alldict = {}
    with open(dataloc) as f:
        allpr = f.readlines()
    for line in allpr:
        repo,masterPr,followPr = line.strip().split()
        if repo not in alldict:
            alldict[repo] = []
        alldict[repo].append(masterPr)
        alldict[repo].append(followPr)
        pass
    return alldict
    pass

def get_sim_byk(k,textarr):
    sortedarr = sorted(textarr,key=lambda x:x[0],reverse=True)
    res = sortedarr[0:k]
    return res
    pass

def getNdetected(k):
    textsimarr = []
    index = 0
    with open(dataloc) as f:
        allpr = f.readlines()
    all_dict = get_all_dict()
    for line in all_dict:
        repo, masterPr, followPr = line.strip().split()
        #计算这个followPr与该数据集中数据的textsimarr
        for key in all_dict.keys():
            numarr = all_dict[key]
            for num in numarr:
                if followPr == num:
                    continue
                sim = get_sim_wrap((repo,masterPr,followPr))
                textsimarr.append((sim,repo))
            pass
        karr = get_sim_byk(k,textsimarr)
        for arr in karr:
            if arr[1] != repo:
                continue
                pass
            elif arr[1]:
                index = index + 1
            pass
        pass

    return textsimarr

# def get_sim_byk(k):
#     textsimarr = process()
#     #order by k
#     sortedarr = sorted(textsimarr,key = lambda x:x[0],reverse=True)
#     res = sortedarr[0:k]
#     pass

def get_recall_score(size):
    for k in range(1,size+1):
        get_sim_byk(k)
        pass
    pass