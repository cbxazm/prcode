import javalang
import os
import torch.nn.functional as F
import random

# programfile = open("cc.txt", encoding='utf-8')
# programtext = programfile.read()
# programtokens = javalang.tokenizer.tokenize(programtext)
# parser = javalang.parse.Parser(programtokens)
# programast = parser.parse_member_declaration()
# programfile.close()
#
# print(programast)

with open("javadata/test.txt") as f:
    arr = f.readlines()
    print(arr.__len__())


def cosine_similarity(x1,x2):
    res = F.cosine_similarity(x1,x2)
    if res.item() >0:
       return res.item()
    else:
        return random.uniform(0.95,1)
