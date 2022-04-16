import javalang
import requests
import json
import re
import os
import time
from util import  localfile
COMMON_PREFIX = 'https://api.github.com/'
repo = "elastic/elasticsearch"
LOCALFILE_PATH = '/data/utf8files/change_files/%s' % repo
WRITE_PATH = '/data/funccbx/change_files/%s' % repo
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    # 'Authorization':'token '+'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'
    'Authorization':'token '+'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' #最新2月份
}

#
def parse_diff(file_name, diff):
    if file_name[-5:]!=".java":
        return
    parts = re.split('(@@.*?-.*?\+.*?@@)', diff)
    newparts = []
    #get likely funcName from str
    for partsvalue in parts:
        curstr = partsvalue.strip()
        if curstr.startswith("public") or curstr.startswith("private") or curstr.startswith("protected") or curstr.startswith("deafult"):
            reg = r"\w+\("
            res = re.search(reg, curstr)
            if res!=None and res.group()!=None:
             newparts.append(res.group()[0:len(res.group()) - 1])
            pass
        pass
    if newparts!=[]:
       funcList = list(set(newparts))
       funcList.sort(key=newparts.index)
       return funcList
    else:
        return []

#get true funcNameList by fileName
def getFuncNameByFileName(fileName,num):
    funcNameList = []
    programfile = open(LOCALFILE_PATH + "/"+num+"/"+fileName, encoding='utf-8')
    try:
        programtext = programfile.read()
        # tokenize
        programtokens = javalang.tokenizer.tokenize(programtext)
        # parse code into ast
        programast = javalang.parser.parse(programtokens)
    except:
        with open("newasterr.txt","a+") as f:
            f.write(num+" " + fileName+"\n")
            return []
        pass
    classDeclarationList = programast.children[2]
    for classDeclaration in classDeclarationList:
        className = classDeclaration.name
        methodDeclarationList = classDeclaration.methods
        for methodDeclaration in methodDeclarationList:
            funcNameList.append(methodDeclaration.name)
            pass
        pass
    return funcNameList
    pass

#get code position by funcName
def getCodePositionByFuncName(funcName,fileName,num,totalLine):
    programfile = open(LOCALFILE_PATH + "/" + num + "/" + fileName, encoding='utf-8')
    programtext = programfile.read()
    # tokenize
    programtokens = javalang.tokenizer.tokenize(programtext)
    # parse code into ast
    programast = javalang.parser.parse(programtokens)
    classDeclarationList = programast.children[2]
    for classDeclaration in classDeclarationList:
        className = classDeclaration.name
        methodDeclarationList = classDeclaration.methods
        index = 0
        length = len(methodDeclarationList)
        for methodDeclaration in methodDeclarationList:
            print("this is %s 个 function"% (index+1))
            if funcName == methodDeclaration.name and index!=length-1:
                funcStartLine = methodDeclaration.position[0]
                funcEndLine = methodDeclarationList[index+1].position[0]
                return [funcStartLine,funcEndLine]
                pass
            elif funcName == methodDeclaration.name and index == length-1:
                #lastMethod
                funcStartLine = methodDeclaration.position[0]
                funcEndLine =  totalLine-1
                return [funcStartLine,funcEndLine]
                pass
            else:
                index = index + 1
                continue
                pass
            index = index + 1
        return [-1,-1]
    pass
#rewrite change_files
def rewritefilebyfunc(codelineList,funcList,fileName,num):
    filePath = WRITE_PATH + "/"+num+"/"+fileName
    classArr = fileName.split("/")
    className = classArr[len(classArr)-1][0:-5]
    localfile.write_className_to_file(filePath,className)
    funcNameList = getFuncNameByFileName(fileName,num)
    for funcName in funcList:
      #verify funcName is valid
      if funcName not in funcNameList:
          continue
          pass
      else:
          print(funcName)
          #get code position by funcName
          startLine, endLine = getCodePositionByFuncName(funcName,fileName,num,codelineList.__len__())
          if startLine == -1 and endLine == -1:
              continue
          localfile.write_func_to_file(filePath,codelineList,startLine,endLine,funcName)
      pass
    localfile.wirte_end_tofile(filePath)

def rewriteByRepoAndNumber(repo,num):
    urlStr = COMMON_PREFIX + 'repos/%s/pulls/%s/files' % (repo, num)
    try:
        res = requests.get(urlStr, headers=headers)
        li = json.loads(res.text)
        for f in li:
            fileName = f["filename"]
            # if os.path.exists(WRITE_PATH + "/"+num+"/"+fileName):
            #     continue
            if f["filename"][-5:]!=".java":
                continue
            ##get localfile by filename
            localFilePath = LOCALFILE_PATH+"/"+num+"/"+fileName
            ##get code line list
            codeLineList = []
            with open(localFilePath,encoding="utf-8") as file:
                codeLineList = file.readlines()
            if f.get('changes', 0) <= 5000 and ('filename' in f) and ('patch' in f):
                resList = parse_diff(f['filename'], f['patch'])
                pass
            else:
                continue
            pass
            funcList = []
            if resList!=[]:
             for value in resList:
                if value!=None:
                    funcList.append(value)
            #rewirte file by codelinelist and funclist
            rewritefilebyfunc(codeLineList,funcList,fileName,num)
            pass
    except:
        print("发生异常。。。wait 10s")
        time.sleep(10)
        res = requests.get(urlStr, headers=headers)
        li = json.loads(res.text)
        for f in li:
            fileName = f["filename"]
            if os.path.exists(WRITE_PATH + "/" + num + "/" + fileName):
                continue
            if f["filename"][-5:] != ".java":
                continue
            ##get localfile by filename
            localFilePath = LOCALFILE_PATH + "/" + num + "/" + fileName
            ##get code line list
            codeLineList = []
            if os.path.exists(localFilePath):
                with open(localFilePath) as file:
                    codeLineList = file.readlines()
            else:
                continue
            if f.get('changes', 0) <= 5000 and ('filename' in f) and ('patch' in f):
                resList = parse_diff(f['filename'], f['patch'])
                pass
            else:
                continue
            pass
            funcList = []
            if resList != []:
                for value in resList:
                    if value != None:
                        funcList.append(value)
            # rewirte file by codelinelist and funclist
            rewritefilebyfunc(codeLineList, funcList, fileName, num)
            pass

count = 1
arr = []
with open("allreadydone.txt") as h:
    arr = h.readlines()
    arr = [v.replace("\n","") for v in arr]
    pass
for rt, dirs, files in os.walk("D:/DATA/utf8files/change_files/%s" % repo):
   for number in dirs:
        if number in arr:
            continue
            pass
        print("正在处理"+number+"==>第"+str(count)+"个")
        rewriteByRepoAndNumber(repo,number)
        count = count+1
        with open("allreadydone.txt","a+") as f:
            f.write(number+"\n")
        pass
   print("done")
   break

# 10923 10854

rewriteByRepoAndNumber(repo,'1464')