import os
import json


def wirte_end_tofile(filePath):
    path = os.path.dirname(filePath)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(filePath, "a+",encoding="utf-8") as file:
        file.write("}")
    print('finish write last %s ....' % file)
    pass

def write_func_to_file(filePath,codeLineList,startLine,endLine,funcName):
    path = os.path.dirname(filePath)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(filePath, "a+",encoding="utf-8") as file:
        for linenum in range(startLine,endLine):
            wirtecontent = codeLineList[linenum-1]
            if wirtecontent.strip().startswith("/**") or wirtecontent.strip().startswith("/*") or wirtecontent.strip().startswith("*") or wirtecontent.strip().startswith("@") or wirtecontent.strip().startswith("//"):
                continue
            file.write(wirtecontent+"\n")
            pass
        pass
    print('finish write func  %s ....' % funcName)
    pass
def write_className_to_file(filePath,className):
    path = os.path.dirname(filePath)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(filePath, "a+",encoding="utf-8") as file:
        file.write("public class %s {"%className+"\n")
        pass
    print('finish write class %s .....' % className)
def write_to_file2(file,obj):
    path = os.path.dirname(file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file, "a+",encoding="utf-8") as file:
        for char in obj:
            if char == "\n\n" or char == "\n":
                file.write("\n")
                pass
            else:
                file.write(char)
            pass
        pass
    print('finish write %s to file....' % file)
def write_to_file(file, obj):
    """ Write the obj as json to file.
    It will overwrite the file if it exist
    It will create the folder if it doesn't exist.
    Args:
        file: the file's path, like : ./tmp/INFOX/repo_info.json
        obj: the instance to be written into file (can be list, dict)
    Return:
        none
    """
    path = os.path.dirname(file)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(file, 'w') as write_file:
        write_file.write(json.dumps(obj))
    print('finish write %s to file....' % file)

def get_file(path):
    if os.path.exists(path):
        with open(path) as f:
            result = json.load(f)
        return result
    else:
        raise Exception('no such file %s' % path)

def try_get_file(path):
    if os.path.exists(path):
        try:
            return get_file(path)
        except:
            return None
    return None
    
