import requests
from util import localfile
import json
import time
import os
#https://raw.githubusercontent.com/elastic/elasticsearch/29aa602090d4c749cf9c2aeceaf1fd2cecf4c0a4/src/main/java/org/elasticsearch/action/ActionModule.java
url = 'https://api.github.com/repos/elastic/elasticsearch/pulls/'
downloadurl = 'https://raw.githubusercontent.com/elastic/elasticsearch/'
datapath = '/data/gmncbx/change_files/elastic/elasticsearch'
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    # 'Authorization':'token '+'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'
    'Authorization':'token '+'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'
}

def getinfobynumber(number):
    urlstr = url+number+"/files"
    try:
        res = requests.get(urlstr, headers=headers)
        print("准备获取" + urlstr)
        if res.status_code == 200:
            print("正在获取" + urlstr)
            arr = json.loads(res.text)
            for dic in arr:
                filename = dic['filename']
                bloburl = dic['blob_url']
                bloburl = bloburl['https://github.com/elastic/elasticsearch/blob/'.__len__():]
                len = str(filename).__len__()+1
                bloburl = bloburl[0:-len]
                downloadurl = 'https://raw.githubusercontent.com/elastic/elasticsearch/'+bloburl+"/"+str(filename)
                filepath = '/data/redownloadfile/change_files/elastic/elasticsearch/'+str(number)+"/"+filename
                with open("errinfo.txt","a+") as f:
                    f.write(filepath+" "+downloadurl)
                    f.write("\n")
    except:
        time.sleep(10)
        res = requests.get(urlstr, headers=headers)
        print("准备获取" + urlstr)
        if res.status_code == 200:
            print("正在获取" + urlstr)
            arr = json.loads(res.text)
            for dic in arr:
                filename = dic['filename']
                bloburl = dic['blob_url']
                bloburl = bloburl['https://github.com/elastic/elasticsearch/blob/'.__len__():]
                len = str(filename).__len__() + 1
                bloburl = bloburl[0:-len]
                downloadurl = 'https://raw.githubusercontent.com/elastic/elasticsearch/' + bloburl + "/" + str(filename)
                filepath = '/data/redownloadfile/change_files/elastic/elasticsearch/' + str(number) + "/" + filename
                with open("errinfo.txt","a+") as f:
                    f.write(filepath+" "+downloadurl)
                    f.write("\n")
        pass


# for rt, dirs, files in os.walk("D:/DATA/gmncbx/change_files/elastic/elasticsearch"):
#     for number in dirs:
#      print(number)
#     pass
#

with open("asterr.txt") as f:
    arr = f.readlines()
    index = 1
    for info in arr:
        num,path = info.split(" ")
        getinfobynumber(num)
        print(index)
        index = index + 1
        pass