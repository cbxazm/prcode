import os
from util import localfile
import requests
import time
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    # 'Authorization':'token '+'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'
    'Authorization':'token '+'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'
}
import json
data_folder = 'data/pyclf'
dataset = [
    data_folder + '/new_first_msr_pairs.txt',
    data_folder + '/new_second_msr_pairs.txt',
    data_folder + '/new_first_nondup.txt',
    data_folder + '/new_second_nondup.txt'
]
LOCAL_DATA_PATH = '/DATA/newcbx'

def deal(savepath,repo,num):
    if os.path.exists(savepath):
      pulljson = localfile.get_file(savepath)
      if "title" not in pulljson and "body" not in pulljson:
          # 重新写
          url = 'https://api.github.com/repos/%s/pulls/%s' % (repo, num)
          try:
              res = requests.get(url)
              r = res.text
              # r = api.get('repos/%s/pulls/%s' % (repo, num))
              r = json.loads(res.text)
              # time.sleep(1.0)
          except:
              print("重发请求" + str(url))
              time.sleep(10)
              res = requests.get(url, headers=headers)
              r = res.text
              # r = api.get('repos/%s/pulls/%s' % (repo, num))
              r = json.loads(res.text)
          localfile.write_to_file(savepath, r)
          return r

for data in dataset:
    p = {}
    with open(data) as f:
        all_pr = f.readlines()
        pass
    for l in all_pr:
        r, n1, n2 = l.strip().split()

        # if 'msr_pairs' not in data:
        #     if check_large(get_pull(r, n1)) or check_large(get_pull(r, n2)):
        #         continue

        if r not in p:
            p[r] = []
        p[r].append((n1, n2))

    print('all=', len(all_pr))

    for r in p:
        for z in p[r]:
            deal(LOCAL_DATA_PATH + '/pr_data/%s/%s/api.json' % (r, z[0]),r,z[0])
            deal(LOCAL_DATA_PATH + '/pr_data/%s/%s/api.json' % (r, z[1]),r,z[1])
            pass
        pass
