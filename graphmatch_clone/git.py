import time
import os
import re
import requests
import json
import fetch_raw_diff

from flask import Flask
# from flask_github import GitHub

from util import localfile

app = Flask(__name__)

app.config['GITHUB_CLIENT_ID'] = os.environ.get('GITHUB_CLIENT_ID')
app.config['GITHUB_CLIENT_SECRET'] = os.environ.get('GITHUB_CLIENT_SECRET')
# app.config['GITHUB_CLIENT_ID'] = 'd0c1d996ad5ffb3b9cf5'
# app.config['GITHUB_CLIENT_SECRET'] = '71576236f34b0fb02ef3b04b5538a1df719896ce'
app.config['GITHUB_BASE_URL'] = 'https://api.github.com/'
app.config['GITHUB_AUTH_URL'] = 'https://github.com/login/oauth/'

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    # 'Authorization':'token '+'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'
    'Authorization':'token '+'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'
}
# datafolder = 'data/clf'
datafolder = 'data/pyclf'
# dataloc = [datafolder + '/first_msr_pairs.txt',datafolder + '/second_msr_pairs.txt',datafolder + '/first_nondup.txt',datafolder + '/second_nondup.txt']
dataloc = [datafolder + '/new_first_msr_pairs.txt',datafolder + '/new_second_msr_pairs.txt',datafolder + '/new_first_nondup.txt',datafolder + '/new_second_nondup.txt']


# LOCAL_DATA_PATH = '/DATA/cbx'
LOCAL_DATA_PATH = '/DATA/newcbx'
COMMON_PREFIX = 'https://api.github.com/'

# api = GitHub(app)
# @api.access_token_getter
# def token_getter():
#     access_token = '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'
#     return access_token

def text2list_precheck(func):
    def proxy(text):
        if text is None:
            return []
        ret = func(text)
        return ret
    return proxy

@text2list_precheck
def get_numbers(text):
    nums = list(filter(lambda x: len(x) >= 3, re.findall('([0-9]+)', text)))
    nums = list(set(nums))
    return nums

@text2list_precheck
def get_version_numbers(text):
    nums = [''.join(x) for x in re.findall('(\d+\.)?(\d+\.)(\d+)', text)]
    nums = list(set(nums))
    return nums

@text2list_precheck
def get_pr_and_issue_numbers(text):
    nums = []
    nums += re.findall('#([0-9]+)', text)
    nums += re.findall('pull\/([0-9]+)', text)
    nums += re.findall('issues\/([0-9]+)', text)
    nums = list(filter(lambda x: len(x) > 0, nums))
    nums = list(set(nums))
    return nums


def check_too_big(pull):
    if not ("changed_files" in pull):
        pull = get_pull(pull["base"]["repo"]["full_name"], pull["number"])

    if not ("changed_files" in pull):
        pull = get_pull(pull["base"]["repo"]["full_name"], pull["number"], True)

    if pull["changed_files"] > 50:
        return True
    if (pull["additions"] >= 10000) or (pull["deletions"] >= 10000):
        return True
    return False

check_large_cache = {}
def check_large(pull):
    global check_large_cache
    if "base" not in pull:
        print("occour problem" + str(pull))
        return True
        pass
    index = (pull["base"]["repo"]["full_name"], pull["number"])
    if index in check_large_cache:
        return check_large_cache[index]

    check_large_cache[index] = True # defalue true

    if check_too_big(pull):
        return True

    try:
        l = len(fetch_pr_info(pull))
    except Exception as e:
        if 'too big' in str(e):
            return True

    '''
    if l == 0:
        try:
            file_list = fetch_file_list(pull, True)
        except:
            path = '/DATA/luyao/pr_data/%s/%s' % (pull["base"]["repo"]["full_name"], pull["number"])
            flag_path = path + '/too_large_flag.json'
            localfile.write_to_file(flag_path, 'flag')
            print('too big', pull['html_url'])
            return True
    '''

    path = '/DATA/cbx/pr_data/%s/%s/raw_diff.json' % (pull["base"]["repo"]["full_name"], pull["number"])
    if os.path.exists(path) and (os.path.getsize(path) >= 50 * 1024):
        return True

    check_large_cache[index] = False
    return False

'''
def fresh_pr_info(pull):
    file_list = fetch_file_list(pull)
    path = '/DATA/luyao/pr_data/%s/%s' % (pull["base"]["repo"]["full_name"], pull["number"])
    parse_diff_path = path + '/parse_diff.json'
    localfile.write_to_file(parse_diff_path, file_list)
'''

file_list_cache = {}

def fetch_pr_info(pull, must_in_local = False):
    global file_list_cache
    print("正在遍历"+str(pull))
    if "base" not in pull:
        print("occour problem"+ str(pull))
        pass
    ind = (pull["base"]["repo"]["full_name"], pull["number"])
    if ind in file_list_cache:
        return file_list_cache[ind]

    # path = '/DATA/cbx/pr_data/%s/%s' % (pull["base"]["repo"]["full_name"], pull["number"])
    path = '/DATA/newcbx/pr_data/%s/%s' % (pull["base"]["repo"]["full_name"], pull["number"])
    parse_diff_path = path + '/parse_diff.json'
    raw_diff_path = path + '/raw_diff.json'
    pull_files_path = path + '/pull_files.json'


    flag_path = path + '/too_large_flag.json'
    if os.path.exists(flag_path):
        raise Exception('too big', pull['html_url'])

    if os.path.exists(parse_diff_path):
        try:
            ret = localfile.get_file(parse_diff_path)
            file_list_cache[ind] = ret
            return ret
        except:
            pass

    if os.path.exists(raw_diff_path) or os.path.exists(pull_files_path):
        if os.path.exists(raw_diff_path):
            file_list = localfile.get_file(raw_diff_path)
        elif os.path.exists(pull_files_path):
            pull_files = localfile.get_file(pull_files_path)
            # file_list = [parse_diff(file["file_full_name"], file["changed_code"]) for file in pull_files]
        else:
            raise Exception('error on fetch local file %s' % path)
    else:
        if must_in_local:
            raise Exception('not found in local')

        try:
            file_list = fetch_file_list(pull)
        except:
            localfile.write_to_file(flag_path, 'flag')
            raise Exception('too big', pull['html_url'])

    # print(path, [x["name"] for x in file_list])
    localfile.write_to_file(parse_diff_path, file_list)
    file_list_cache[ind] = file_list
    return file_list


# -------------------About Repo--------------------------------------------------------
def getpageNumber(url):
    arr = url.split("&")
    str = arr[len(arr)-1]
    number = int(str[5:])
    return number
    pass

def reget(urlstr,index,curpath):
        res = requests.get(urlstr, headers=headers)
        print("准备获取" + urlstr)
        print("正在获取" + urlstr)
        print(index)
        index = index + 1
        arr = json.loads(res.text)
        return arr

def get_repo_info(repo,type, renew=False):
    save_path = LOCAL_DATA_PATH + '/pr_data/' + repo + '/%s_list.json' % type
    if type == 'fork':
        save_path = LOCAL_DATA_PATH + '/result/' + repo + '/forks_list.json'

    if (os.path.exists(save_path)) and (not renew):
        try:
            return localfile.get_file(save_path)
        except:
            pass

    print('start fetch new list for ', repo, type)
    if (type == 'pull') or (type == 'issue'):
        print('repos/%s/%ss?state=closed' % (repo, type))
        # str = COMMON_PREFIX+'repos/%s/%ss?state=all' % (repo, type)
        # ret = requests.get(str,headers = headers).text
        ret = []
        urlstr = COMMON_PREFIX+'repos/%s/%ss?state=all' % (repo, type)
        res = requests.get(urlstr,headers = headers)
        link = res.links
        pageNumber = getpageNumber(link['last']['url'])
        pageNumber = pageNumber if pageNumber<16 else 16
        # pageNumber = 5
        print('get repo info by page',link['last']['url'])
        arr = json.loads(res.text)
        ret.extend(arr)
        for num in range(2,pageNumber+1):
            curstr = urlstr + '&page=' + str(num)
            try:
                print(num)
                res = requests.get(curstr, headers=headers)
                arr = json.loads(res.text)
                pass
            except:
                print("occur exception num {}".format(num))
                time.sleep(10)
                res = requests.get(str, headers=headers)
                arr = json.loads(res.text)
                pass
            ret.extend(arr)
            pass
        for s in dataloc:
            print("正在遍历{}".format(s))
            # datapath = "/data/cbx/other"
            datapath = "/data/newcbx/other"
            index = 1
            with open(s) as f:
                all_pr = f.readlines()
                pass
            for l in all_pr:
                r,n1,n2 = l.strip().split()
                if r == repo:
                    urlstr = COMMON_PREFIX + 'repos/%s/%ss/' % (repo, type) + str(n1)
                    curpath = datapath + "/" + repo + "/" + "%s.json" % n1
                    if os.path.exists(curpath):
                        print("加载本地文件")
                        ret.append(localfile.get_file(curpath))
                        pass
                    else:
                        try:
                            res = requests.get(urlstr, headers=headers)
                            print("准备获取" + urlstr)
                            if res.status_code == 200:
                                print("正在获取" + urlstr)
                                print(index)
                                index = index + 1
                                arr = json.loads(res.text)
                                ret.append(arr)
                                localfile.write_to_file(curpath, arr)
                            else:
                                print("没有获取到数据,重新发送请求"+str(curstr))
                                time.sleep(10)
                                arr = reget(urlstr,index,curpath)
                                ret.append(arr)
                                localfile.write_to_file(curpath, arr)
                            pass
                        except:
                            print("http请求发生问题，重发请求"+urlstr)
                            time.sleep(10)
                            res = requests.get(urlstr, headers=headers)
                            print("准备获取" + urlstr)
                            if res.status_code == 200:
                                print("正在获取" + urlstr)
                                print(index)
                                index = index + 1
                                arr = json.loads(res.text)
                                ret.append(arr)
                                localfile.write_to_file(curpath, arr)
                            else:
                                print("没有获取到数据,重新发送请求" + str(curstr))
                                time.sleep(10)
                                arr = reget(urlstr, index, curpath)
                                ret.append(arr)
                                localfile.write_to_file(curpath, arr)
                            pass
                            pass
                    urlstr = COMMON_PREFIX + 'repos/%s/%ss/' % (repo, type) + str(n2)
                    curpath2 = datapath + "/" + repo + "/" + "%s.json" % n2
                    if os.path.exists(curpath2):
                        print("加载本地文件")
                        ret.append(localfile.get_file(curpath2))
                        pass
                    else:
                      try:
                        res = requests.get(urlstr, headers = headers)
                        if res.status_code == 200:
                            print("正在获取" + urlstr)
                            print(index)
                            index = index + 1
                            arr = json.loads(res.text)
                            ret.append(arr)
                            localfile.write_to_file(curpath2, arr)
                            pass
                        else:
                            print("没有获取到数据")
                            pass
                      except:
                          print("http请求发生问题，重发请求" + urlstr)
                          time.sleep(10)
                          res = requests.get(urlstr, headers=headers)
                          if res.status_code == 200:
                              print("正在获取" + urlstr)
                              print(index)
                              index = index + 1
                              arr = json.loads(res.text)
                              ret.append(arr)
                              localfile.write_to_file(curpath2, arr)
                              pass
                          else:
                              print("没有获取到数据")
                              pass


    else:
        if type == 'branch':
            type = 'branche'
        # ret = api.request('GET', 'repos/%s/%ss' % (repo, type), True)

    localfile.write_to_file(save_path, ret)
    return ret

def fetch_commit(url, renew=False):
    save_path = LOCAL_DATA_PATH + '/pr_data/%s.json' % url.replace('https://api.github.com/repos/','')
    if os.path.exists(save_path) and (not renew):
        try:
            return localfile.get_file(save_path)
        except:
            pass

    # c = api.get(url)
    # time.sleep(0.7)
    # file_list = []
    # for f in c['files']:
    #     if 'patch' in f:
    #         file_list.append(fetch_raw_diff.parse_diff(f['filename'], f['patch']))
    # localfile.write_to_file(save_path, file_list)
    # return file_list


# ------------------About Pull Requests----------------------------------------------------

def get_pull(repo, num, renew=False):
    save_path = LOCAL_DATA_PATH + '/pr_data/%s/%s/api.json' % (repo, num)
    if os.path.exists(save_path) and (not renew):
        try:
            return localfile.get_file(save_path)
        except:
            pass
    url = 'https://api.github.com/repos/%s/pulls/%s' % (repo, num)
    try:
        res = requests.get(url,headers = headers)
        r = res.text
        # r = api.get('repos/%s/pulls/%s' % (repo, num))
        r = json.loads(res.text)
        # time.sleep(1.0)
    except:
        print("重发请求"+str(url))
        time.sleep(10)
        res = requests.get(url, headers=headers)
        r = res.text
        # r = api.get('repos/%s/pulls/%s' % (repo, num))
        r = json.loads(res.text)
    localfile.write_to_file(save_path, r)
    return r

# def get_pull_commit(pull, renew=False):
#     save_path = LOCAL_DATA_PATH + '/pr_data/%s/%s/commits.json' % (pull["base"]["repo"]["full_name"], pull["number"])
#     if os.path.exists(save_path) and (not renew):
#         try:
#             return localfile.get_file(save_path)
#         except:
#             pass
#     # commits = api.request('GET', pull['commits_url'], True)
#     time.sleep(0.7)
#     localfile.write_to_file(save_path, commits)
#     return commits

def get_another_pull(pull, renew=False):
    save_path = LOCAL_DATA_PATH + '/pr_data/%s/%s/another_pull.json' % (pull["base"]["repo"]["full_name"], pull["number"])
    if os.path.exists(save_path) and (not renew):
        try:
            return localfile.get_file(save_path)
        except:
            pass

    comments_href = pull["_links"]["comments"]["href"]
    # comments = api.request('GET', comments_href, True)
    res = requests.get(comments_href,headers = headers)
    time.sleep(0.7)
    comments = json.loads(res.text)
    candidates = []
    for comment in comments:
        candidates.extend(get_pr_and_issue_numbers(comment["body"]))
    candidates.extend(get_pr_and_issue_numbers(pull["body"]))

    result = list(set(candidates))

    localfile.write_to_file(save_path, result)
    return result

def fetch_file_list(pull, renew=False):
    repo, num = pull["base"]["repo"]["full_name"], str(pull["number"])
    save_path = LOCAL_DATA_PATH + '/pr_data/' + repo + '/' + num + '/raw_diff.json'

    if os.path.exists(save_path) and (not renew):
        try:
            return localfile.get_file(save_path)
        except:
            pass

    # t = api.get('repos/%s/pulls/%s/files?page=3' % (repo, num))
    urlstr = COMMON_PREFIX +'repos/%s/pulls/%s/files?page=3' % (repo, num)
    res = requests.get(urlstr,headers = headers)
    t = json.loads(res.text)
    file_list = []
    if len(t) > 0:
        raise Exception('too big', pull['html_url'])
    else:
        # li = api.request('GET', 'repos/%s/pulls/%s/files' % (repo, num), True)
        urlstr = COMMON_PREFIX +'repos/%s/pulls/%s/files' % (repo, num)
        # time.sleep(0.8)
        res = requests.get(urlstr,headers = headers)
        li = json.loads(res.text)
        for f in li:
            if f.get('changes', 0) <= 5000 and ('filename' in f) and ('patch' in f):
                file_list.append(fetch_raw_diff.parse_diff(f['filename'], f['patch']))

    localfile.write_to_file(save_path, file_list)
    return file_list


# pull_commit_sha_cache = {}
# def pull_commit_sha(p):
#     index = (p["base"]["repo"]["full_name"], p["number"])
#     if index in pull_commit_sha_cache:
#         return pull_commit_sha_cache[index]
#     c = get_pull_commit(p)
#     ret = [(x["sha"], x["commit"]["author"]["name"]) for x in list(filter(lambda x: x["commit"]["author"] is not None, c))]
#     pull_commit_sha_cache[index] = ret
#     return ret

# ------------------Data Pre Collection----------------------------------------------------
def run_and_save(repo, skip_big=False):
    repo = repo.strip()

    skip_exist = True

    pulls = get_repo_info(repo, 'pull', True)

    for pull in pulls:
        num = str(pull["number"])
        pull_dir = LOCAL_DATA_PATH + '/pr_data/' + repo + '/' + num

        pull = get_pull(repo, num)

        if skip_big and check_too_big(pull):
            continue

        if skip_exist and os.path.exists(pull_dir + '/raw_diff.json'):
            continue

        fetch_file_list(repo, pull)

        print('finish on', repo, num)


if __name__ == "__main__":
    #r = get_pull('angular/angular.js', '16629', 1)
    #print(r['changed_files'])
    # get_pull_commit(get_pull('ArduPilot/ardupilot', '8008'))

    print(len(get_repo_info('FancyCoder0/INFOX', 'fork', True)))
    print(len(get_repo_info('FancyCoder0/INFOX', 'pull', True)))
    print(len(get_repo_info('FancyCoder0/INFOX', 'issue', True)))
    print(len(get_repo_info('FancyCoder0/INFOX', 'commit', True)))
    print(len(get_repo_info('tensorflow/tensorflow', 'branch', True)))

    print(len(fetch_file_list(get_pull('FancyCoder0/INFOX', '113', True))))
    # print(get_another_pull(get_pull('facebook/react', '12503'), True))
    # print([x['commit']['message'] for x in get_pull_commit(get_pull('facebook/react', '12503'),True)])
