import nlp
import os
from util import  localfile
#
# LOCAL_DATA_PATH = '/DATA/cbx'
# save_path = LOCAL_DATA_PATH + '/pr_data/' + 'kubernetes/kubernetes' + '/pull_list.json'
#
#
# arr = localfile.get_file(save_path)

# print(arr)

from git import *

pullA = get_pull("django/django",4419)
get_another_pull(pullA)

print(os.path.exists("/DATA/cbx/model/d.dictionary"))