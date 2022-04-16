import os

data_folder = 'data/clf'

dataset = [
    data_folder + '/first_msr_pairs.txt',
    data_folder + '/second_msr_pairs.txt',
    data_folder + '/first_nondup.txt',
    data_folder + '/second_nondup.txt'
]


def process(data):
    with open(data) as f:
        all_pr = f.readlines()
        for curstr in all_pr:
             if curstr.__contains__("saltstack") or curstr.__contains__("scikit-learn"):
                 with open("data/newclf/new_first_nondup.txt", 'a+') as write_file:
                     write_file.write(curstr)
                     pass
                 pass
             pass
        pass
    pass

# for s in dataset:
process("data/clf/first_nondup.txt")
    # pass

# curstr = "ccbbaa"

# print(curstr.__contains__("ca"))

