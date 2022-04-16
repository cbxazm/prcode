import matplotlib.pyplot as plt

name_list = ['all clues','all-title','all-description','all-commit','all-code','all-file_list']
num_list = [0.712,0.507,0.610,0.652,0.243,0.506]
plt.bar(range(len(name_list)), num_list, tick_label=name_list,width=0.3)
plt.xlabel('Clues')
plt.ylabel('Precision')
plt.savefig('4.jpg')
plt.show()