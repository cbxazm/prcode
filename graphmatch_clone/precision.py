import matplotlib.pyplot as plt

name_list = ['all clues','all-title','all-description','all-commit','all-code','all-file_list']
num_list = [0.632,0.467,0.589,0.611,0.341,0.526]
plt.bar(range(len(name_list)), num_list, tick_label=name_list,width=0.3)
plt.xlabel('Clues')
plt.ylabel('Precision')
plt.savefig('3.jpg')
plt.show()