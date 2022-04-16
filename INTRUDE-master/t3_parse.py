import os
import sys
import matplotlib.pyplot as plt

way = 'new'
oldway = 'old'

if len(sys.argv) == 2:
    way = sys.argv[1].strip()

file = 'evaluation/result_on_topk_%s.txt' % way
file2 = 'evaluation/result_on_topk_%s.txt' % oldway

cnt_num = 0
top_acc = [0 for i in range(30)]
top_acc2 = [0 for i in range(30)]

last_run1 = {}
last_run2 = {}
def if_exists(file,top_acc):
    global cnt_num
    if os.path.exists(file):
        with open(file) as f:
            for t in f.readlines():
                ps = t.replace(',','').replace('[','').replace(']','').split()

                r, n1, n2 = ps[0], ps[1], ps[2]
                li = [int(x) for x in ps[3:]]
                cnt_num += 1
                for i in range(30):
                    if int(n1) in li[:i+1]:
                        top_acc[i] += 1
    return top_acc
                # print('now=', r, cnt_num, 'top1 acc =', 1.0 * top_acc[0] / cnt_num)
top_acc = if_exists(file,top_acc)
print('cnt = ', cnt_num)
y1 = []
for i in range(30):
    t = 1.0 * top_acc[i] / cnt_num
    top_acc[i] = t
    y1.append(t)
    print('%d \t %.4f' % (i+1, t))

cnt_num = 0
top_acc2 = if_exists(file2,top_acc2)
print('cnt2 =',cnt_num)
y2 = []
##draw plot
for i in range(30):
    t = 1.0 * top_acc2[i] / cnt_num
    top_acc2[i] = t
    y2.append(t)
    print('%d \t %.4f' % (i+1, t))
x = [i for i in range(1,31)]

plt.plot(x,y1,label = "new")
plt.plot(x,y2,label = "old")
plt.xticks([1,5,10,15,20,25,30])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])

plt.xlabel("Size of TopkList")
plt.ylabel("Recall")
plt.legend()
plt.show()