import os
import sys
import matplotlib.pyplot as plt
way = 'gmnstr' #基于字符串的代码相似性计算  ren的方法 + commit
oldway = 'gmnold' #Zhixingli的方法 基准
renway = 'gmnren' # ren的方法
newway = 'gmnnew' # 代码文件处理后的
# beforenewway = 'gmnbeforenew'# 代码文件处理前的
beforenewway = 'gmnbefore'# 代码文件处理前的

if len(sys.argv) == 2:
    way = sys.argv[1].strip()

file = 'evaluation/result_on_topk_%s.txt' % way
file2 = 'evaluation/result_on_topk_%s.txt' % oldway
file3 = 'evaluation/result_on_topk_%s.txt' % newway
file4 = 'evaluation/result_on_topk_%s.txt'% beforenewway
file5 = 'evaluation/result_on_topk_%s.txt'% renway

cnt_num = 0
top_acc = [0 for i in range(20)]
top_acc2 = [0 for i in range(20)]
top_acc3 = [0 for i in range(20)]
top_acc4 = [0 for i in range(20)]
top_acc5 = [0 for i in range(20)]

last_run1 = {}
last_run2 = {}
last_run3 = {}
last_run4 = {}
last_run5 = {}
def if_exists(file,top_acc):
    global cnt_num
    if os.path.exists(file):
        with open(file) as f:
            for t in f.readlines():
                ps = t.replace(',','').replace('[','').replace(']','').split()
                r, n1, n2 = ps[0], ps[1], ps[2]
                li = [int(x) for x in ps[3:]]
                cnt_num += 1
                for i in range(20):
                    if int(n1) in li[:i+1]:
                        top_acc[i] += 1
    return top_acc
                # print('now=', r, cnt_num, 'top1 acc =', 1.0 * top_acc[0] / cnt_num)
top_acc = if_exists(file,top_acc)
print('cnt = ', cnt_num)
y1 = []
for i in range(20):
    t = 1.0 * top_acc[i] / cnt_num
    top_acc[i] = t
    y1.append(t)
    print('%d \t %.4f' % (i+1, t))
cnt_num = 0
top_acc2 = if_exists(file2,top_acc2)
print('cnt2 =',cnt_num)
y2 = []
##draw plot
for i in range(20):
    t = 1.0 * top_acc2[i] / cnt_num
    top_acc2[i] = t
    y2.append(t)
    print('%d \t %.4f' % (i+1, t))
x = [i for i in range(1,21)]

cnt_num = 0
top_acc3 = if_exists(file3,top_acc3)
print('cnt3 = ',cnt_num)
y3=[]
for i in range(20):
    t = 1.0 * top_acc3[i] / cnt_num
    top_acc3[i] = t
    y3.append(t)
    print('%d \t %.4f' % (i+1, t))
x = [i for i in range(1,21)]

cnt_num = 0
top_acc4 = if_exists(file4,top_acc4)
print('cnt4 = ',cnt_num)
y4=[]
for i in range(20):
    t = 1.0 * top_acc4[i] / cnt_num
    top_acc4[i] = t
    y4.append(t)
    print('%d \t %.4f' % (i+1, t))
x = [i for i in range(1,21)]

cnt_num = 0
top_acc5 = if_exists(file5,top_acc5)
print('cnt5 = ',cnt_num)
y5=[]
for i in range(20):
    t = 1.0 * top_acc5[i] / cnt_num
    top_acc5[i] = t
    y5.append(t)
    print('%d \t %.4f' % (i+1, t))
x = [i for i in range(1,21)]

# plt.plot(x,y1,label = "new")
# plt.plot(x,y2,label = "old")
# plt.plot(x,y3,label = "gmnnew")
# plt.plot(x,y4,label = "beforegmnnew")
# plt.xticks([1,5,10,15,20,25,30])
# plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])

plt.plot(x,y1,label = "Ren's method + commit")
plt.plot(x,y2,label = "Li's method")
plt.plot(x,y3,label = "Ren's method + commit + gmn + coderewirte")
plt.plot(x,y4,label = "Ren's method + commit + gmn")
plt.plot(x,y5,label = "Ren's method")
plt.xticks([1,5,10,15,20])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])

plt.xlabel("Size of TopkList")
plt.ylabel("Recall")
plt.legend()
plt.savefig("2.jpg")
plt.show()