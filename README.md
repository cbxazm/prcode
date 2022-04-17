# 基于图神经网络的代码相似性检测模型实验

## 数据集

```
1.训练集 /javadata/trainall.txt
2.验证集 /javadata/valid.txt
3.测试集 /javadata/test.txt
```

## 运行文件

/run_java.py 如果有gpu可自行调整运行参数

# 重复Pull Request检测实验

## 数据集

```
数据集均在/data文件夹下面，其中训练集是first开头 测试集是second开头
/data/pyclf 是python项目
/data/clf 是java项目，保存的是变更代码基于字符串的数据
/data/clf/beforegmnclf 是java项目。保存的是变更代码基于第一个实验模型计算的数据，但是变更文件未经过预处理
/data/clf/gmnclf 是java项目。保存的是变更代码基于第一个实验模型计算的数据，变更文件经过预处理
```

## 运行文件

```
1./clf/clf.py 训练基于python项目和java项目的重复Pull request检测模型
2./clf/gmnclf.py 训练java项目，文件重写后
3./clf/beforegmnclf.py 训练java项目，文件重写前
4./rewirtecodefile.py 文件重写
```

