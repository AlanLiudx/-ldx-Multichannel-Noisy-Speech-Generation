#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" simulate train and dev set, use multiprocessing.Pool to accelerate the pipline;
it's not totally random
@author: nwpuykjv@163.com
        arrowhyx@foxmail.com
        ldx20@mails.tsinghua.edu.cn
"""


import numpy as np
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import os
import warnings
import sys
eps = np.finfo(np.float32).eps
import argparse
import multiprocessing as mp
import traceback
import linecache
import random
import pandas as pd
import tqdm
from tqdm import tqdm


clean_tmp_name="./tmp/clean.name"
noise_tmp_name="./tmp/noise.name"
rir_tmp_name="./tmp/rir.name"


#几个.name的条目数
clean_lines_num=217950
noise_lines_num=930
rir_lines_num=3600

fetch_nums=40000 #随机取其中40000行，其中28000训练集，8000验证集，4000测试集
#我们先做实验做一个小数据集,700/200/100
# snr_list=[-5,0,5,10] #snr从四个里选一个
train_nums=28000
valid_nums=8000
test_nums=4000
#直接把nums在这里记录，方便修改
# snr_list=[-5,0,5,10] #snr四选一
snr_list=[-10,-5,0,5] #snr四选一，新增了snr低至-10dB的情景，10dB的情景先舍弃
#改进了噪声生成模型，我们考虑应用信噪比更低的情景

# for index in range(fetch_nums):
#     rand_num=random.randint(1,clean_lines_num)
#     #测试随机选取其中一行，注意randint函数俩参数是闭区间
#     line_read=linecache.getline(clean_tmp_name,rand_num).strip()
#     # print("we randomly choose one line, the index is:",rand_num)
#     # print("line read:",line_read)


#之后我们考虑通过pandas写一个csv文件，csv的表头各列分别是{clean,noise,rir,snr,scale},每一行 对应一个混合数据
df_train=pd.DataFrame(columns=['clean','noise','rir','snr','scale','arrayshape'])
df_valid=pd.DataFrame(columns=['clean','noise','rir','snr','scale','arrayshape'])
df_test=pd.DataFrame(columns=['clean','noise','rir','snr','scale','arrayshape'])
#上面是先创建一个空的dataframe，共有三列分别为clean，noise，rir
tqdm_bar=tqdm(range(fetch_nums))
for index in tqdm_bar:
    tqdm_bar.set_description("generate lists:data num:{}".format(fetch_nums))
    rand_clean=random.randint(1,clean_lines_num)
    rand_noise=random.randint(1,noise_lines_num)
    rand_rir=random.randint(1,rir_lines_num)
    #测试随机选取其中一行，注意randint函数俩参数是闭区间
    clean_read=linecache.getline(clean_tmp_name,rand_clean).strip()
    noise_read=linecache.getline(noise_tmp_name,rand_noise).strip()
    rir_read=linecache.getline(rir_tmp_name,rand_rir).strip()
    snr=random.choice(snr_list) #从几个给定的snr值中随机选择一个
    scale=random.uniform(0.2,0.9) #从0.2-0.9的均匀分布scale值选一个
    a=rir_read.split("/")
    arrayshape=a[-2] #按照/分开，然后取倒数第二组
    if(index<train_nums):
        df_train.loc[index]=[clean_read,noise_read,rir_read,snr,scale,arrayshape] #注意新增行应该用loc，使用iloc不能扩大表的值，只能修改原有行的值
    elif(index>=train_nums and index<(train_nums+valid_nums)):
        df_valid.loc[index-train_nums]=[clean_read,noise_read,rir_read,snr,scale,arrayshape]
    elif(index>=(train_nums+valid_nums) and index<fetch_nums):
        df_test.loc[index-(train_nums+valid_nums)]=[clean_read,noise_read,rir_read,snr,scale,arrayshape]
    # df_temp=pd.DataFrame([[clean_read,noise_read,rir_read]],columns=['clean','noise','rir']) #追加一行，这部分是正常的
    # df_empty.append(df_temp,ignore_index=False) #追加一行，append到空的后面无法正常工作，全是空的
    # print("index is:",index)
    # print("dftemp:")
    # print(df_temp)
    # print("dfempty:")
    # print(df_empty)
    # print("we randomly choose one line, the index is:",rand_num)
    # print("line read:",line_read)

    #
df_train.to_csv(path_or_buf='./tmp/train_lst.csv',sep=',',na_rep='None',header=True,index=False)
df_valid.to_csv(path_or_buf='./tmp/valid_lst.csv',sep=',',na_rep='None',header=True,index=False)
df_test.to_csv(path_or_buf='./tmp/test_lst.csv',sep=',',na_rep='None',header=True,index=False)