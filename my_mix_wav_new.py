#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" simulate train and dev set, use multiprocessing.Pool to accelerate the pipline;
it's not totally random
@author: nwpuykjv@163.com
        arrowhyx@foxmail.com
"""

from tqdm import tqdm
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
import pandas as pd
import json
# import gpuRIR

# clean_lines_num=217950
# noise_lines_num=930
# rir_lines_num=3600
# clean_tmp_name="/data/liudx/ConferenceSpeechsimulation/tmp/clean.name"
# noise_tmp_name="/data/liudx/ConferenceSpeechsimulation/tmp/noise.name"
# rir_tmp_name="/data/liudx/ConferenceSpeechsimulation/tmp/rir.name"
#clean,noise,rir源的行数（可选数量），以及对应的列表位置（分别是每个可选文件的完整路径），后面可能会根据实际情况再修改
def audioread(path, fs=16000):
    #读取全通道音频
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L x C or L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data

def get_firstchannel_read(path, fs=16000):
    '''
    args
        path: wav path
        fs: sample rate
    return
        wave_data: L
    '''
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data
#上面俩是读取全部通道，和读取第一个通道
# def clip_data(data, start, segment_length):
#     '''
#     according the start point and segment_length to split the data
#     args:
#         data: numpy.array
#         start: -2, -1, [0,...., L - 1]
#         segment_length: int
#     return:
#         tgt: numpy.array
#     '''
#     tgt = np.zeros(segment_length)
#     data_len = data.shape[0]
#     if start == -2:
#         """
#         this means segment_length // 4 < data_len < segment_length // 2
#         padding to A_A_A
#         """
#         if data_len < segment_length//3:
#             data = np.pad(data, [0, segment_length//3 - data_len], 'constant')
#             tgt[:segment_length//3] += data
#             st = segment_length//3
#             tgt[st:st+data.shape[0]] += data
#             st = segment_length//3 * 2
#             tgt[st:st+data.shape[0]] += data
        
#         else:
#             """
#             padding to A_A
#             """
#             # st = (segment_length//2 - data_len) % 101
#             # tgt[st:st+data_len] += data
#             # st = segment_length//2 + (segment_length - data_len) % 173
#             # tgt[st:st+data_len] += data
#             data = np.pad(data, [0, segment_length//2 - data_len], 'constant')
#             tgt[:segment_length//2] += data
#             st = segment_length//2
#             tgt[st:st+data.shape[0]] += data
    
#     elif start == -1:
#         '''
#         this means segment_length < data_len*2
#         padding to A_A
#         '''
#         if data_len % 4 == 0:
#             tgt[:data_len] += data
#             tgt[data_len:] += data[:segment_length-data_len]
#         elif data_len % 4 == 1:
#             tgt[:data_len] += data
#         elif data_len % 4 == 2:
#             tgt[-data_len:] += data
#         elif data_len % 4 == 3:
#             tgt[(segment_length-data_len)//2:(segment_length-data_len)//2+data_len] += data
    
#     else:
#         tgt += data[start:start+segment_length]
    
#     return tgt

def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy>=low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    '''
    mix clean and noise according to snr
    '''
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
    new_noise = noise * k
    return new_noise

def mix_noise(clean, noise, snr, channels=8):
    '''
    split/pad the noise data and then mix them according to snr
    '''
    clean_length = clean.shape[0]
    noise_length = noise.shape[0]
    st = 0  # choose the first point
    # padding the noise
    if clean_length > noise_length:
        # st = numpy.random.randint(clean_length + 1 - noise_length)
        noise_t = np.zeros([clean_length, channels])
        noise_t[st:st+noise_length] = noise
        noise = noise_t
    # split the noise
    elif clean_length < noise_length:
        # st = numpy.random.randint(noise_length + 1 - clean_length)
        noise = noise[st:st+clean_length]
    
    snr_noise = snr_mix(clean, noise, snr)
    return snr_noise

def add_reverb(cln_wav, rir_wav, channels=8, predelay=50,sample_rate=16000):
    """
    add reverberation
    args:
        cln_wav: L
        rir_wav: L x C
        rir_wav is always [Lr, C] 
        predelay is ms
    return:
        wav_tgt: L x C
    """

    #add reverb的长度问题需要注意
    #我们不需要early reverb了

    rir_len = rir_wav.shape[0]
    # rir_wav_new=np.transpose(rir_wav,(1,0))
    # rir_wav_new=rir_wav_new[np.newaxis,:,:]
    # wav_tgt=gpuRIR.simulateTrajectory(cln_wav,rir_wav_new)
    # wav_tgt=wav_tgt[450:450+cln_wav.shape[0],:]
    # return wav_tgt
    wav_tgt = np.zeros([channels, cln_wav.shape[0] + rir_len-1]) #wav tgt的长度是原始clean wave与rir len长度之和减1，这跟卷积操作有关
    dt = np.argmax(rir_wav, 0).min() #沿着行这个维度算argmax，返回是各列（channel）对应的最大值索引，dt是这索引取个最小值，也就是找的最早出现最大值的位置
    et = dt+(predelay*sample_rate)//1000 
    et_rir = rir_wav[:et]  #取到et的位置，相当于只取rir的前面一部分，也就是early reverb
    wav_early_tgt = np.zeros([channels, cln_wav.shape[0] + et_rir.shape[0]-1])  #相当于取了early reverb的tgt
    for i in range(channels):
        wav_tgt[i] = sps.oaconvolve(cln_wav, rir_wav[:, i]) 
        wav_early_tgt[i] = sps.oaconvolve(cln_wav, et_rir[:, i]) 
        #卷积流程，采用scipy oaconvolve
    # L x C
    wav_tgt = np.transpose(wav_tgt)
    wav_tgt = wav_tgt[:cln_wav.shape[0]] 
    #截断到clean wave的长度
    wav_early_tgt = np.transpose(wav_early_tgt)
    wav_early_tgt = wav_early_tgt[:cln_wav.shape[0]]
    return wav_tgt

def get_one_spk_noise(clean, noise, snr, scale):
    """
    mix clean and noise according to the snr and scale
    args:
        clean: numpy.array, L x C  L is always segment_length
        noise: numpy.array, L' x C
        snr: float
        scale: float
    """
    gen_noise = mix_noise(clean, noise, snr)
    noisy = clean + gen_noise

    max_amp = np.max(np.abs(noisy))
    max_amp = np.maximum(max_amp, eps)
    #这个操作主要是避免max_amp=0的情况,eps应该是内置的一个很小的值
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    noise=noise*noisy_scale
    #相当于同乘一个scale，noisy的幅度最大值限定死
    return noisy, clean,noise, noisy_scale

def generate_data(clean_path, noise_path, rir_path, snr, scale, segment_length=16000*4, channels=8):
    clean = get_firstchannel_read(clean_path)
    # chunk the clean wav into the segment length
    if(len(clean)>segment_length):
        clean=clean[0:segment_length]
    elif(len(clean)<segment_length):
        clean=np.pad(clean,(0,segment_length-len(clean)),'constant')
    # clean = clip_data(clean, strat_time, segment_length)
    noise = get_firstchannel_read(noise_path)
    if(len(noise)>segment_length):
        noise=noise[0:segment_length]
    elif(len(noise)<segment_length):
        noise=np.pad(noise,(0,segment_length-len(noise)),'constant')
    # add linear/circle rir
    rir = audioread(rir_path) 

    L, C = rir.shape
    noise_temp=0
    # linear array rir is [Lr, 16]
    if(channels!=8):
        raise RuntimeError("You need to edit your code, only 8 mics for generation is valid now")
    if(C!=56):
        raise RuntimeError("6 noise sources,the rir channel should be (1+6)*8=56 channels")
    clean_rir = rir[:, :channels]
    #纯净部分卷积
    clean = add_reverb(clean, clean_rir, channels=channels)
    # noise_rir_list=[]
    for i in range(1,7):
        noise_rir=rir[:,i*8:(i+1)*8]
        #噪声卷积rir，我们把6个噪声源卷积后的结果加起来
        noise_temp=noise_temp+add_reverb(noise, noise_rir, channels=channels)
    # if C%channels == 0 and C==2*channels:
    #     clean_rir = rir[:, :channels]
    #     noise_rir = rir[:, channels:]
    #     #这个地方默认了rir文件对应的channels，前一半是clean wave，后一半是noise rir
    # elif C==channels:
    #     warnings.warn("the clean'rir and noise's rir will be same")
    #     clean_rir = rir 
    #     noise_rir = rir
    # # circle array rir is [Lr, 32]
    # #elif C%channels == 0 and C%channels == 0:
    # elif C%(channels*2) == 0:
    #     skip = C//channels//2
    #     clean_rir = rir[:, :C//2:skip]   #every C//channels channels
    #     noise_rir = rir[:, C//2::skip]  #every C//channels channels 
    #     #这操作，相当于我们录制了8个channels，但是我们一共有16+16，所以clean是从0-16隔一个取一个共计取8个，noise同理
    # else:
    #     raise RuntimeError("Can not generate target channels data, please check data or parameters")


#原版

    # clean = add_reverb(clean, clean_rir, channels=channels)
    # noise= add_reverb(noise, noise_rir, channels=channels)
    #卷积完毕
    noise=noise_temp.copy()
    #麦克风底噪在70dB以下（我们用的麦克风自身信噪比大于70dB），可以忽略；空间的弥散噪声不知道如何建模，还是抽象成声源吧
    inputs, labels,noise_return, noisy_scale = get_one_spk_noise(clean, noise, snr, scale)
    #按照scale返回一个混合后的多通道语音，从最终结果来看是为了把early reverb也一起混进去
    return inputs, labels,noise_return

# def preprocess_func(line, segment_length, result):
#     try:
#         path = line.strip()
#         data = get_firstchannel_read(path)
#         length = data.shape[0]

#         if length < segment_length:
#             if length * 2 < segment_length and length * 4 > segment_length:
#                 result.append('{} -2\n'.format(path))
#             elif length * 2 > segment_length:
#                 result.append('{} -1\n'.format(path))
#         else:
#             sample_index = 0
#             while sample_index + segment_length <= length:
#                 result.append('{} {}\n'.format(path, sample_index))
#                 sample_index += segment_length
#             if sample_index < length:
#                 result.append('{} {}\n'.format(path, length - segment_length))
#         #切割一下原始语音的长度
#     except :
#         traceback.print_exc()
        #自己重写一个主函数data_gen_main
def data_gen_main(label,wav_list,save_dir,index_sum):
    if label=='train':
        # index_sum=700 #总条数
        output_dir=os.path.join(save_dir, 'train')
    elif label=='valid':
        # index_sum=200
        output_dir=os.path.join(save_dir, 'valid')
    elif label=='test':
        # index_sum=100
        output_dir=os.path.join(save_dir, 'test')
    #循环体
    df=pd.read_csv(wav_list)
    tqdm_bar=tqdm(range(index_sum))
    for index in tqdm_bar:
        tqdm_bar.set_description("generate {} data".format(label))
        clean_name=df.iat[index,0]
        noise_name=df.iat[index,1]
        rir_name=df.iat[index,2]
        snr=df.iat[index,3]
        scale=df.iat[index,4]
        arrayshape=df.iat[index,5]
        noisy,clean,noise=generate_data(clean_path=clean_name, noise_path=noise_name, rir_path=rir_name, snr=snr, scale=scale, segment_length=16000*4, channels=8)
        rir_name_spilt=rir_name.split("_")
        clean_angle=rir_name_spilt[-3]
        noisy_angle=rir_name_spilt[-2]
        clean_dist=rir_name_spilt[-5]
        noisy_dist=rir_name_spilt[-4]
        direction_info={'arrayshape':arrayshape,"clean_angle":clean_angle,"noisy_angle":noisy_angle,"clean_dist":clean_dist,"noisy_dist":noisy_dist}
        #建立sample文件夹
        if not os.path.isdir(os.path.join(output_dir, ('sample'+str(index+1)))):
            os.mkdir(os.path.join(output_dir, ('sample'+str(index+1))))
        output_position=os.path.join(output_dir, ('sample'+str(index+1)))
        #写文件
        sf.write(os.path.join(output_position, "mixture.wav"), noisy, 16000)
        sf.write(os.path.join(output_position, "speech.wav"), clean, 16000)
        sf.write(os.path.join(output_position, "noise.wav"), noise, 16000)
        json_name=os.path.join(output_position, "direction_info.json")
        with open(json_name,"w") as f:
            json.dump(direction_info,f)
        # sf.write(os.path.join(output_position, "speech_early.wav"), clean_early, 16000)
        #np.max(a)取得是a这个二维数组全局最大值，可以最后以此为依据归一化一下音频幅度，除以max scale再乘以0.9


def main(args):
    train_list_path=args.train_list
    valid_list_path=args.valid_list
    test_list_path=args.test_list
    # clean_chunk_path = args.clean_wav_list+'.{}.duration'.format(args.chunk_len)

    # noise_path = args.noise_wav_list 
    # rir_path = args.rir_wav_list 
    # config_path = args.mix_config_path
    
    save_dir = args.save_dir 
    train_num=args.train_num
    valid_num=args.valid_num
    test_num=args.test_num
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        #检查是否有对应的文件夹，没有就新建一个
    if not os.path.isdir(os.path.join(save_dir, 'train')):
        os.mkdir(os.path.join(save_dir, 'train'))
    if not os.path.isdir(os.path.join(save_dir, 'valid')):
        os.mkdir(os.path.join(save_dir, 'valid'))
    if not os.path.isdir(os.path.join(save_dir, 'test')):
        os.mkdir(os.path.join(save_dir, 'test'))
    # if args.generate_config: 
        #if not os.path.exists(clean_chunk_path):
        # print('LOG: preparing clean start time')
        # # get_clean_chunk(clean_path, clean_chunk_path, chunk=args.chunk_len)

        # print('LOG: preparing mix config')
        # get_mix_config(clean_chunk_path, noise_path, rir_path, config_path)
    
    print('LOG: generating')
    data_gen_main(label='test',wav_list=test_list_path,save_dir=save_dir,index_sum=test_num)
    data_gen_main(label='valid',wav_list=valid_list_path,save_dir=save_dir,index_sum=valid_num)
    data_gen_main(label='train',wav_list=train_list_path,save_dir=save_dir,index_sum=train_num)
    # get_data(config_path, save_dir, chunk=args.chunk_len)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_list',
        type=str,
        # default="/data/liudx/ConferenceSpeechsimulation/tmp/train_lst.csv",
        default="./tmp/train_lst.csv",
        help='the csv of train wav to read([cleanname,noisename,rirname,snr,scale,arrayshape])'
        ) 
     
    parser.add_argument(
        '--valid_list',
        type=str,
        # default="/data/liudx/ConferenceSpeechsimulation/tmp/valid_lst.csv",
        default="./tmp/valid_lst.csv",
        help='the csv of valid wav to read([cleanname,noisename,rirname,snr,scale,arrayshape])'
        ) 

    parser.add_argument(
        '--test_list',
        type=str,
        # default="/data/liudx/ConferenceSpeechsimulation/tmp/test_lst.csv",
        default="./tmp/test_lst.csv",
        help='the csv of test wav to read([cleanname,noisename,rirname,snr,scale,arrayshape])'
        ) 
    
    # parser.add_argument(
    #     '--mix_config_path',
    #     type=str,
    #     default='mix.config',
    #     help='the save path of config path to save'
    #     ) 
    
    parser.add_argument(
        '--save_dir',
        type=str,
        # default='/data/liudx/ConferenceSpeechsimulation/data/',
        default="/data/liudx/fardata_large_multinoise/",
        help='the dir to save generated_data'
        ) 
    parser.add_argument(
        '--train_num',
        type=int,
        default=28000,
        # default=700,
        help='the num of train data'
        ) 
    parser.add_argument(
        '--valid_num',
        type=int,
        default=8000,
        # default=200,
        help='the num of valid data'
        ) 
    parser.add_argument(
        '--test_num',
        type=int,
        default=4000,
        # default=100,
        help='the num of test data'
        ) 
    args = parser.parse_args()
    # parser.add_argument(
    #     '--chunk_len',
    #     type=float,
    #     default=6,
    #     help='the length of one chunk sample'
    #     ) 
    # parser.add_argument(
    #     '--generate_config',
    #     type=str,
    #     default='True',
    #     help='generate mix config file or not '
    #     ) 
    # args = parser.parse_args()
    # if args.generate_config == 'True' \
    #     or args.generate_config == 'true' \
    #     or args.generate_config == 't' \
    #     or args.generate_config == 'T':
    #     args.generate_config = True
    # else:
    #     args.generate_config = False
    main(args)
