##Credit to Liu Dongxu

#我们尝试生成一个大空间远距离的音频对应的rir


import numpy as np
import soundfile as sf
import math
import tqdm
from tqdm import tqdm
# import pyrirgen
import gpuRIR
import os
import soundfile as sf
import librosa

c=340
fs=16000
#声速与采样率，这两个参数基本不变（声速本身也不是关键的参数）
room_x=20
room_y=20
room_z=3
#房间大小,长宽高分别为(20m,20m,3m),我们暂时考虑把声源先放正中间,其实不一定合适
nb_src = 2  # Number of sources
nb_rcv = 8 # Number of receivers  
#2个声源，8个麦克风可以接收
#(注意：音频文件，噪声在musan文件夹下，语音在train-clean-360文件夹下而不是在librispeech360，不然会像大海捞针一样瞎找)
mic_pattern = "omnidirectional"
# mic_middle_point=np.array([10,10,1.5])
mic_middle_point=[10,10,1.5]
#麦克风中点固定在(10m,10m,1.5m)
#先考虑一个8mic圆阵
room_sz=[room_x,room_y,room_z]  #房间尺寸
source_distance=10
#声源坐标
source_angle=np.random.uniform(0.0,360.0)
noise_angle=np.random.uniform(source_angle+10.0,source_angle+350.0)
if(noise_angle>=360.0):
    noise_direction=noise_angle-360.0
#随机生成两个角度，让声源和噪声源的角度约束住，角度差值在10度以上（叠加在一起的话就是纯粹的单通道问题了）
#此外，我们限制角度不超过360度
source_direction=60.0*np.pi/180.0  #声源到mic中心点的角度,角度转弧度
noise_direction=135.0*np.pi/180.0


pos_src=np.array([[mic_middle_point[0]+source_distance*np.cos(source_direction),mic_middle_point[1]+source_distance*np.sin(source_direction),mic_middle_point[2]],
[mic_middle_point[0]+source_distance*np.cos(noise_direction),mic_middle_point[1]+source_distance*np.sin(noise_direction),mic_middle_point[2]]])
#两个源，第一行是信号源，第二行是噪声源
#接收麦克风坐标，麦克风直径30cm，用8个
mic_radius=0.15 #直径30cm就是半径15cm
inner_radius=0.1
baseangle=0.25*np.pi #8mic，沿着一个圈均匀分布
receiver_shape='round'
#接收麦克风的布局，目前初步考虑圆阵线阵以及内外圆阵
#圆阵的接收位置
if receiver_shape=='round':
    pos_rcv=[[mic_middle_point[0]+mic_radius*np.cos(8*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*8), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(7*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*7), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(6*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*6), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(5*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*5), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(4*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*4), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(3*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*3), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(2*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*2), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(1*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*1), mic_middle_point[2]]
    ]
#线阵的接收位置(距离5,5,4,2,4,5,5cm)
elif receiver_shape=='linear':
    pos_rcv=[[mic_middle_point[0]-0.15, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]-0.10, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]-0.05, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]-0.01, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]+0.01, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]+0.05, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]+0.10, mic_middle_point[1], mic_middle_point[2]],
    [mic_middle_point[0]+0.15, mic_middle_point[1], mic_middle_point[2]]
    ]
#内外圆的接收位置，外圈直径30cm，内圈直径20cm
elif receiver_shape=='concentric': #内外圆
    pos_rcv=[[mic_middle_point[0]+mic_radius*np.cos(8*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*8), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(6*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*6), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(4*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*4), mic_middle_point[2]],
    [mic_middle_point[0]+mic_radius*np.cos(2*baseangle), mic_middle_point[1]+mic_radius*np.sin(baseangle*2), mic_middle_point[2]],
    [mic_middle_point[0]+inner_radius*np.cos(8*baseangle), mic_middle_point[1]+inner_radius*np.sin(baseangle*8), mic_middle_point[2]],
    [mic_middle_point[0]+inner_radius*np.cos(6*baseangle), mic_middle_point[1]+inner_radius*np.sin(baseangle*6), mic_middle_point[2]],
    [mic_middle_point[0]+inner_radius*np.cos(4*baseangle), mic_middle_point[1]+inner_radius*np.sin(baseangle*4), mic_middle_point[2]],
    [mic_middle_point[0]+inner_radius*np.cos(2*baseangle), mic_middle_point[1]+inner_radius*np.sin(baseangle*2), mic_middle_point[2]]
    ]
#转化为应有的格式
pos_rcv=np.array(pos_rcv)
#输出发射源和接收源的数目
print("position of source size:",pos_src.shape)
print("position of receiver size:",pos_rcv.shape)
#
T60=1.0 #先设置较高的rt60，如果可能的话再改成低的看看是否会报错
#结论是并不会报错，我设置成1e-5都不会报错，但是实际上这个时间可能还没传播到墙上呢，应该自己实地仿真出几个不同大小的T60自己卷积完了以后听一下靠不靠谱，对比一下
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)
nb_img = gpuRIR.t2n(T60, room_sz)
Time_max=4.0 #取4s时长的rir
RIR = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Time_max, fs,c=c)
print("RIR shape:",RIR.shape)
#两段声音的路径
source_audio_dir="/data/share/train-clean-360/100/121669/100-121669-0000.flac"
source_noise_dir="/data/share/musan/noise/sound-bible/noise-sound-bible-0000.wav"
source_audio,sr=sf.read(source_audio_dir)
source_noise,sr=sf.read(source_noise_dir)
print("source audio length:",source_audio.shape)
print("source noise length:",source_noise.shape)
#分别代入那个什么trajctory
RIR_source=RIR[0:1,:,:]
RIR_noise=RIR[1:2,:,:]
print("RIR source shape:",RIR_source.shape)
print("RIR noise shape:",RIR_noise.shape)
filtered_audio=gpuRIR.simulateTrajectory(source_audio, RIR_source)
filtered_noise=gpuRIR.simulateTrajectory(source_noise, RIR_noise)
print("filtered audio shape:",filtered_audio.shape)
print("filtered noise shape:",filtered_noise.shape)
# if not os.path.isdir('/home/liudx/test_gpurir/outwave_near'):
#     os.mkdir('/home/liudx/test_gpurir/outwave_near')
# folder=os.path.join('/home','liudx')
# print("folder name:",folder)
folder='/home/liudx/test_gpurir/'
folder=os.path.join(folder,receiver_shape)
if not os.path.isdir(folder):
    os.mkdir(folder)
folder=os.path.join(folder,str(T60))
print("folder name:",folder)
if not os.path.isdir(folder):
    os.mkdir(folder)
filtered_audio_out=filtered_audio.transpose(1,0)
print("out shape:",filtered_audio_out.shape)
#注意多通道输出应该是把通道放在第二个维度，输出filtered audio，不需要再转置了
filtered_audio_out_ref=filtered_audio_out[0,:]
print("out ref shape:",filtered_audio_out_ref.shape)
wav_out_name=os.path.join(folder,'outputwave.wav')
sf.write(wav_out_name,filtered_audio_out_ref,16000)
wav_out_name=os.path.join(folder,'outputwave_all.wav')
sf.write(wav_out_name,filtered_audio,16000)
# if not os.path.isdir('./models'):
#         os.mkdir('./models')

#综上，我们注意到改变t60，在t60很小(1e-5)的情况仍然不报错，而且在t60小到一定程度就没有混响了
#但是有一个问题，t60很小的情况，各个channel怎么输出是完全一致的呢