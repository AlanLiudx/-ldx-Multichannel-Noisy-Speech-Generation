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
room_x=50
room_y=50
room_z=3
#房间大小,长宽高分别为(50m,50m,3m),我们暂时考虑把声源先放正中间,其实不一定合适
nb_src = 7  # Number of sources
nb_rcv = 8 # Number of receivers  
#7个声源，8个麦克风可以接收
#这里说明一下，7个声源是1个语音源以及6个噪声源，当然噪声源在角度上各自间隔60度，每一个语音源与初始的噪声源距离大于+-10度
#(注意：音频文件，噪声在musan文件夹下，语音在train-clean-360文件夹下而不是在librispeech360，不然会像大海捞针一样瞎找)
mic_pattern = "omnidirectional"
# mic_middle_point=np.array([10,10,1.5])
mic_middle_point=[25,25,1.5]
#麦克风中点固定在(25m,25m,1.5m),算了还是随机吧
#先考虑一个8mic圆阵
room_sz=[room_x,room_y,room_z]  #房间尺寸
source_distance=20
noise_distance=10
#声源距离，源距离20m左右
#噪声距离，我们先把源距离设定在10m，后面再考虑近端远端距离问题

#两个源，第一行是信号源，第二行是噪声源
#接收麦克风坐标，麦克风直径30cm，用8个
mic_radius=0.15 #直径30cm就是半径15cm
inner_radius=0.1
baseangle=0.25*np.pi #8mic，沿着一个圈均匀分布
# receiver_shape='round'
T60=0.01 #先设置较高的rt60，如果可能的话再改成低的看看是否会报错
#结论是并不会报错，我设置成1e-5都不会报错，但是实际上这个时间可能还没传播到墙上呢，应该自己实地仿真出几个不同大小的T60自己卷积完了以后听一下靠不靠谱，对比一下
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)
nb_img = gpuRIR.t2n(T60, room_sz)
Time_max=4.0 #取4s时长的rir
#声源坐标
rir_num=4000 #生成200条rir,不过这次我们多来点，搞4000条
receiver_shape_list=['round','linear','concentric']
for index in tqdm(range(rir_num)):
    
    source_angle=np.random.uniform(0.0,360.0)
    noise_angle=np.random.uniform(source_angle+10.0,source_angle+350.0)
    while(noise_angle>=360.0):
        noise_angle=noise_angle-360.0
    noise_angle_list=[]
    for num_noise in range(0,6):
        noise_angle_list.append(noise_angle+60.0*num_noise)
#随机生成两个角度，让声源和噪声源的角度约束住，角度差值在10度以上（叠加在一起的话就是纯粹的单通道问题了）
#此外，我们限制角度不超过360度
    source_direction=source_angle*np.pi/180.0  #声源到mic中心点的角度,角度转弧度
    # noise_direction=noise_angle*np.pi/180.0
    noise_direction_list=noise_angle_list.copy()
    for angle_idx,angle in enumerate(noise_direction_list):
        noise_direction_list[angle_idx]=noise_direction_list[angle_idx]*np.pi/180.0
  
    # for angle in noise_direction_list:
    #     angle=angle*np.pi/180.0
    #     ##test
    #     print("direction in process:",angle)
    #     ##finish test
   


    pos_src=np.array([[mic_middle_point[0]+source_distance*np.cos(source_direction),mic_middle_point[1]+source_distance*np.sin(source_direction),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[0]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[0]),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[1]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[1]),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[2]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[2]),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[3]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[3]),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[4]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[4]),mic_middle_point[2]],
    [mic_middle_point[0]+noise_distance*np.cos(noise_direction_list[5]),mic_middle_point[1]+noise_distance*np.sin(noise_direction_list[5]),mic_middle_point[2]]
    ])

    #每一种阵列结构都生成一部分
    for receiver_shape in receiver_shape_list:
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
        #每一类生成一个rir

        RIR = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Time_max, fs,c=c)
        #生成文件夹

        folder='/data/liudx/rir_largeroom_multinoise/' #仅限于mirana上,生成rir
        if not os.path.isdir(folder):
            os.mkdir(folder)
        folder=os.path.join(folder,receiver_shape)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        
        wav_name="10m"+"_"+"10m"+"_"+('%.4f' % source_angle)+"_"+('%.4f' % noise_direction_list[0])+"_"+".wav"
        wave_name_all=os.path.join(folder,wav_name)
        ##RIR整形
        RIR=np.array(RIR)
        #gpurir的输出是(7（源数目），8（接收器数目），xxxxx（tmax对应的rir长度）)
     

        out=np.zeros([8*7,RIR.shape[2]])
        for i in range(7):
            out[i*8:(i+1)*8]=RIR[i,:,:]
        # out[0:8]=RIR[0,:,:]
        # out[8:16]=RIR[1,:,:]
        out = out.transpose(1,0)
        
        #把三维的rir整成两个维度，第二个维度还是rir长度，第一个维度(0：7)是源，第一个维度(8：56)是噪声rir
        #(8:16)是第一个噪声源,(16:24)是第二个噪声源,依次类推
        #最后维度（rir长度，56）
        sf.write(wave_name_all,out,16000)
        # folder=os.path.join(folder,str(T60))
        # # print("folder name:",folder)
        # if not os.path.isdir(folder):
        #     os.mkdir(folder)
