ConferenceSpeech相关的数据生成模块(在gen_fardata文件夹下面)
my_challenge_rirgenerator.py:(需要gpurir，应该在mirana上运行)根据房间环境模拟rir
之后将rir复制到allrelia上面执行下一步操作
（复制操作：在allrelia下执行scp -r liudx@101.6.64.91:/data/liudx/rir_largeroom_multinoise/ /data/liudx
在mirana下执行复制命令会报错误，无法连接）
my_prepare_temp.sh:生成一个全部可选语音，噪声以及混响的列表
generate_list.py:根据my_prepare.sh生成train,valid,test数据集的全部内容（包含snr，生成带噪语音最高的幅度，以及接收阵列的几何排布）
my_mix_wav_new.py：根据list的参数生成多通道音频数据集
后面这几个文件应该在allrelia下面执行

rir路径：在Mirana上是/data/liudx/rir_largeroom_multinoise/
allrelia下复制到同样位置
数据量：一个简单的demo：700/200/100
实际用，考虑28000/8000/4000
数据位置：小数据集生成的位置：（暂时空缺）

这几个脚本在allrelia下的位置：/home/liudx/Data_generation/gen_fardata_large_new/，那几个list在这个文件夹下面/tmp的子文件夹下
###大数据集的脚本位置：/home/liudx/gen_fardata_large_multinoise/
改进之后数据集生成的位置/data/liudx/fardata_large_multinoise/
(注：我们考虑数据生成模型可能还是需要改进一下，尤其是模拟杂散噪声，考虑多个方向的源，我们直接搞出来6个噪声源。作为区别，这一版本我们生成的数据
路径最后多个_multinoise)
考虑低信噪比的情景，我们设定信噪比选择的范围变为[-10,-5,0,5]


