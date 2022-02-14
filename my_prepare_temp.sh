#!/bin/bash
###我们从指定文件夹下面提取出来所有可供添加到list里面的文件的绝对路径
rir="/data/liudx/rir_largeroom_multinoise/"
##先只用圆阵rir,算了每一种都用上
librispeech="/data/share/datasets/train-clean-360/"
#注意，librispeech都是flac文件
aishell1="/data/share/datasets/aishell_1/"
###噪声路径
noise="/data/share/datasets/musan/noise/"
##RIR路径
rir="/data/liudx/rir_largeroom_multinoise/"
#如果不存在data文件夹，就创建data文件夹
#如果不存在temp文件夹，就创建temp文件夹
#实际上除了librispeech是.flac文件之外，剩下的全都是.wav文件

if [ ! -d data ] ; then
    mkdir data
fi
if [ ! -d tmp ] ; then
    mkdir tmp
fi

for name_path in    ${librispeech} ; do
#这个name实际上是把前面目录去掉，只留下文件名以及后缀的
    name=`basename ${name_path}`
    find ${name_path} -regex ".*\.wav\|.*\.flac" >./tmp/clean.name
    #regex是正则表达式，相当于从这一圈里面专门找.wav格式以及.flac格式的文件，然后输出到一个临时文件里面
    echo $name
  
done

for name_path in    ${aishell1} ; do
#这个name实际上是把前面目录去掉，只留下文件名以及后缀的
    name=`basename ${name_path}`
    find ${name_path} -regex ".*\.wav\|.*\.flac" >>./tmp/clean.name
    #regex是正则表达式，相当于从这一圈里面专门找.wav格式以及.flac格式的文件，然后输出到一个临时文件里面
    echo $name
  
done
#这样输出的文件，aishell的list被追加在librispeech后面，clean name一共217950条

for name_path in    ${noise} ; do
#这个name实际上是把前面目录去掉，只留下文件名以及后缀的
    name=`basename ${name_path}`
    find ${name_path} -regex ".*\.wav\|.*\.flac" >./tmp/noise.name
    #regex是正则表达式，相当于从这一圈里面专门找.wav格式以及.flac格式的文件，然后输出到一个临时文件里面
    echo $name
  
done
#noise name一共930条


for name_path in    ${rir} ; do
#这个name实际上是把前面目录去掉，只留下文件名以及后缀的
    name=`basename ${name_path}`
    find ${name_path} -regex ".*\.wav\|.*\.flac" >./tmp/rir.name
    #regex是正则表达式，相当于从这一圈里面专门找.wav格式以及.flac格式的文件，然后输出到一个临时文件里面
    #这操作默认就是包含全部子文件夹下的情况的
    echo $name
  
done
#rir部分，把三种阵的rir都扔进去了，能生成多少条rir就是多少条。但是考虑一下，我们不需要adhoc，建议还是每一种类型的数据搞成一类。目前rir应该是3600条