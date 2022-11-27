# make mixture
print('hello')


# import
import pyroomacoustics as pra
import os

# global variables

dataset_dir = r'D:\Database\CHiME5'
noiseset_dir = r'D:\Database\CHiME2\chime2_wsj0\data\background'
N_sets = 333


## class definition

class stft_Para:
    def __init__(self):
        self.fs = 16000
        # self.fft_len = 4096
        # self.lap_len = 3072
        self.fft_len = 2048
        self.lap_len = 1024

class simu_Para:
    def __init__(self):
        self.wall_x = 5







## function definition



## working flow

### 读取wav文件名称序列并且生成相应的列表


noise_dir_list = [speaker for speaker in os.listdir(noiseset_dir)# 一个存储了noise的文件
                    if speaker.endswith('.wav')]








print('done')








