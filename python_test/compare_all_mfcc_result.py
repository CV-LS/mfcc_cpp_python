import math

import librosa
import numpy as np
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import torchaudio.transforms
import torch

y, sr = librosa.load("1.wav")

n_mels = 40 # mel取40个特征
n_mfcc = 13 # 取mel的前13个特征
n_fft = 2048 if sr > 20000 else 512
win_length = int(sr*25*y.ndim/1000)
hop_length = int(sr*10*y.ndim/1000)
fmin = 0
fmax = sr // 2
center = True
norm = True
window = "hann"
pad_mode = "reflect"
power = 2.0

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length":hop_length, "f_min" : fmin, "f_max" : fmax,"win_length":win_length,"mel_scale":'htk'}

# Nearly identical to above
# mfcc_lib_db = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc, htk=False)

# Modified librosa with log mel scale
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin,
                                    fmax=fmax, hop_length=hop_length)
mfcc_lib_log = librosa.feature.mfcc(S=np.log(S+1e-6), n_mfcc=n_mfcc, htk=False,hop_length=hop_length,win_length=win_length)

# Default librosa with db mel scale
mfcc_lib_db = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)

# Python_speech_features
mfcc_speech = python_speech_features.mfcc(signal=y, samplerate=sr, winlen=win_length / sr, winstep=hop_length / sr,
                                          numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=sr//2,
                                          preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=hann)

# Torchaudio 'textbook' log mel scale
mfcc_torch_log = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                                            dct_type=2, norm='ortho', log_mels=True,
                                            melkwargs=melkwargs)(torch.from_numpy(y))

# Torchaudio 'librosa compatible' default dB mel scale
mfcc_torch_db = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                                           dct_type=2, norm='ortho', log_mels=False,
                                           melkwargs=melkwargs)(torch.from_numpy(y))

# mfcc c++ implement
data = []
with open('data.txt', 'r') as file:
    for line in file:
        # 使用 np.fromstring 将每行的字符串转换为浮点数数组
        # sep=', ' 指定了分隔符为逗号和一个空格
        data.append(np.fromstring(line, sep=', ', dtype=float))

# 将列表转换为 numpy 数组
data_array = np.array(data)

if __name__=='__main__':
    feature = 1 # 特征index
    print(mfcc_lib_log.T.shape)
    print(mfcc_torch_db.T.shape)
    print(mfcc_speech.shape)
    print(mfcc_torch_log.T.shape)
    print(mfcc_lib_db.T.shape)
    print(data_array.shape)
    plt.subplot(2, 1, 1)
    # 绘制不同的MFCC数据集，使用不同的颜色和图例
    plt.subplot(2, 1, 1)  # 2行1列的第一个子图
    plt.plot(mfcc_lib_log.T[3:, feature], 'k', label='Black - Lib Log')   # 黑色
    plt.plot(mfcc_torch_db.T[:, feature], 'b', label='Blue - Torch DB')   # 蓝色
    plt.plot(mfcc_speech[1:, feature], 'r', label='Red - Speech')        # 红色
    plt.plot(mfcc_torch_log.T[3:, feature], 'c', label='Cyan - Torch Log') # 青色
    plt.plot(data_array[:, feature], 'g', label='Green - C++ Imp')     # 绿色
    plt.plot(mfcc_lib_db.T[:, feature], 'y', label='Yellow - Lib DB')      # 黄色
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(mfcc_lib_log.T[:,feature], 'k')
    plt.plot(mfcc_torch_log.T[:,feature], 'c')
    plt.plot(mfcc_speech[:,feature], 'r')
    plt.plot(data_array[:,feature], 'g')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(mfcc_torch_db.T[:,feature], 'b')
    plt.plot(mfcc_lib_db.T[:,feature], 'y')
    plt.grid()
    plt.show()