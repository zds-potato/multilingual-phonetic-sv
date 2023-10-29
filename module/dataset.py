import collections
import os
import random

import numpy as np
import pandas as pd
import torch, soundfile
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from .augment import WavAugment

import torchaudio.compliance.kaldi as kaldi
import torchaudio

def load_audio(waveform, sample_rate, second=2):
    #waveform, sample_rate = soundfile.read(filename)
    #waveform, sample_rate = torchaudio.load(filename)
    audio_length = waveform.shape[1]

    if second <= 0:
        length = 160 * 10000 + 240
        if audio_length > length:
            #print(f'filename: {filename}')
            start = random.randint(0, audio_length-length)
            waveform =  waveform[:,start:start+length]
        return waveform
    length = sample_rate * second + 240

    if audio_length <= length:
        repeat_factor = length // audio_length + 1
        repeat_shape = repeat_factor
        waveform = waveform.repeat(1, repeat_shape)
        waveform = waveform[:,:length]
    else:
        start = random.randint(0, audio_length-length)
        waveform =  waveform[:,start:start+length]

    return waveform

class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, num_classes=1211,
                 speed_perturb_flag=False, 
                 add_reverb_noise=False, noise_csv_path="data/musan_lst.csv", rir_csv_path="data/rirs_lst.csv", 
                 spec_aug_flag=False,
                 **kwargs):
        
        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.second = second
        self.num_classes = num_classes
        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))
        
        self.speed_perturb_flag = speed_perturb_flag

        self.add_reverb_noise = add_reverb_noise
        if self.add_reverb_noise:
            self.wav_aug = WavAugment(noise_csv_path=noise_csv_path, rir_csv_path=rir_csv_path, second=self.second, sample_rate=16000)

        self.spec_aug_flag=spec_aug_flag

        print(f'speed_perturb_flag: {self.speed_perturb_flag}')
        print(f'add_reverb_noise: {self.add_reverb_noise}')
        print(f'spec_aug_flag: {self.spec_aug_flag}')

    def __getitem__(self, index):
        filename, label = self.paths[index], self.labels[index]
        waveform, sample_rate = torchaudio.load(filename)
        
        if self.speed_perturb_flag:
            waveform, label = speed_perturb(waveform, label, num_spks=self.num_classes)
        
        waveform = load_audio(waveform, sample_rate, self.second)
        
        if self.add_reverb_noise:
            waveform = self.wav_aug(waveform)

        waveform = waveform * (1 << 15)
        feature = kaldi.fbank(waveform,
                              num_mel_bins=80,
                              frame_length=25,
                              frame_shift=10,
                              dither=1.0,
                              sample_frequency=16000,
                              window_type='hamming',
                              use_energy=False)
        
        #CMN
        feature = feature - torch.mean(feature, dim=0) #[200,80]
        
        #spec_aug
        if self.spec_aug_flag:
            feature = spec_aug(feature)
        
        feature = feature.T #[80,200]
        feature = feature.unsqueeze(0) #[1,80,200]
            
        return feature, label

    def __len__(self):
        return len(self.paths)


def speed_perturb(data, label, num_spks, sample_rate=16000):
    speeds = [1.0, 0.9, 1.1]
    speed_idx = random.randint(0, 2)
    if speed_idx > 0:
        data, _ = torchaudio.sox_effects.apply_effects_tensor(
                data, sample_rate, [['speed', str(speeds[speed_idx])], ['rate',str(sample_rate)]])
        label = label + num_spks * speed_idx
    return data, label


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8):
    #input size: [200,80]
    y = data.detach()  # inplace operation
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self.paths[index])
        waveform = load_audio(waveform, sample_rate, self.second)
        waveform = waveform * (1 << 15)
        feature = kaldi.fbank(waveform,
                              num_mel_bins=80,
                              frame_length=25,
                              frame_shift=10,
                              dither=0.0,
                              sample_frequency=16000,
                              window_type='hamming',
                              use_energy=False)
        #CMN
        feature = feature - torch.mean(feature, dim=0)
        feature = feature.T
        feature = feature.unsqueeze(0)
        return feature, self.paths[index]

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    dataset = Train_Dataset(train_csv_path="data/train.csv", second=3)
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )
    for x, label in loader:
        pass

