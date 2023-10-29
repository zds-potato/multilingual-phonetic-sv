import numpy as np
import torch, soundfile
import torch.nn as nn
import pandas as pd
import random
from scipy.io import wavfile
from scipy import signal


def get_random_chunk(data, chunk_len):
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data


class WavAugment(object):
    def __init__(self, noise_csv_path="data/noise.csv", rir_csv_path="data/rir.csv", second=2, sample_rate=16000, add_reverb_noise_prob=0.6):
        self.second = second
        self.sample_rate = sample_rate
        self.max_audio = np.int64(sample_rate * second + 240)

        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}

        df = pd.read_csv(noise_csv_path)
        augment_files = df["utt_paths"].values
        augment_types = df["speaker_name"].values
        for idx, file in enumerate(augment_files):
            if not augment_types[idx] in self.noiselist:
                self.noiselist[augment_types[idx]] = []
            self.noiselist[augment_types[idx]].append(file)
        df = pd.read_csv(rir_csv_path)
        self.rirs_files = df["utt_paths"].values
        
        self.add_reverb_noise_prob = add_reverb_noise_prob

    def __call__(self, waveform):
        if self.add_reverb_noise_prob > random.random():
            idx = np.random.randint(0, 5)
            
            if idx == 0: # Reverberation
                waveform = self.reverberate(waveform)
                
            if idx == 1: # Babble
                waveform = self.add_real_noise(waveform, 'speech')
            
            if idx == 2: # Music
                waveform = self.add_real_noise(waveform, 'music')
            
            if idx == 3: # Noise
                waveform = self.add_real_noise(waveform, 'noise')
            
            if idx == 4: # Television noise
                waveform = self.add_real_noise(waveform, 'speech')
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                waveform = self.add_real_noise(waveform, 'music') 
            
            # normalize into [-1, 1]
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-4)
            waveform = torch.from_numpy(waveform).unsqueeze(0)

        return waveform

    def add_real_noise(self, waveform, noisecat, resample_rate=16000):
        audio = waveform.numpy()[0]
        audio_len = audio.shape[0]
        audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
        snr_range = self.noisesnr[noisecat]
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noise_audio, noise_sr = soundfile.read(noise)
            noise_audio = noise_audio.astype(np.float32)
            if noise_sr != resample_rate:
                noise_audio = get_random_chunk(
                    noise_audio,
                    int(audio_len / resample_rate * noise_sr))
                noise_audio = signal.resample(noise_audio, audio_len)
            else:
                noise_audio = get_random_chunk(noise_audio, audio_len)
            noise_snr = random.uniform(snr_range[0], snr_range[1])
            noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
            noise_audio = np.sqrt(10**(
                (audio_db - noise_db - noise_snr) / 10)) * noise_audio
            noises.append(noise_audio)
        audio = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True) + audio
        return audio

    def reverberate(self, waveform, resample_rate=16000):
        audio = waveform.numpy()[0]
        audio_len = audio.shape[0]
        rirs_file = random.choice(self.rirs_files)
        rir_audio, rir_sr = soundfile.read(rirs_file)
        rir_audio = rir_audio.astype(np.float32)
        if rir_sr != resample_rate:
            rir_audio = signal.resample(
                rir_audio,
                int(len(rir_audio) / rir_sr * resample_rate))
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
        if audio.ndim == rir_audio.ndim:
            audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]
        return audio

if __name__ == "__main__":
    print("test")