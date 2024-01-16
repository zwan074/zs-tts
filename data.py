# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

# Research
class TextMelSpeakerMelRefDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank

        random.seed(random_seed)
        

    def get_triplet(self, line):
        
        #update data here
        #
        #

        filepath, text, speaker = line[0], line[1], line[2]
        filepath = filepath.replace('DUMMY','/workspace/dm_datasets/LibriTTS/')
        #filepath = filepath.replace('DUMMY','workspace') #filepath.replace('DUMMY','/workspace/dm_datasets/LibriTTS/')
        #print(filepath,text)
        text_gt = text
        text = self.get_text(text, add_blank=self.add_blank)
        mel, audio, sr = self.get_mel(filepath)
        
        speaker = self.get_speaker(speaker)

        return (text, mel, speaker, filepath.replace('/workspace/dm_datasets/LibriTTS/','').split('/')[-1], text_gt, audio, sr) # new

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        #print(sr, self.sample_rate)
        #print(audio)
        #assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        #print(audio.shape)
        
        
        return mel, audio, sr

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, file_path, text_gt, audio, sr = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker,  
                'file_path':file_path, 'text_gt': text_gt, 
                'audio':audio, 'sr':sr}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

    def get_ref_batch(self, size):
        idx = np.arange(size)
        test_batch = []
        #print(self.filelist)
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class TextMelSpeakerMelRefBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)
        
        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)

        return {'x': x, 'x_lengths': x_lengths, 
                'y': y, 'y_lengths': y_lengths, 
                'spk': spk
                }
    
    

class VCTKMelRefDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank

        random.seed(random_seed)
        

    def get_triplet(self, line):
        
        #update data here
        #
        #

        filepath, text, speaker = line[0], line[1], line[2]
        #print(filepath,text)
        text_gt = text
        text = self.get_text(text, add_blank=self.add_blank)
        mel, audio, sr = self.get_mel(filepath)

        speaker = self.get_speaker(speaker.replace('p','').replace('s',''))

        return (text, mel, speaker, filepath, text_gt, audio, sr) # new

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, sr, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel, audio, sr 

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, file_path, text_gt, audio, sr = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker,  
                'file_path':file_path, 'text_gt': text_gt, 
                'audio':audio, 'sr':sr}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

    def get_ref_batch(self, size):
        idx = np.arange(size)
        test_batch = []
        #print(self.filelist)
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class VCTKMelRefBatchCollate(object):
    def __call__(self, batch):

        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, (y_max_length//4 ) * 4 + 4), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)
        
        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)

        return {'x': x, 'x_lengths': x_lengths, 
                'y': y, 'y_lengths': y_lengths, 
                'spk': spk
                }