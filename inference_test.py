# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
from utils import plot_tensor, save_plot

import torch

import params
from model import GradTTSVitE2E
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
from data import TextMelSpeakerMelRefDataset,VCTKMelRefDataset

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/g.pt' # generator_VCTK generator_v1 g
train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank
n_spks = params.n_spks
spk_emb_dim = params.spk_emb_dim

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

def inference (start, end, test_batch, generator) :
    for idx_t in range (start,end):
        gt_out_file_name, text = test_batch[idx_t]['file_path'], test_batch[idx_t]['text_gt']
        gt_out_file_name = gt_out_file_name.replace('.wav','')
        print(text)

        mel = test_batch[idx_t]['y']
        #save_plot(mel.squeeze(), f'./out/{args.out_path}/images/original_{idx_t}.png')

        for idx in range (start,end):
            if idx_t == idx:
                continue

            y_ref_, y_ref_lengths_, speaker, out_file_name = test_batch[idx]['y'], test_batch[idx]['y'].shape[-1], \
                                            test_batch[idx]['spk'].item(), test_batch[idx]['file_path']
            
            texts = [text]
            #out_file_name = gt_out_file_name + '-' + out_file_name
            out_file_name = gt_out_file_name.split('/')[-1].replace('/workspace/dm_datasets/VCTK/wav_processed/','') + '-' + out_file_name.split('/')[-1].replace('/workspace/dm_datasets/VCTK/wav_processed/','')
           
            print('speaker_ref', out_file_name)

            length = (y_ref_.shape[-1] // 8 ) * 8 + 8
            y_ref = torch.zeros((1, 80, length), dtype=torch.float32)
            y_ref  [:,:,:y_ref_.shape[-1]] = y_ref_

            #y_ref = torch.zeros((1, n_feats, y_ref_lengths_), dtype=torch.float32)
            #y_ref_lengths = torch.zeros((1), dtype=torch.int32)
            #y_ref[0, :, : y_ref_lengths_] = y_ref_
            #y_ref_lengths[0] = y_ref_lengths_

            with torch.no_grad():
                for i, text in enumerate(texts):
                    print(f'Synthesizing {i} text...', end=' ')
                    x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
                    x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                    
                    t = dt.datetime.now()
                    y_enc, y_dec, attn = generator.forward(x, x_lengths, args.timesteps, y_ref, 
                                                            torch.LongTensor(length) ,#y_ref_lengths, 
                                                           temperature=1.5,
                                                           stoc=False, spk=spk, length_scale=0.91)
                    t = (dt.datetime.now() - t).total_seconds()
                    print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
                    
                    audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    #audio_enc = (vocoder.forward(y_enc).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

                    #print(audio)
                    #write(f'./out/sample_{i}.wav', 22050, audio)
                    #audio_no_HfGan = y_dec.cpu().squeeze().numpy()
                    #print((audio_no_HfGan * 32768).astype(np.int16))
                    print('saving at : ./out/',args.out_path,out_file_name)
                    write(f'./out/{args.out_path}/{out_file_name}', 22050, audio)
                    #write(f'./out/{args.out_path}/enc_{out_file_name}', 22050, audio_enc)

                    image_path = out_file_name.replace('wav','png')
                    save_plot(y_dec.squeeze().cpu(), f'./out/{args.out_path}/images/{image_path}')
                    #save_plot(y_enc.squeeze().cpu(), f'./out/{args.out_path}/images/enc_{image_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-cp', '--checkpoint_prior', type=str, required=False, help='path to a checkpoint of Grad-TTS Prior')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    parser.add_argument('-i', '--speaker_idx', type=int, required=False, default=None, help='speaker idx for multispeaker model')
    parser.add_argument('-o', '--out_path', type=str, required=False, default=None, help='out path')


    args = parser.parse_args()
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None
   
    print('Initializing Grad-TTS-Ref...')
    
    #GradTTSTransE2E,GradTTSVitE2E

    generator = GradTTSVitE2E(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc), strict=False)
    #generator.load_state_dict(torch.load(args.checkpoint), strict=False)

    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')

    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    #test_dataset = TextMelSpeakerMelRefDataset(valid_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)
    
    test_dataset = VCTKMelRefDataset(valid_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)
    
    print('Logging test batch...')
    test_batch = test_dataset.get_ref_batch(size=20)

    inference (0, 5, test_batch, generator)
    inference (5, 10, test_batch, generator)    
    inference (10, 15, test_batch, generator)   
    inference (15, 20, test_batch, generator)           
    print('Done. Check out `out` folder for samples.')
