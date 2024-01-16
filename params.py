# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters

train_filelist_path = 'resources/filelists/libri-tts/train.txt'
#train_filelist_path = 'resources/filelists/libri-tts/train-clean-full.txt'
#train_filelist_path = 'resources/filelists/libri-tts/test-clean.txt'

#train_filelist_path = 'resources/filelists/libri-tts/train-clean-full-F.txt'
#train_filelist_path = 'resources/filelists/libri-tts/train-clean-full-M.txt'

valid_filelist_path = 'resources/filelists/libri-tts/valid_v2.txt'

train_filelist_path = 'resources/filelists/VCTK/train__.txt'
valid_filelist_path = 'resources/filelists/VCTK/test__.txt'

#alid_filelist_path = 'resources/filelists/libri-tts/valid_maori_ref.txt'

#test_filelist_path = 'resources/filelists/ljspeech/test.txt'

cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
#n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
n_spks = 247
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 22050
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000
ms_hd = 512

# encoder parameters
n_enc_channels = 512 #256 #512 #1024 
filter_channels = 768
filter_channels_dp =  512 # 512 #256
n_enc_layers = 6 # 12 # 10 #6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 6 # 3 #2 # 12
window_size = 4

# decoder parameters
dec_dim = 128 #128 #256 #64 #128
beta_min =  0.05
beta_max = 20.0 
pe_scale = 1000 #1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters

#log_dir = 'logs/tts_ref_in_context_LibriTTS'
#log_dir = 'logs/tts_ref_in_context_VCTK'

#log_dir = 'logs/tts_ref_in_context_LibriTTS_6'
#log_dir = 'logs/tts_ref_in_context_LibriTTS_7'
log_dir = 'logs/tts_ref_in_context_VCTK_Huggingface'
#log_dir = 'logs/tts_ref_in_context_LibriTTS_Huggingface'

test_size = 20
n_epochs = 500
batch_size = 16 # 32 #96
learning_rate = 1e-5 #1e-4#1e-5
seed = 32
save_every = 5
out_size = fix_len_compatibility(2*22050//256)
