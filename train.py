# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTSVitE2E
from data import TextMelSpeakerMelRefDataset, TextMelSpeakerMelRefBatchCollate,VCTKMelRefDataset
from utils import plot_tensor, save_plot
from text.symbols import symbols

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cntn', '--continue_train', type=bool, required=False, default=False, help='continue train?')
    parser.add_argument('-ckpt', '--checkpoint', type=str, required=False, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-i', '--starting_epoch', type=int, required=False, default=1, help='starting epoch')
    args = parser.parse_args()

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    #train_dataset = TextMelSpeakerMelRefDataset(train_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)
    train_dataset = VCTKMelRefDataset(train_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)

    batch_collate = TextMelSpeakerMelRefBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=True)
    #test_dataset = TextMelSpeakerMelRefDataset(valid_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)
    test_dataset = VCTKMelRefDataset(valid_filelist_path, cmudict_path, add_blank,n_fft, n_feats, sample_rate, hop_length,win_length, f_min, f_max)
    print('Initializing model...')

    model = GradTTSVitE2E(nsymbols, n_spks, spk_emb_dim, n_enc_channels,
                    filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale)
    #'''GradTTSTransE2E,GradTTSVitE2E
    
    #print('Number of encoder parameters = %.2fm' % (model.encoder.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)

    i = 0
    for item in test_batch:
        mel, spk = item['y'], item['spk']
        i += 1 #int(spk.cpu())
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    if args.continue_train : 
        print('Loading previous model...')
        model.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
        print(f'Number of parameters: {model.nparams}')
    
    model.cuda()

    print('Start training...')
    iteration = 0
    for epoch in range(args.starting_epoch, n_epochs + 1):
        model.eval()
        print('Synthesis...')
        #'''
        with torch.no_grad():
            i = 0 
            for item in test_batch:
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                spk = item['spk'].to(torch.long).cuda()
                i += 1 #int(spk.cpu())
                
                y_ref = item['y'].unsqueeze(0).cuda()
                y_ref_lengths = torch.LongTensor([y_ref.shape[-1]]).cuda()

                y_enc, y_dec, attn  = model(x, x_lengths, 5, y_ref, y_ref_lengths , spk=spk)
                
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')
        #'''
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []

        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch in progress_bar:
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                spk = batch['spk'].cuda()

                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                        y, y_lengths,
                                                                        out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 
                                                            max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 
                                                            max_norm=1)
                
                optimizer.step()

                logger.add_scalar('training/diffusion_loss', diff_loss,
                                global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                global_step=iteration)
                
                msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                progress_bar.set_description(msg)
                
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                iteration += 1

                #if iteration > 1:
                #    print(asd)

        msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)

        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(msg)

        if epoch % params.save_every > 0:
            continue
        
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
