# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random
import numpy as np
import torch

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder_ref_vit import TextEncoderRefVit
from model.diffusion_ref import DiffusionRef
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
import torch.nn.functional as F

from typing import Any
from torch.hub import load_state_dict_from_url

__all__ = ['Model', 'model']

model_urls = {
    'model': 'https://huggingface.co/zwan074/maori_ASR/resolve/main/grad_120.pt',
}


class GradTTSVitE2E(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTSVitE2E, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        #if n_spks > 1:
        #    self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoderRefVit(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
            
        self.decoder = DiffusionRef(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)


    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, y_ref, y_ref_lengths, 
                temperature=1.0, stoc=False, spk=None, length_scale=1.0):

        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths, y_ref, y_ref_lengths = self.relocate_input([x, x_lengths, y_ref, y_ref_lengths])
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask  = self.encoder(x, x_lengths, y_ref)

        #print(mu_x.shape,logw.shape)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]
        
        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature 
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps,  stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def spec_mask (self,mu_y):

        for i in range (0,2):
            roll = np.random.randint(0, 2, size=1)[0]
            if roll > 0 : 
                mask_length = np.random.randint(0, mu_y.shape[2] // 3 + 1, size=1)[0]
                start_idx = np.random.randint(0, mu_y.shape[2]-mask_length, size=1)[0]
                end_idx = start_idx + mask_length + 1
                mu_y = torch.cat((mu_y[:,:,:start_idx] , mu_y[:,:,end_idx:] ), dim=-1)


            roll = np.random.randint(0, 2, size=1)[0]
            if roll > 0 : 
                mask_length = np.random.randint(0, mu_y.shape[1] //10 + 1, size=1)[0]
                start_idx = np.random.randint(0, mu_y.shape[1]-mask_length, size=1)[0]
                end_idx = start_idx + mask_length + 1
                mu_y[:,start_idx:end_idx,:] = 0
        
        return mu_y
    
    def spec_mask_2 (self,mu_y):

        for i in range (0,2):
            roll = np.random.randint(0, 2, size=1)[0]
            if roll > 0 : 
                mask_length = np.random.randint(0, mu_y.shape[2] // 10 + 1, size=1)[0]
                start_idx = np.random.randint(0, mu_y.shape[2]-mask_length, size=1)[0]
                end_idx = start_idx + mask_length + 1
                mu_y[:,:,start_idx:end_idx] = 0

            roll = np.random.randint(0, 2, size=1)[0]
            if roll > 0 : 
                mask_length = np.random.randint(0, mu_y.shape[1] //10 + 1, size=1)[0]
                start_idx = np.random.randint(0, mu_y.shape[1]-mask_length, size=1)[0]
                end_idx = start_idx + mask_length + 1
                mu_y[:,start_idx:end_idx,:] = 0
        
        return mu_y
    
    def compute_loss(self, x, x_lengths, y, y_lengths, out_size=None):

        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        y_ref = y
        y_ref = self.spec_mask (y_ref)
        y_ref_lengths = y_ref.shape[-1]

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask  = self.encoder(x, x_lengths, y_ref)


        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad(): 
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        #print('dur_loss', dur_loss)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)
            
            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask
        
        #print(mu_x.shape,logw.shape, logw_.shape, attn.shape)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        #print(mu_y.shape)
        mu_y_masked = self.spec_mask_2(mu_y) 
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        #prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        #prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        #ref_loss = F.mse_loss( mel_ref, y_ref, reduction='mean') 
        prior_loss = F.mse_loss( y, mu_y, reduction='mean') #+ ref_loss
        
        #print(y.shape, y_mask.shape,mu_y_masked.shape)

        if y_mask.shape[2] != y.shape[2]:
            print(y_mask.shape, y.shape)
            y_mask = torch.ones(y.shape[0],1,y.shape[2]).cuda()

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y_masked)

        return dur_loss, prior_loss, diff_loss

def torch_hub_model(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> GradTTSVitE2E:
    r"""
    
    """
    import params
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

    
    model = GradTTSVitE2E(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['model'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
