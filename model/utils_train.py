#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:22:23 2023

@author: valentinacamagni
"""

#_________________________________LIBRERIES____________________________________

import numpy as np
import torch
import torch.nn.functional as F
from utils_model import CustomLoss
from model_TPMT import build_transformer
from model_NF import build_NF
import matplotlib.pyplot as plt



def get_NFmodel(config):

    model = build_NF(config['input_features_taus'], config['input_features_tauprod'], config['input_features_jets'],
                              config['input_features_met'], config['input_features_mass'], config['out_features'], 
                              config['embed_dim'], config['use_tauprod'], config['num_objects'])
    
    return model

def get_TPMTmodel(config):

    model = build_transformer(config['input_features_taus'], config['input_features_tauprod'], config['input_features_jets'],
                              config['input_features_met'], config['input_features_mass'], config['out_features'], 
                              config['embed_dim'], config['use_tauprod'])
    
    return model



def TPMTvalidation(model, validation_ds, device, use_tauprod, tau_loss_weight, mass_loss_weight):
    
    """
    Inference during training, every epoch
    """
    
    model.eval()

    criterion = CustomLoss(tau_loss_weight, mass_loss_weight)
    val_running_loss = 0.0
    val_running_mass_loss = 0.0
    val_running_tau_loss = 0.0

    # disabling the gradient calculation for every tensor will run inside this block
    with torch.no_grad():
        for batch in validation_ds:

            taus = batch['tau'].float().to(device)
            jets = batch['jets'].float().to(device)
            met = batch['met'].float().to(device)
            mass = batch['inv_mass_reco'].float().to(device)
            if use_tauprod:
                tauprods = batch['tauprod'].float().to(device)
            else:
                tauprods = torch.tensor([]).to(device)
            padding_masks = batch['padding_masks'].to(device)
            target, mass_target = batch['tar'].float().to(device), batch['inv_mass_tar'].float().to(device)
            # Forward pass through model
            out_embedding = model.embedding(taus, tauprods, jets, met, mass)
            encoder_tau_tauprod_jets_met_mass_out, _ = model.encode(out_embedding, padding_masks, None)
            proj = model.project(encoder_tau_tauprod_jets_met_mass_out)

            # Attention tracking
            #attn_tauprod_iterations.append(torch.mean(attn_tauprod[0], dim=0))
            #attn_tau_iterations.append(torch.mean(attn_tau[0], dim=0))
            #attn_cross_iterations.append(torch.mean(attn_cross[0], dim=0))

            # Compute loss
            val_loss, val_mass_loss, val_tau_loss = criterion(proj, target, taus, mass_target)

            val_running_loss += val_loss.item()
            val_running_mass_loss += val_mass_loss.item()
            val_running_tau_loss += val_tau_loss.item()

    # Mean loss per epoch
    epoch_val_loss = val_running_loss / len(validation_ds)
    epoch_val_mass_loss = val_running_mass_loss / len(validation_ds)
    epoch_val_tau_loss = val_running_tau_loss / len(validation_ds)

    
    return epoch_val_loss, epoch_val_mass_loss, epoch_val_tau_loss


def NFvalidation(model, validation_ds, device, use_tauprod):
    
    """
    Inference during training, every epoch
    """
    
    model.eval()

    criterion = CustomLoss()
    val_running_loss = 0.0
    val_running_nll_loss = 0.0
    val_running_mmd_loss = 0.0
    val_running_mass_loss = 0.0
    val_running_tau_loss = 0.0

    # disabling the gradient calculation for every tensor will run inside this block
    with torch.no_grad():
        for batch in validation_ds:

            taus = batch['tau'].float().to(device)
            jets = batch['jets'].float().to(device)
            met = batch['met'].float().to(device)
            mass = batch['inv_mass_reco'].float().to(device)
            if use_tauprod:
                tauprods = batch['tauprod'].float().to(device)
            else:
                tauprods = torch.tensor([]).to(device)
            padding_masks = batch['padding_masks'].to(device)
            target, mass_target = batch['tar'].float().to(device), batch['inv_mass_tar'].float().to(device)
            # Forward pass through model
            out_embedding = model.embedding(taus, tauprods, jets, met, mass)
            context_vector = model.encode(out_embedding, padding_masks, None)
            generated_samples = model.flow_net(context_vector).rsample((1,))
            log_prob = model.flow_net(context_vector).log_prob(target)
            val_nll_loss = - log_prob.mean()
            val_mmd_loss = compute_mmd(generated_samples.squeeze(0), target)
            _, val_mass_loss = criterion(generated_samples.squeeze(0), target, taus, mass_target)

            val_loss = val_nll_loss + val_mmd_loss + val_mass_loss

            val_running_loss += val_loss.item()
            val_running_nll_loss += val_nll_loss.item()
            val_running_mmd_loss += val_mmd_loss.item()
            val_running_mass_loss += val_mass_loss.item()
            #val_running_tau_loss += val_tau_loss.item()

    # Mean loss per epoch
    epoch_val_loss = val_running_loss / len(validation_ds)
    epoch_val_nll_loss = val_running_nll_loss / len(validation_ds)
    epoch_val_mmd_loss = val_running_mmd_loss / len(validation_ds)
    epoch_val_mass_loss = val_running_mass_loss / len(validation_ds)
    #epoch_val_tau_loss = val_running_tau_loss / len(validation_ds)

    return epoch_val_loss, epoch_val_nll_loss, epoch_val_mmd_loss, epoch_val_mass_loss#, epoch_val_tau_loss





def plot_losses(train_loss, val_loss, xlabel, ylabel, title, filename, log_scale=False):
 
    fig = plt.figure(figsize=(10, 7), dpi=100)
    plt.grid(True, which='major', axis="x", linestyle=':')
    plt.grid(True, which='major', axis="y", linestyle=':')
    
    if log_scale:
        plt.plot(np.log(train_loss), label='Training Loss', color='blue')
        plt.plot(np.log(val_loss), label='Validation Loss', color='red')
        plt.ylabel(f'Log({ylabel})', fontsize=12)
    else:
        plt.plot(train_loss, label='Training Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='red')
        plt.ylabel(ylabel, fontsize=12)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig(filename)
    plt.close(fig) 




def plot_attention_maps(attention_layer, layer_index, config, n_rows=2, n_cols=4):
    """Plots attention maps for a given attention layer."""
    numpy_tensor = attention_layer.detach().cpu().numpy()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8))
    axes = axes.flatten()

    # Set color limits based on layer index
    if layer_index == 2:
        vmin, vmax = 0.0, 0.5
    else:
        vmin, vmax = 0.0, 1.0       

    # Plot each attention head as a heatmap
    for i in range(numpy_tensor.shape[0]):
        im = axes[i].imshow(numpy_tensor[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Head {i + 1}')
        fig.colorbar(im, ax=axes[i])

        # Annotate the heatmap
        for k in range(numpy_tensor.shape[1]):
            for j in range(numpy_tensor.shape[2]):
                value = numpy_tensor[i, k, j]
                if layer_index == 2:
                    if 0.2 < value < 0.5:
                        color = 'white'
                    else:
                        color = 'black' if value >= 0.5 else 'none'
                elif layer_index == 1:
                    if 0.2 < value < 0.5:
                        color = 'white'
                    else:
                        color = 'black' if value >= 0.5 else 'none'
                else:  # layer_index == 0
                    if 0.2 < value < 0.35:
                        color = 'white'
                    else:
                        color = 'black' if value >= 0.35 else 'none'

                if color != 'none':
                    axes[i].text(j, k, f'{value:.2f}', ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.suptitle('Attention Map', y=1.02)
    
    layer_names = ['tauprod', 'tau', 'cross']
    plt.savefig(config['model_folder'] + f'/attn_{layer_names[layer_index]}_{layer_index + 1}.png')
    plt.close(fig)




def compute_mmd(x, y, kernel='rbf', bandwidth=1.0):
    """
    Compute the MMD loss between two sets of samples using an RBF kernel.
    
    :param x: Tensor of shape [batch_size, num_features], generated samples.
    :param y: Tensor of shape [batch_size, num_features], real samples (targets).
    :param kernel: The kernel type ('rbf' for Radial Basis Function).
    :param bandwidth: Kernel bandwidth for the RBF.
    :return: MMD loss value.
    """
    if kernel == 'rbf':
        # Compute pairwise squared distances
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
        ry = torch.diag(yy).unsqueeze(0).expand_as(yy)

        K_xx = torch.exp(-((rx.t() + rx - 2 * xx) / (2 * bandwidth ** 2)))
        K_yy = torch.exp(-((ry.t() + ry - 2 * yy) / (2 * bandwidth ** 2)))
        K_xy = torch.exp(-((rx.t() + ry - 2 * xy) / (2 * bandwidth ** 2)))

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")



