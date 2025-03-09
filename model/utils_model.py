#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:23 2023

@author: valentinacamagni
"""

#_________________________________LIBRERIES____________________________________

import torch
import torch.nn as nn
from typing import Optional


# _____________________________INPUT EMBEDDING_________________________________


class ParticleEmbedding(nn.Module):
    """
    ParticleEmbedding allows to convert the original sentence (event) into words 
    (particles) or vectors of 128 dimension, each one encoded from the input particle 
    features using 3-layer MLP with (128, 512, 128) nodes, each layer with GELU and LN  
    in between for normalization.
    
    Args:
        input_dim : number of input features for each particle in the event 
        dims : list of nodes for each of the 3-layer MLP
               the last element of the list is the number of expected features 
               in the encoder/decoder inputs (in the paper d_model): 128, also
               called 'embedding size'
        
    """
    
    # Constructor 
    def __init__(self, input_dim, dims, activation='gelu'):
        super().__init__()

        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        # x: (batch, seq_len, input_features)
        return self.embed(x) # (batch, seq_len, d_model)



# __________________________INTERACTION EMBEDDING______________________________
#%%

class InteractionsEmbedding(nn.Module):
    """
    InteractionsEmbedding allows to convert the interaction features into 
    vectors of 8 dimension, each one encoded using a 4-layer pointwise 1D convolution 
    with (64, 64, 64, 8) channels with GELU nonlinearity and batch normalization 
    in between to yield a d' = 8 dimensional interaction matrix
    
    Args:
        input_dim : number of input paire-wise features for each pair of particles 
                    in the event 
        dims : list of nodes for each of the 4-layer pointwise 1D convolution
               the last element of the list is the number of expected features 
               in the encoder MHA mask
    
    Interaction Features (4): 
        lnDELTA, lnKT, lnZ, lnM2
    """
    
    # Constructor 
    def __init__(self, pair_dim, dims, activation='gelu'):
        super().__init__()
        # Module list is a way to organize a list of modules 
        # in this case we need 4 times the same list of layers 
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(pair_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            pair_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        # x: (batch, pair_features, seq_len, seq_len)
        if x != None:
            batch_size, pair_features, seq_len, seq_len_2 = x.size()
            x = x.view(-1, pair_features, seq_len*seq_len_2)
            x = self.embed(x)
            x = x.view(-1, 8, seq_len, seq_len_2)
        else: 
            x = x
        return x # (batch, d', seq_len, seq_len)


#%%

#_____________________________ADD & NORM LAYER_________________________________



class LayerNormalization(nn.Module):
    
    """
    Every word of a sentence has some features, numbers.
    For each one we calculate mean and variance independently from the other words 
    of the sentence. Then we calculate the new values for each word through its 
    own mean and variance.
    We want the model to have the possibility to amplify these values so 
    2 PARAMETERS are learned during training:
        - ALPHA (multiplicative)
        - BIAS (additive) 
        
        
    Note: it is applied to each batch for each word in the sentence, 
          along the embedding dimension is calculated the mean and the variance, 
          each number is normalized through them
        
    
    """
    
    # Constructor
    def __init__(self, eps:float=10**-6) -> None:
        
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1) 
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias




#____________________________FEED-FORWARD LAYER________________________________



class FeedForwardBlock(nn.Module):
    
    """
    Fully-connected layer that the model uses both in the encoder and decoder.
    It consists of two linear transformations with a ReLU activation in between:

                        FFN(x) = max(0, xW1 + b1)W2 + b2
                        
    Args: 
        d_model : embedding_size 
        d_ff : inner-layer dimensionality (up to factor 4 with respect to d_model)
               (d_model = 128 --> d_ff = 512)
        dropout : 0.1 
    """

    # Constructor 
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1 (because bias for default is True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2 (because bias for default is True)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        #                        linear_1                   linear_2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))




#____________________________RESIDUAL CONNECTION_______________________________


# between the Add&Norm and the previous layer 
class ResidualConnection(nn.Module):
    
        """
        Output from the first Add&Norm is taken from the FF, pass through the second
        Add&Norm and then is combined with this output. 
        ResidualConnection is the layer that manages this skip connection 
    
        """
    
        # Constructor
        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout) 
            self.norm = LayerNormalization()
    
        def forward(self, x, sublayer): # sublayer: previous layer 
            return x + self.dropout(sublayer(self.norm(x)))



#________________________________ENCODER_______________________________________



class EncoderBlock(nn.Module):
    
    """
    All the previous layers are combined in the encoder block, repeated N times
    the output of the previous is sent to the next one
    The output of the last one is sent to the decoder 
    Encoder will contain: 
        - Multi-head attention
        - 2 Add & Norm 
        - 1 Feed Forward 
    """

    def __init__(self, feed_forward_block: FeedForwardBlock, d_model: int, h: int, dropout: float) -> None:

        super().__init__()

        self.self_attention_block = nn.MultiheadAttention(d_model, h, dropout, batch_first = True)
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # we need two residual connections

    def forward(self, x, padding_mask, interaction_mask):
        if interaction_mask != None:
            batch_size = interaction_mask.size(0)
            h = interaction_mask.size(1)
            L =  interaction_mask.size(2)
            S =  interaction_mask.size(3)
            interaction_mask = interaction_mask.view(batch_size * h, L, S)

            x, attn_weights = self.self_attention_block(x, x, x, key_padding_mask = padding_mask, attn_mask = interaction_mask, average_attn_weights = False)
        else:
            x, attn_weights = self.self_attention_block(x, x, x, key_padding_mask = padding_mask, average_attn_weights = False)
        x = self.residual_connections[0](x, lambda x: x)  # Pass only attention layer's output (without weights)
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x, attn_weights
    
    
    
# The encoder is made up of several of n **EncoderBlock**
class Encoder(nn.Module):
    # list of layers, applied one after another, so we give a module list 
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization() # at the end LN

    def forward(self, x, padding_mask, interaction_mask):
        all_attn_weights = []
        for layer in self.layers: 
            x, attn_weights = layer(x, padding_mask, interaction_mask)
            all_attn_weights.append(attn_weights)
        x = self.norm(x) #Last Norm
        return x, all_attn_weights



#_______________________________________________________CROSS_ATTENTION_______________________________________________________



class Cross_Attention(nn.Module):

    def __init__(self, feed_forward_block: FeedForwardBlock, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.h = h
        self.cross_attention_block = nn.MultiheadAttention(d_model, h, dropout, batch_first = True)
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 skip connections 

    def forward(self, encoder_output_tauprod, encoder_output_tau, st_padding):
        if st_padding!= None:
            batch_size = st_padding.size(0)
            L =  st_padding.size(1)
            S =  st_padding.size(2)
            attn_mask_expanded = st_padding.unsqueeze(1).repeat(1, self.h, 1, 1)
            attn_mask_expanded = attn_mask_expanded.view(batch_size *self.h, L, S)

            x, attn_weights = self.cross_attention_block(encoder_output_tau, encoder_output_tauprod, encoder_output_tauprod, attn_mask = attn_mask_expanded, average_attn_weights = False)
        else: 
            x, attn_weights = self.cross_attention_block(encoder_output_tau, encoder_output_tauprod, encoder_output_tauprod, average_attn_weights = False)
        x = self.residual_connections[0](x, lambda x: x) 
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x, attn_weights



# N times the **CrossBlock**
class Cross(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, encoder_output_tauprod, encoder_output_tau, sc_padding):
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(encoder_output_tauprod, encoder_output_tau, sc_padding)
            all_attn_weights.append(attn_weights)
        x = self.norm(x)
        return x, all_attn_weights







def safe_sqrt(x, epsilon=1e-9):
    return torch.sqrt(torch.clamp(x, min=epsilon))



def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    px1 = pt1 * torch.cos(phi1)
    py1 = pt1 * torch.sin(phi1)
    pz1 = pt1 * torch.sinh(eta1)
    E1 = safe_sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
    
    px2 = pt2 * torch.cos(phi2)
    py2 = pt2 * torch.sin(phi2)
    pz2 = pt2 * torch.sinh(eta2)
    E2 = safe_sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)
    
    mass_squared = (E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2
    return safe_sqrt(mass_squared)




# CUSTOM LOSS
class CustomLoss(nn.Module):
    def __init__(self, l1_weight_tau=float, l1_weight_mass=float): 
        super(CustomLoss, self).__init__()
        self.l1_weight_tau = l1_weight_tau
        self.l1_weight_mass = l1_weight_mass
        self.l1_loss = nn.L1Loss()


    def forward(self, proj, target, reco_taus, target_mass):
        # Compute invariant mass
        predicted_mass = invariant_mass(torch.exp(proj[:, 0]), reco_taus[:, 0, 1], reco_taus[:, 0, 2], 0, torch.exp(proj[:, 1]), reco_taus[:, 1, 1], reco_taus[:, 1, 2], 0)
        # Compute L1 loss between predicted_mass e target_mass
        l1_loss = self.l1_loss(proj, target)

        # L1 loss between predicted_mass e target_mass
        l1_mass_loss = self.l1_loss(predicted_mass, target_mass)

        # Weighted combination of the different terms in the loss
        loss = self.l1_weight_tau * l1_loss + self.l1_weight_mass * l1_mass_loss

        return loss, self.l1_weight_mass * l1_mass_loss, self.l1_weight_tau*l1_loss






