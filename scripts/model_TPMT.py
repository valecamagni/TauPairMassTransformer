
#_________________________________LIBRERIES____________________________________

import torch
import torch.nn as nn
from typing import Optional
from utils_model import ParticleEmbedding, Encoder, EncoderBlock, FeedForwardBlock
#%% Section 1  - ENCODER




#_________________________________OUTPUT_______________________________________



class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, out_features, pooling_type='mean'):
        super(ProjectionLayer, self).__init__()
        self.pooling_type = pooling_type
        self.pooling_layer = self.get_pooling_layer(pooling_type)
        self.linear1 = nn.Linear(input_dim, 64)  
        self.linear2 = nn.Linear(64, 32)  
        self.linear_regression = nn.Linear(32, out_features)  # Regression output with 2 nodes

    def get_pooling_layer(self, pooling_type):
        if pooling_type == 'mean':
            return nn.AdaptiveAvgPool1d(1)
        elif pooling_type == 'max':
            return nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("Pooling type not supported.")

    def forward(self, encoder_output):
        pooled_output = self.pooling_layer(encoder_output.permute(0, 2, 1)).squeeze(-1)  # Pooling sull'output dell'encoder
        x = torch.relu(self.linear1(pooled_output))
        x = torch.relu(self.linear2(x))
        x_regression = self.linear_regression(x)

        return x_regression





#_________________________________________________MODEL____________________________________________________

class MultiScaleTransformer(nn.Module):

    def __init__(self, tau_embed: ParticleEmbedding, jets_embed: ParticleEmbedding, met_embed: ParticleEmbedding,
                 mass_embed: ParticleEmbedding, encoder_tau_tauprod_jets_met_mass: Encoder, 
                 projection_layer: ProjectionLayer, tauprod_embed: Optional[ParticleEmbedding] = None) -> None:
        super().__init__()
        self.tau_embed = tau_embed
        self.jets_embed = jets_embed
        self.met_embed = met_embed
        self.mass_embed = mass_embed
        self.encoder_tau_tauprod_jets_met_mass = encoder_tau_tauprod_jets_met_mass
        self.projection_layer = projection_layer
        self.tauprod_embed = tauprod_embed  # This will be None if tauprod is not used

    def embedding(self, taus, tauprod, jets, met, mass):
        """
        Creates embeddings and combines them for the transformer input.
        Handles the case where `tauprod` is not used.
        """
        # Embedding for all components
        taus = self.tau_embed(taus)
        jets = self.jets_embed(jets)
        met = self.met_embed(met)
        mass = self.mass_embed(mass)

        # Handle tauprod embedding
        if self.tauprod_embed is not None:
            tauprod = self.tauprod_embed(tauprod)
            objs = torch.concat([taus, tauprod, jets, met, mass], dim=1)
        else:
            objs = torch.concat([taus, jets, met, mass], dim=1)
        
        return objs

    def encode(self, objs, padding_mask, interaction_mask):
        return self.encoder_tau_tauprod_jets_met_mass(objs, padding_mask, interaction_mask)
    
    def project(self, x):
        return self.projection_layer(x)

    # for ONNX export
    def forward(self, taus, jets, met, mass, tauprods, padding_masks):
        out_embedding = self.embedding(taus, tauprods, jets, met, mass)
        encoder_output, _ = self.encode(out_embedding, padding_masks, None)
        proj = self.project(encoder_output)
        return proj 
    
    


def build_transformer(input_features_tau: int, input_features_tauprod: Optional[int], input_features_jets: int, 
                      input_features_met: int, input_features_mass: int, out_features: int, embed_dim: [], 
                      use_tauprod: bool, N_enc: int=1, h: int=8, dropout: float=0.1, d_ff: int=512) -> MultiScaleTransformer:
    """
    Builds the transformer model. Allows optional inclusion of tauprod with different input features.
    """
    # Create the embedding layers
    tau_embed = ParticleEmbedding(input_features_tau, embed_dim)
    jets_embed = ParticleEmbedding(input_features_jets, embed_dim)
    met_embed = ParticleEmbedding(input_features_met, embed_dim)
    mass_embed = ParticleEmbedding(input_features_mass, embed_dim)
    
    tauprod_embed = None  # Default: tauprod is not used
    if use_tauprod:
        tauprod_embed = ParticleEmbedding(input_features_tauprod, embed_dim)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N_enc):
        feed_forward_block = FeedForwardBlock(embed_dim[-1], d_ff, dropout)
        encoder_block = EncoderBlock(feed_forward_block, embed_dim[-1], h, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the encoder 
    encoder_tau_tauprod_jets_met_mass = Encoder(nn.ModuleList(encoder_blocks))

    # Create the projection layer
    proj = ProjectionLayer(embed_dim[-1], out_features)
    
    # Create the transformer
    transformer = MultiScaleTransformer(
        tau_embed, jets_embed, met_embed, mass_embed, encoder_tau_tauprod_jets_met_mass, proj, tauprod_embed
    )
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    return transformer


