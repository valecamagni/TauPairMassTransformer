
#_________________________________LIBRERIES____________________________________

import torch
import torch.nn as nn
from typing import Optional
from model_TPMT import ParticleEmbedding, Encoder, FeedForwardBlock, EncoderBlock
from zuko.flows import Flow, UnconditionalDistribution, UnconditionalTransform
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.distributions import DiagNormal
from zuko.transforms import RotationTransform



def get_obj_type_tensor(tauprod_included=False, num_tauprods=10, num_taus=2, num_getti=3, num_met=1, num_mass=1):

    if tauprod_included:
        num_objects = num_tauprods + num_taus + num_getti + num_met + num_mass
        num_categories = 5
    else: 
        num_objects =  num_taus + num_getti + num_met + num_mass
        num_categories = 4

    o = torch.zeros((num_objects, num_categories), dtype=torch.float32)

    if tauprod_included:
        o[:num_tauprods, 0] = 1.0  # tauprods
        o[num_tauprods:num_tauprods+num_taus, 1] = 1.0  # taus
        start_idx = num_tauprods + num_taus
        o[start_idx:start_idx + num_getti, 2] = 1.0  # getti
        start_idx += num_getti
        o[start_idx:start_idx + num_met, 3] = 1.0  # MET
        start_idx += num_mass
        o[start_idx:, 4] = 1.0  #reco mass
    else: 
        o[:num_taus, 0] = 1.0  # taus
        o[num_taus:num_taus+num_getti, 1] = 1.0  # getti
        start_idx = num_taus + num_getti
        o[start_idx:start_idx + num_met, 2] = 1.0  # MET
        start_idx += num_mass
        o[start_idx:, 3] = 1.0  #reco mass

    return o




class ScoreNetwork(nn.Module):
    def __init__(self, d_model, output_dim = int(), output_net = [64, 64]):
        super(ScoreNetwork, self).__init__()

        _layers = [nn.Linear(d_model, output_net[0]), nn.SELU()]
        for i in range(len(output_net) - 1):
            _layers.append(nn.Linear(output_net[i], output_net[i + 1]))
            _layers.append(nn.SELU())
            
        _layers.append(nn.Linear(output_net[-1], 1))
        _layers.append(nn.Sigmoid())  # To get a score between 0 and 1
        self.output_network = nn.Sequential(*_layers)
        self.proj_out = nn.Linear(d_model, output_dim)


    def forward(self, encoder_output, padding_mask):
        score = self.output_network(encoder_output)
        weighted_sum = torch.sum((encoder_output * score * padding_mask.unsqueeze(2)), axis=1) # Calcola il context vector tramite somma pesata degli output
        context_vector = self.proj_out(weighted_sum)

        return context_vector
    



class FlowNet(nn.Module):
    def __init__(self, context_dim, out_features, flow_nlayers=4, flow_hidden_net=(64,64), flow_coupling=True):
        super(FlowNet, self).__init__()

        flow_transf_layers = []
        for i in range(flow_nlayers-1):
            flow_transf_layers.append(MaskedAutoregressiveTransform(features=out_features, 
                                                                    context=context_dim, 
                                                                    hidden_features=flow_hidden_net, 
                                                                    passes=2 if flow_coupling else None))
            # Let's add a rotation of the features
            flow_transf_layers.append(UnconditionalTransform(RotationTransform, torch.randn(out_features, out_features)))
        # adding last transformation
        flow_transf_layers.append(MaskedAutoregressiveTransform(features=out_features, 
                                                                    context=context_dim, 
                                                                    hidden_features=flow_hidden_net, 
                                                                    passes=2 if flow_coupling else None))
  
        self.flow = Flow(
                transform=flow_transf_layers,
                base=UnconditionalDistribution(
                    DiagNormal,
                    torch.zeros(out_features),
                    torch.ones(out_features),
                    buffer=True,
                ),
            )

    def forward(self, conditioning_vector):
        return self.flow(conditioning_vector)
    



#_________________________________________________MODEL____________________________________________________

class NFTransformer(nn.Module):

    def __init__(self, tau_embed: ParticleEmbedding, jets_embed: ParticleEmbedding, met_embed: ParticleEmbedding,
                 mass_embed: ParticleEmbedding, encoder_tau_tauprod_jets_met_mass: Encoder, score_net: ScoreNetwork, 
                 flow: FlowNet, tauprod_embed: Optional[ParticleEmbedding] = None) -> None:
        super().__init__()
        self.tau_embed = tau_embed
        self.jets_embed = jets_embed
        self.met_embed = met_embed
        self.mass_embed = mass_embed
        self.encoder_tau_tauprod_jets_met_mass = encoder_tau_tauprod_jets_met_mass
        self.tauprod_embed = tauprod_embed  # This will be None if tauprod is not used
        self.score_network = score_net
        self.flow =flow
        
        if self.tauprod_embed is not None:
            self.tauprod_included = True
        else: 
            self.tauprod_included = False
        
        self.register_buffer('obj_type', get_obj_type_tensor(self.tauprod_included)) 

    def embedding(self, taus, tauprod, jets, met, mass):
 
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
        
        batch_size = objs.size(0)
        labels = self.obj_type.expand(batch_size, *list(self.obj_type.shape))
        objs = torch.cat([objs, labels], dim=-1)

        return objs

    def encode(self, objs, padding_mask, interaction_mask):

        num_objects = objs.size(1)
        encoder_output, _ = self.encoder_tau_tauprod_jets_met_mass(objs, padding_mask, interaction_mask)
        context_vector = self.score_network(encoder_output, padding_mask)
        assert context_vector.shape[1] == num_objects, f"Il context vector deve avere dimensione {num_objects} per essere usato nel flow."
        return context_vector

    def flow_net(self, context_vector):
        return self.flow(context_vector)
        


    


def build_NF(input_features_tau: int, input_features_tauprod: Optional[int], input_features_jets: int, 
                      input_features_met: int, input_features_mass: int, out_features: int, embed_dim: [], 
                      use_tauprod: bool, num_objects: int, N_enc: int=2, h: int=8, dropout: float=0.1, d_ff: int=512) -> NFTransformer:

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
        feed_forward_block = FeedForwardBlock(embed_dim[0], d_ff, dropout)
        encoder_block = EncoderBlock(feed_forward_block, embed_dim[0], h, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the encoder 
    encoder_tau_tauprod_jets_met_mass = Encoder(nn.ModuleList(encoder_blocks))

    score_net = ScoreNetwork(embed_dim[0], num_objects)
    flow = FlowNet(num_objects, out_features)
    
    # Create the transformer
    nftransformer = NFTransformer(
        tau_embed, jets_embed, met_embed, mass_embed, encoder_tau_tauprod_jets_met_mass, score_net, flow, tauprod_embed
    )
    
    # Initialize the parameters
    for p in nftransformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    return nftransformer





