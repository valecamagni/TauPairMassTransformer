from config import get_config, get_weights_file_path
from utils_train import get_TPMTmodel
from utils_datasets import get_test_ds 
from utils_model import invariant_mass
import torch
import time 
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
use_tauprod = config['use_tauprod']

te, dataset = get_test_ds(base_folder)
model = get_TPMTmodel(config).to(device)

epoch = str(config["weight_epoch"])
model_filename = get_weights_file_path(config, epoch)
print(model_filename)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


def test(model, test_ds, dataset, device, use_tauprod): 

    # masses
    reco_mass, predicted_mass, target_mass = [], [], []
    # reco 4-momentum
    R_pt_1, R_eta_1, R_phi_1 = [], [], []
    R_pt_2, R_eta_2, R_phi_2 = [], [], []
    # predicted
    P_pt_1, P_pt_2 = [], []
    # target
    T_pt_1, T_pt_2 = [], []
    # times
    processing_times = []

    model.eval()

    # disabling the gradient calculation for every tensor will run inside this block
    with torch.no_grad():
        for batch in test_ds:

            start_time = time.time()

            taus = batch['tau'].float().to(device)
            #print(taus.shape)
            jets = batch['jets'].float().to(device)
            #print(jets.shape)
            met = batch['met'].float().to(device)
            #print(met.shape)
            mass = batch['inv_mass_reco'].float().to(device)
            #print(mass.shape)
            if use_tauprod:
                tauprods = batch['tauprod'].float().to(device)
                #print(tauprods.shape)
            else: 
                tauprods = torch.tensor([]).to(device)
            padding_masks = batch['padding_masks'].to(device)
            #print(padding_masks.shape)
            target, mass_target = batch['tar'].float().to(device), batch['inv_mass_tar'].float().to(device)
            # Forward pass through model
            out_embedding = model.embedding(taus, tauprods, jets, met, mass)
            encoder_tau_tauprod_jets_met_mass_out, _ = model.encode(out_embedding, padding_masks, None)
            proj = model.project(encoder_tau_tauprod_jets_met_mass_out)

            R_pt_1.append(torch.exp(taus[0, 0, 0]).item())
            R_pt_2.append(torch.exp(taus[0, 1, 0]).item())
            R_eta_1.append(taus[0, 0, 1].item())
            R_eta_2.append(taus[0, 1, 1].item())
            R_phi_1.append(taus[0, 0, 2].item())
            R_phi_2.append(taus[0, 1, 2].item())

            P_pt_1.append(torch.exp(proj[0, 0]).item())
            P_pt_2.append(torch.exp(proj[0, 1]).item())
            T_pt_1.append(torch.exp(target[0, 0]).item())
            T_pt_2.append(torch.exp(target[0, 1]).item())

            tpmt_mass = invariant_mass(torch.exp(proj[0, 0]), taus[0, 0, 1], taus[0, 0, 2], torch.tensor(0.0), 
                                       torch.exp(proj[0, 1]), taus[0, 1, 1], taus[0, 1, 2], torch.tensor(0.0))

            predicted_mass.append(tpmt_mass.item())   
            target_mass.append(mass_target[0].item())
            reco_mass.append(mass[0].item())

            end_time = time.time()  # End timing
            processing_time = end_time - start_time  # Compute the time taken for this batch
            processing_times.append(processing_time)
        
    
    # Mean processing time
    mean_processing_time = sum(processing_times) / len(processing_times)
    print(f"Mean processing time over all batches: {mean_processing_time:.6f} seconds") 

    results = pd.DataFrame({'reco_mass': reco_mass, 'pred_mass': predicted_mass, 'tar_mass': target_mass,   
                            'R_pt_1': R_pt_1, 'R_pt_2': R_pt_2, 'R_eta_1': R_eta_1, 'R_eta_2': R_eta_2, 'R_phi_1': R_phi_1, 'R_phi_2': R_phi_2, 
                            'P_pt_1': P_pt_1, 'P_pt_2': P_pt_2, 
                            'T_pt_1': T_pt_1, 'T_pt_2': T_pt_2
                            })

    os.makedirs(config['model_folder']+ '/RESULTS', exist_ok=True)
    os.makedirs(config['model_folder']+ '/RESULTS/'+ dataset[0], exist_ok=True)

    results.to_csv(config['model_folder'] + '/RESULTS/' + dataset[0] + '/results_inference_' + config['test_pairType'] +'.csv')



test(model, te, dataset, device, use_tauprod)
    
    
    
    
    
    
    
    
