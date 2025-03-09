# Imports
import sys
import awkward as ak
import numpy as np
import argparse
import uproot
import os
from utils import _pad, compute_new_features, compute_invariant_masses, tau_tau_preprocessing, mass_plot, SVFit_input

# Parse arguments
parser = argparse.ArgumentParser('Running Custom NanoAODs pre-processing for Tau Pair Mass Transformer input making.')
parser.add_argument('-i', '--input'  , type=str, default='/eos/cms/store/group/phys_higgs/HLepRare/HTT_skim_v1/Run2_2018/DYJetsToLL_M-50-madgraphMLM/nanoHTT_0.root', help='Input file')
parser.add_argument('-f', '--flat'  , type=int, default=0, help='0 if pre-processing a flat mass sample')
parser.add_argument('-p', '--pairType' , type=str, default="tau_tau", help='pairType')
parser.add_argument('-t', '--test'   , type=bool, default=True, help='If True writes also designed dataset for running SVFit algorithm later')
parser.add_argument('-pT', '--pTcut', type=float, default=20.0, help='pT threshold')
args = parser.parse_args()

filepath = args.input
parts = filepath.split('/')
sample = parts[-2]
root_file = parts[-1][:-5]
flat = args.flat
pairType = args.pairType
test = args.test
pT_threshold =  args.pTcut
is_true_tau = False if "TTT" in filepath else True

columns_to_load = ["Tau_pt", "Tau_eta", "Tau_phi", "Tau_mass", "Tau_decayMode", "Tau_genPartFlav", "Tau_genPartIdx",             
                  "Tau_dxy", "Tau_dz", "Tau_ptCorrPNet", "Tau_rawPNetVSjet", "Tau_rawDeepTau2018v2p5VSjet", "Tau_charge", 
                  "Tau_leadTkDeltaEta", "Tau_leadTkDeltaPhi", "Tau_leadTkPtOverTauPt",                                           # Tau collection                                                      
                  "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass", "Electron_genPartFlav", "Electron_genPartIdx", # Electron collection
                  "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_genPartFlav", "Muon_genPartIdx",                         # Muon collection                                   
                  "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId", "GenPart_genPartIdxMother",       # GenPart collection
                  "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",                                                                    # Jet collection
                  "GenVisTau_*", "TauProd*"]
columns_to_load.append("PFMET*" if 'SUSY' in filepath else "MET*")

# ____________________________________________FEATURES________________________________________________

max_num_tau_products = 10
max_num_jets = 3
tauprod_features = ['logpt', 'eta', 'phi', 'pt_ratio', 'tauIdx', 'is_ele', 'is_mu', 'is_pho', 'is_pi', 'is_k'] 
tau_features = ['logpt', 'eta', 'phi', 'mass', 'dxy', 'dz', 'ptCorrPNet', 'rawPNetVSjet', 'rawDeepTau2018v2p5VSjet', 'charge', 
                'dM_0', 'dM_1', 'dM_2', 'dM_10', 'dM_11', 'leadTkDeltaEta', 'leadTkDeltaPhi', 'leadTkPtOverTauPt'] 
jet_features = ['logpt', 'eta', 'phi', 'mass'] 
extra_variables = ['inv_mass_reco', 'inv_mass_target']
if 'MET*' in columns_to_load:
    met_features = ['MET_logpt', 'MET_phi', 'MET_covXX', 'MET_covXY', 'MET_covYY', 'MET_significance', 'MET_sumEt', 'MET_sumPtUnclustered']
elif 'PFMET*' in columns_to_load:
    met_features = ['PFMET_logpt', 'PFMET_phi', 'PFMET_covXX', 'PFMET_covXY', 'PFMET_covYY', 'PFMET_significance', 'PFMET_sumEt', 'PFMET_sumPtUnclustered']

# for SVFIT
svfit_tau_features = ['pt', 'eta', 'phi', 'mass', 'decayMode'] 
svfit_met_features  = [feature.replace('logpt', 'pt') if 'logpt' in feature else feature for feature in met_features][0:5]

#____________________________________________________________________________________________________

log_file = sample + "/LOGS/Log_" + str(pairType) + "_" + str(root_file) + '.txt'
mass_plot_file = sample + "/PLOTS/gen_reco_invmass_" + str(pairType) + "_" + str(root_file) +".png"

with open(log_file, "w") as file:
    sys.stdout = file

    num_cores = os.cpu_count()

    data = []
    gen_taus = []

    for data_chunk in uproot.iterate(
        filepath + ":Events", 
        filter_name=columns_to_load, 
        step_size=10000,  
        how="zip", 
        num_workers=num_cores
    ):
        # Chunck process
        data_chunk["Tau"]= data_chunk["Tau"][ak.argsort(data_chunk["Tau"]["pt"], ascending=True)]
        indices_chunk, gen_taus_chunk, selected_jets_chunk = tau_tau_preprocessing(data_chunk, pT_threshold, is_true_tau)    
        data_chunk = data_chunk[indices_chunk]
        data_chunk = compute_new_features(data_chunk, selected_jets_chunk, pairType)
        data.append(data_chunk)
        gen_taus.append(gen_taus_chunk)

    data = np.concatenate(data)
    gen_taus = sum(gen_taus, [])
    print("Events with at least 2 gen matched taus with pt > 20 GeV: {count}".format(count = len(data)))


    # ____________________________________________________SAVE OUTPUTS_________________________________________________________


    x_taus = np.stack([ak.to_numpy(data.Tau[n]) for n in tau_features], axis=1)
    x_taus = np.transpose(x_taus, (0, 2, 1))
    x_met = np.stack([ak.to_numpy(data[n]) for n in met_features], axis=1)
    x_met = x_met.reshape((x_met.shape[0], 1, x_met.shape[1]))
    x_jets = np.stack([ak.to_numpy(_pad(data.Jet[n], maxlen = max_num_jets)) for n in jet_features], axis=1)
    x_jets = np.transpose(x_jets, (0, 2, 1))
    x_tauprod = np.stack([ak.to_numpy(_pad(data.TauProd[n], maxlen= max_num_tau_products)) for n in tauprod_features], axis=1)
    x_tauprod = np.transpose(x_tauprod, (0, 2, 1))
    target = np.array([[[subsublist[0]] for subsublist in sublist] for sublist in gen_taus])  
    inv_mass_reco, inv_mass_tar = compute_invariant_masses(data, gen_taus, pairType)
    mass_plot(inv_mass_reco, inv_mass_tar, flat, mass_plot_file)

    print("Tau Input: ", x_taus.shape)
    print("MET Input: ", x_met.shape)
    print("Jet Input: ", x_jets.shape)
    print("TauProd Input: ", x_tauprod.shape)
    print("GenPart Input: ", target.shape)   
    print("Reco Mass Input: ", inv_mass_reco.shape)
    print("Gen Mass Input: ", inv_mass_tar.shape)  

    np.savez(sample + "/DATA/" + pairType + "_" + root_file + ".npz", 
         x_taus = x_taus, x_tauprod = x_tauprod, x_met = x_met, x_jets = x_jets, target = target,
         inv_mass_reco = inv_mass_reco, inv_mass_tar = inv_mass_tar)
    print(f"Preprocessed data saved to {sample}/DATA/{pairType}_{root_file}.npz")

    if test: 
        svfit_data, ak_array = SVFit_input(data, svfit_tau_features, svfit_met_features, pairType)
        output_file = sample + "/DATA/SVFit_" + pairType + "_" + root_file + '.root'
        branches = {col: ak_array[col].type for col in svfit_data.columns} # Define the schema of the tree
        with uproot.recreate(output_file) as file2:  
            file2["Events"] = ak.zip({col: ak_array[col] for col in svfit_data.columns}) # Write the Awkward Array to a ROOT file
        print(f"SVFit Information saved to {sample}/DATA/SVfit_{pairType}_{root_file}.root")

    sys.stdout = sys.__stdout__

print(f"Log saved in {log_file}.")
    