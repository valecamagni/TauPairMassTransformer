import numpy as np
import awkward as ak
from numba import njit
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import argparse




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



@njit
def has_opposite_charges(charges):
    """Checks if the input list contains exactly two distinct charges"""
    unique_values = set()
    for charge in charges:
        unique_values.add(charge)
    return len(unique_values) == 2



@njit
def tau_tau_preprocessing(dataset, pt_threshold, is_true_tau = True):
    """
    Selects events with exactly two taus (either true - genmatched - or fake) meeting certain conditions such as tau pt, 
    decay mode, and charge properties. Also identifies jets that are not too close to the selected taus using the delta R criterion. 
    Returns indices of selected events, info of the corrisponding gentaus, and associated jets.
    """
    indices = []
    gen_taus = []
    selected_jets = []

    for num, event in enumerate(dataset):
        if len(event.Tau)>1:
            count = 0
            for tau in event.Tau:
                if is_true_tau:
                    if (tau.genPartFlav==5) and (tau.pt>pt_threshold) and (tau.genPartIdx in [0,1]) and (tau.decayMode not in [5,6]):
                        count += 1
                else:
                    if (tau.genPartFlav not in [3,4,5]) and (tau.pt > pt_threshold) and (tau.decayMode not in [5,6]):
                        count += 1

            if (count!=2) or (len(event.Tau.genPartFlav)>2) or (not has_opposite_charges(event.Tau.charge)): continue
            else: 
                gen_taus_event = []
                
                for n, tau in enumerate(event.Tau):
                    if is_true_tau: 
                        genpart_pdgId = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].pdgId
                        if abs(genpart_pdgId) == 15:
                            tau_pt = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].pt
                            tau_eta = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].eta
                            tau_phi = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].phi
                            tau_mass = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].mass
                            gen_taus_event.append([tau_pt, tau_eta, tau_phi, tau_mass])
                        else: continue
                    else:
                        min_idx, min_dr = match_gen_particle(tau, event.GenPart)

                        if (min_idx >= 0) and (min_dr < 0.4):
                            gen_taus_event.append([tau.pt, tau.eta, tau.phi, tau.mass])
                        else:
                            continue
                            
                if len(gen_taus_event) in [0,1]: continue
                else:
                    indices.append(num)
                    gen_taus.append(gen_taus_event)

                    jet_per_event = np.zeros(len(event.Jet), dtype=np.bool_)
                    for num, jet in enumerate(event.Jet): 
                        jet_eta, jet_phi = jet.eta, jet.phi
                        djt1 = delta_r(jet_eta, jet_phi, event.Tau[0].eta, event.Tau[0].phi)
                        djt2 = delta_r(jet_eta, jet_phi, event.Tau[1].eta, event.Tau[1].phi)
                        jet_per_event[num] = (djt1>0.4) & (djt2>0.4)

                    selected_jets.append(jet_per_event)

    return indices, gen_taus, selected_jets



@njit
def ele_tau_preprocessing(dataset, pt_threshold, is_true_tau = True):
    """
    Select events with exactly one electron and one tau that meet specific criteria. 
    The function applies selection based on transverse pt, origin of the particles (true or fake tau/electron), and tau decay mode. 
    It returns indices of selected events along with boolean masks for electrons, taus, and tau-related products.
    """
    indices = []
    mask_electrons = []
    mask_taus = []
    mask_tauprods = []

    for num, event in enumerate(dataset):
        if len(event.Electron)>0:
            count_ele = 0
            mask_ele = np.zeros(len(event.Electron), dtype=np.bool_)
            for num_ele, electron in enumerate(event.Electron):
                if is_true_tau: 
                    if (electron.genPartFlav == 15) and (electron.pt > pt_threshold):
                        count_ele = count_ele + 1
                        mask_ele[num_ele] = True
                else:
                    if (electron.genPartFlav not in [15]) and (electron.pt > pt_threshold):
                        count_ele = count_ele + 1
                        mask_ele[num_ele] = True
            if (count_ele == 1) and len(event.Tau)>0:
                count_tau = 0
                mask_tau = np.zeros(len(event.Tau), dtype=np.bool_)
                for num_tau, tau in enumerate(event.Tau):
                    if is_true_tau:
                        if (tau.genPartFlav == 5) and (tau.pt > pt_threshold) and (tau.genPartIdx in [0,1]) and (tau.decayMode not in [5,6]):
                            count_tau = count_tau + 1
                            mask_tau[num_tau] = True
                    else: 
                        if (tau.genPartFlav not in [3,4,5]) and (tau.pt > pt_threshold) and (tau.decayMode not in [5,6]):
                            count_tau = count_tau + 1
                            mask_tau[num_tau] = True

                if count_tau == 1:
                    indices.append(num)
                    mask_electrons.append(mask_ele)
                    mask_taus.append(mask_tau)
                    mask_tauprod = np.zeros(len(event.TauProd), dtype=np.bool_)
                    for num_t, tprod in enumerate(event.TauProd):
                        if tprod.tauIdx in [index for index, value in enumerate(mask_tau) if not value]:
                            mask_tauprod[num_t] = False
                        else:
                            mask_tauprod[num_t] = True
                    mask_tauprods.append(mask_tauprod)
            else: continue
    
    return indices, mask_electrons, mask_taus, mask_tauprods



@njit
def mu_tau_preprocessing(dataset, pt_threshold, is_true_tau = True ):
    """
    Select events with exactly one muon and one tau that meet specific criteria. 
    The function applies selection based on transverse pt, origin of the particles (true or fake tau/electron), and tau decay mode. 
    It returns indices of selected events along with boolean masks for muons, taus, and tau-related products.
    """
    indices = []
    mask_muons = []
    mask_taus = []
    mask_tauprods = []

    for num, event in enumerate(dataset):
        if len(event.Muon)>0:
            count_mu = 0
            mask_mu = np.zeros(len(event.Muon), dtype=np.bool_)
            for num_mu, muon in enumerate(event.Muon):
                if is_true_tau: 
                    if (muon.genPartFlav == 15) and (muon.pt > pt_threshold):
                        count_mu = count_mu + 1
                        mask_mu[num_mu] = True
                else:
                    if (muon.genPartFlav not in [15]) and (muon.pt > pt_threshold):
                        count_mu = count_mu + 1
                        mask_mu[num_mu] = True

            if (count_mu == 1) and len(event.Tau)>0:
                count_tau = 0
                mask_tau = np.zeros(len(event.Tau), dtype=np.bool_)
                for num_tau, tau in enumerate(event.Tau):
                    if is_true_tau:
                        if (tau.genPartFlav == 5) and (tau.pt > pt_threshold) and (tau.genPartIdx in [0,1]) and (tau.decayMode not in [5,6]):
                            count_tau = count_tau + 1
                            mask_tau[num_tau] = True
                    else: 
                        if (tau.genPartFlav not in [3,4,5]) and (tau.pt > pt_threshold) and (tau.decayMode not in [5,6]):
                            count_tau = count_tau + 1
                            mask_tau[num_tau] = True
                            
                if count_tau == 1:
                    indices.append(num)
                    mask_muons.append(mask_mu)
                    mask_taus.append(mask_tau)
                    mask_tauprod = np.zeros(len(event.TauProd), dtype=np.bool_)
                    for num_t, tprod in enumerate(event.TauProd):
                        if tprod.tauIdx in [index for index, value in enumerate(mask_tau) if not value]:
                            mask_tauprod[num_t] = False
                        else:
                            mask_tauprod[num_t] = True
                    mask_tauprods.append(mask_tauprod)
            else: continue
    
    return indices, mask_muons, mask_taus, mask_tauprods




@njit
def store_gen_level_information_plus_jets(dataset, pairType, is_true_tau = True):
    """
    It extracts relevant information about the gen-level taus, selected jets, and stores them for each event.
    The function distinguishes between true taus and reconstructed ones, and applies different logic depending on 
    the pairType (e.g., "ele_tau", "mu_tau").

    Returns:
    - indices (list): A list of indices corresponding to the events that have valid gen-level information.
    - gen_taus (list): A list of gen-level tau information for each valid event. Each entry contains a list of properties [pt, eta, phi, mass].
    - selected_jets (list): A list of boolean arrays, each indicating which jets are selected for an event based on the delta-R separation 
      from the tau and other relevant particles (electron or muon).
    """

    indices = []
    gen_taus = []
    selected_jets = []

    for num, event in enumerate(dataset):
        gen_taus_event = []

        # ______________
        # First tau 
        # ______________

        tau = event.Tau[0]

        if is_true_tau:
            pdgId_genpart_tau = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].pdgId

            if abs(pdgId_genpart_tau)==15: 
                tau_pt = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].pt
                tau_eta = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].eta
                tau_phi = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].phi
                tau_mass = event.GenPart[event.GenVisTau[tau.genPartIdx].genPartIdxMother].mass
                gen_taus_event.append([tau_pt, tau_eta, tau_phi, tau_mass])
        else:
            min_idx, min_dr = match_gen_particle(tau, event.GenPart)

            if (min_idx >= 0) and (min_dr < 0.4):
                gen_taus_event.append([tau.pt, tau.eta, tau.phi, tau.mass])
            else:
                continue

        # ________________
        # Second tau
        # ________________
        
        if pairType == "ele_tau":
            idx = event.Electron[0].genPartIdx
            pdgId_target = 11
        elif pairType == 'mu_tau':
            idx = event.Muon[0].genPartIdx
            pdgId_target = 13     
                
    
        if is_true_tau: 
            pdgId_genpart = event.GenPart[idx].pdgId
            pdgId_genpart_mother = event.GenPart[event.GenPart[idx].genPartIdxMother].pdgId      
            if (abs(pdgId_genpart)==pdgId_target) and (abs(pdgId_genpart_mother)==15):
                tau_pt = event.GenPart[event.GenPart[idx].genPartIdxMother].pt
                tau_eta = event.GenPart[event.GenPart[idx].genPartIdxMother].eta
                tau_phi = event.GenPart[event.GenPart[idx].genPartIdxMother].phi
                tau_mass = event.GenPart[event.GenPart[idx].genPartIdxMother].mass
                gen_taus_event.append([tau_pt, tau_eta, tau_phi, tau_mass])
        else:
            if pairType == 'ele_tau':
                
                min_idx, min_dr = match_gen_particle(event.Electron[0], event.GenPart)

                if (min_idx >= 0) and (min_dr < 0.4):
                    gen_taus_event.append([event.Electron[0].pt, event.Electron[0].eta, event.Electron[0].phi, event.Electron[0].mass])
                else:
                    continue

            elif pairType == 'mu_tau':
                min_idx, min_dr = match_gen_particle(event.Muon[0], event.GenPart)

                if (min_idx >= 0) and (min_dr < 0.4):
                    gen_taus_event.append([event.Muon[0].pt, event.Muon[0].eta, event.Muon[0].phi, event.Muon[0].mass])
                else:
                    continue
                
        if len(gen_taus_event) in [0, 1]: continue 
        else: 
            gen_taus.append(gen_taus_event)
            indices.append(num)
            tau = event.Tau[0]
            
            if pairType == "ele_tau":
                ele = event.Electron[0]

                jet_per_event = np.zeros(len(event.Jet), dtype=np.bool_)
                for num, jet in enumerate(event.Jet): 
                    jet_eta, jet_phi = jet.eta, jet.phi
                    djt = delta_r(jet_eta, jet_phi, tau.eta, tau.phi)
                    djp = delta_r(jet_eta, jet_phi, ele.eta, ele.phi)
                    jet_per_event[num] = (djt>0.4) & (djp>0.4)
                
                selected_jets.append(jet_per_event)

            elif pairType == 'mu_tau': 
                mu = event.Muon[0]

                jet_per_event = np.zeros(len(event.Jet), dtype=np.bool_)
                for num, jet in enumerate(event.Jet): 
                    jet_eta, jet_phi = jet.eta, jet.phi
                    djt = delta_r(jet_eta, jet_phi, tau.eta, tau.phi)
                    djp = delta_r(jet_eta, jet_phi, mu.eta, mu.phi)
                    jet_per_event[num] = (djt>0.4) & (djp>0.4)
                
                selected_jets.append(jet_per_event)
         
    return indices, gen_taus, selected_jets


def compute_new_features(dataset, selected_jets, pairType):
    """
    Enhances the dataset by computing and adding new features for tau decay products, 
    tau properties, jet selection, and missing transverse energy (MET).

    Key Steps:
    - Sorts tau decay products (`TauProd`) by transverse momentum (`pt`).
    - Performs one-hot encoding for particle type (`pdgId`) in `TauProd` (electron, muon, photon, pion, kaon).
    - Computes and stores the logarithm of `pt` for `TauProd` and `Tau`.
    - Applies one-hot encoding to the `decayMode` feature in `Tau`.
    - Adjusts tau product indexing (`tauIdx_new`) for non tau-tau events.
    - Computes the ratio of tau product `pt` to tau `pt` (`pt_ratio`).
    - Selects the first 3 highest `pt` jets per event after filtering based on `selected_jets`.
    - Computes and stores the logarithm of `pt` for jets.
    - Computes the logarithm of MET (`PFMET_logpt` or `MET_logpt`), depending on the dataset.
    """
    # TAUPROD
    dataset["TauProd"]= dataset["TauProd"][ak.argsort(dataset["TauProd"]["pt"], ascending=False)] # Sort tauprod with respect to pt for the future truncation
    ## one-hot encoding pdgId
    dataset["TauProd"] = ak.with_field(dataset["TauProd"], np.log(dataset["TauProd"]["pt"]), "logpt")
    dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array([[1 if val == 11 else 0 for val in sublist] for sublist in abs(dataset["TauProd"]["pdgId"]).tolist()]), "is_ele")
    dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array([[1 if val == 13 else 0 for val in sublist] for sublist in abs(dataset["TauProd"]["pdgId"]).tolist()]), "is_mu")
    dataset["TauProd"]= ak.with_field(dataset["TauProd"], ak.Array([[1 if val == 22 else 0 for val in sublist] for sublist in abs(dataset["TauProd"]["pdgId"]).tolist()]), "is_pho")
    dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array([[1 if val == 211 else 0 for val in sublist] for sublist in abs(dataset["TauProd"]["pdgId"]).tolist()]), "is_pi")
    dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array([[1 if val == 130 else 0 for val in sublist] for sublist in abs(dataset["TauProd"]["pdgId"]).tolist()]), "is_k")
    ## one-hot encoding decayMode
    # [0, 1, 2, 10, 11]
    dataset["Tau"] = ak.with_field(dataset["Tau"], np.log(dataset["Tau"]["pt"]), "logpt")
    dataset["Tau"] = ak.with_field(dataset["Tau"], ak.Array([[1 if val == 0 else 0 for val in sublist] for sublist in dataset["Tau"]["decayMode"].tolist()]), "dM_0")
    dataset["Tau"] = ak.with_field(dataset["Tau"], ak.Array([[1 if val == 1 else 0 for val in sublist] for sublist in dataset["Tau"]["decayMode"].tolist()]), "dM_1")
    dataset["Tau"]= ak.with_field(dataset["Tau"], ak.Array([[1 if val == 2 else 0 for val in sublist] for sublist in dataset["Tau"]["decayMode"].tolist()]), "dM_2")
    dataset["Tau"] = ak.with_field(dataset["Tau"], ak.Array([[1 if val == 10 else 0 for val in sublist] for sublist in dataset["Tau"]["decayMode"].tolist()]), "dM_10")
    dataset["Tau"] = ak.with_field(dataset["Tau"], ak.Array([[1 if val == 11 else 0 for val in sublist] for sublist in dataset["Tau"]["decayMode"].tolist()]), "dM_11")
    # tauIdx_new will be filled of zeros because there is only one tau in the tau collection, "tauIdx_new" so all the tauproducts will refer to the first tau
    if pairType != "tau_tau":
        assert all(len(np.unique(i)) == 1 for i in dataset.TauProd.tauIdx), "Expected Tauprod information related to only one tau for ele_tau, mu_tau pairtype" 
        dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array([[0] * len(sublist) for sublist in dataset.TauProd["tauIdx"]]), "tauIdx_new")
        dataset["TauProd"] = ak.with_field(dataset["TauProd"], ak.Array(dataset.TauProd.pt[i]/dataset.Tau.pt[i] for i in range(0, len(dataset))), "pt_ratio")   
        if pairType == 'ele_tau':
            dataset['Electron'] = ak.with_field(dataset["Electron"], np.log(dataset["Electron"]["pt"]), "logpt")
        elif pairType == 'mu_tau':
            dataset['Muon'] = ak.with_field(dataset["Muon"], np.log(dataset["Muon"]["pt"]), "logpt")
    else: 
        pt_ratio_tprod = pt_ratio_tauprod(dataset)
        dataset["TauProd"]= ak.with_field(dataset["TauProd"], ak.Array(pt_ratio_tprod), "pt_ratio")
    # jets
    dataset["Jet"] = dataset["Jet"][selected_jets]
    dataset["Jet"] = dataset["Jet"][ak.argsort(dataset["Jet"]["pt"], ascending=False)]
    dataset["Jet"] = dataset["Jet"][:, :3] # Select the first 3 jets for each event
    dataset["Jet"] = ak.with_field(dataset["Jet"], np.log(dataset["Jet"]["pt"]), "logpt")
    assert list(ak.num(dataset["Jet"]["pt"])).count(4) == 0
    if "PFMET_pt" in dataset.fields:
        dataset["PFMET_logpt"] = ak.Array(np.log(dataset.PFMET_pt))
    else:
        dataset["MET_logpt"] = ak.Array(np.log(dataset.MET_pt))
        
    return dataset




def compute_invariant_masses(dataset, gen_taus, pairType):
    """
    Returns:
    - `inv_mass_reco`: Array of reconstructed invariant masses.
    - `inv_mass_tar`: Array of true (MC) invariant masses.
    """
    flattened_sublists = [sum(sublist, []) for sublist in gen_taus] # Flatten each sublist to pass the 8 values to the invariant mass function
    result = [[sublist[0], sublist[1], sublist[2], sublist[3], sublist[4], sublist[5], sublist[6], sublist[7]] for sublist in flattened_sublists]
    inv_mass_tar = [invariant_mass(*particle_properties) for particle_properties in result] # Compute invariant mass for each pair of particles

    pt1 = np.array(dataset.Tau.pt[:, 0])
    eta1 = np.array(dataset.Tau.eta[:, 0])
    phi1 = np.array(dataset.Tau.phi[:, 0])
    mass1 = np.array(dataset.Tau.mass[:, 0])
    
    if pairType == "ele_tau":
        pt2 = np.array(dataset.Electron.pt[:, 0])
        eta2 = np.array(dataset.Electron.eta[:, 0])
        phi2 = np.array(dataset.Electron.phi[:, 0])
        mass2 = np.array(dataset.Electron.mass[:, 0])
    elif pairType == "mu_tau":
        pt2 = np.array(dataset.Muon.pt[:, 0])
        eta2 = np.array(dataset.Muon.eta[:, 0])
        phi2 = np.array(dataset.Muon.phi[:, 0])
        mass2 = np.array(dataset.Muon.mass[:, 0])
    else:
        pt2 = np.array(dataset.Tau.pt[:, 1])
        eta2 = np.array(dataset.Tau.eta[:, 1])
        phi2 = np.array(dataset.Tau.phi[:, 1])
        mass2 = np.array(dataset.Tau.mass[:, 1])

    inv_mass_reco = invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2)

    return inv_mass_reco, np.array(inv_mass_tar)





def mass_plot(inv_mass_reco, inv_mass_tar, out_name):
    """
    Generates and saves a histogram comparing the invariant mass distributions of 
    reconstructed (RECO) and Monte Carlo truth (MC) tau-tau pairs.
    """
    plt.figure(figsize=(6, 4))
    plt.style.use(hep.style.CMS)

    r_max = 150
    plt.hist(inv_mass_tar, bins=200, range=(0, r_max), histtype='step', linewidth=2, label=r'$m_{\tau\tau}^{MC}$ - $\mu = $' + str(np.round(np.mean(inv_mass_tar), 3)) + ' - ' + '$\sigma =$' + str(np.round(np.std(inv_mass_tar), 3)))
    plt.hist(inv_mass_reco, bins=200, range=(0, r_max), histtype='step', linewidth=2, color = "orange", label=r'$m_{\tau\tau}^{RECO}$ - $\mu = $' + str(np.round(np.mean(inv_mass_reco), 3)) + ' - ' + '$\sigma =$' + str(np.round(np.std(inv_mass_reco), 3)))
    plt.xlim(0, r_max)
    plt.xlabel('Invariant Mass', fontsize = 12)
    plt.ylabel('Events', fontsize = 12)
    plt.title('MC & Reco Invariant Mass Distributions', fontsize = 16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper left', fontsize=10)
    plt.savefig(out_name)

    

def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    """Function to compute the invariant mass"""
    
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
    
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)
    
    return np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)



def _pad(a, maxlen, value=0, dtype='float32'):
    """Function to pad matrices"""
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x
    

@njit
def delta_r(eta1, phi1, eta2, phi2):
    """Function to compute deltaR"""
    dphi = np.abs(phi1 - phi2)
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def delta_r_numpy(eta1, phi1, eta2, phi2):
    dphi = phi1 - phi2
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi  # corretta gestione del range [-pi, pi]
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)

def compute_delta_r(taus_array):
    # Assumendo: taus_array shape = (num_events, 2, num_features)
    eta1 = taus_array[:, 0, 1]
    phi1 = taus_array[:, 0, 2]
    eta2 = taus_array[:, 1, 1]
    phi2 = taus_array[:, 1, 2]

    return delta_r_numpy(eta1, phi1, eta2, phi2)

@njit
def pt_ratio_tauprod(dataset):
    """Computes the pt ratio of tau decay products relative to their parent taus.
    Returns a list of pt ratios for each event."""
    pt_ratio = []
    
    for i in range(0, len(dataset)):
        pt_ratio_event = []
        tauprod = dataset.TauProd[i]
        tau = dataset.Tau[i]
        for tprod in tauprod: 
            tprod_pt = tprod.pt
            tprod_idx = tprod.tauIdx
            tau_pt = tau[tprod_idx].pt
            pt_ratio_event.append(tprod_pt/tau_pt)
        pt_ratio.append(pt_ratio_event)
    return pt_ratio



def InteractionMask(encoder_input, num_pair_features = 4, eps=1e-8, mask_self_interaction=True):
    """
    Computes pairwise interaction features for tau candidates in an event. 
    It calculates four key features: ΔR (spatial separation), kT (transverse momentum-based distance), 
    z (momentum fraction), and m² (invariant mass squared). 
    It applies a mask to remove self-interactions if specified. 
    Returns a tensor containing the interaction features.
    """
    input_seq_len = encoder_input.shape[1]  # (num_events, num_taus, num_features) -> num_taus (6)
    
    logpt = encoder_input[..., 0]
    eta = encoder_input[..., 1]
    phi = encoder_input[..., 2]
    mass = encoder_input[..., 3]
    
    # delta feature
    eta_diff = eta[:, :, np.newaxis] - eta[:, np.newaxis, :]
    phi_diff = delta_phi(phi[:, :, np.newaxis], phi[:, np.newaxis, :])
    delt = np.sqrt(eta_diff**2 + phi_diff**2 + eps)
    lndelta = np.log(delt)

    # kT feature
    pt_a = np.exp(logpt[:, :, np.newaxis])
    pt_b = np.exp(logpt[:, np.newaxis, :])
    min_pt = np.minimum(pt_a, pt_b)
    kt = min_pt * delt
    lnkt = np.log(kt + eps)

    # z feature
    zeta = min_pt / (pt_a + pt_b + eps)
    lnz = np.log(zeta + eps)

    # m2 feature
    m2 = m_2_batch(np.exp(logpt), eta, phi, mass)
    lnm2 = np.log(m2 + eps)

    result = np.stack([lndelta, lnkt, lnz, lnm2], axis=1)

    # Apply a mask to remove self-interactions 
    if mask_self_interaction:
        diag_mask = np.eye(input_seq_len, dtype=bool)
        result[:, :, diag_mask] = 0

    return result


def delta_phi(phi1, phi2):
    """Calculates the difference between phi angles, corrected for the range [-pi, pi]"""
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def m_2_batch(pt, eta, phi, mass):
    """Computes the invariant mass m2 in a vectorized manner for all particles."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    E_sum = E[:, :, np.newaxis] + E[:, np.newaxis, :]
    p_sum = np.stack([px, py, pz], axis=-1)
    p_sum = p_sum[:, :, np.newaxis, :] + p_sum[:, np.newaxis, :, :]

    return E_sum**2 - np.sum(p_sum**2, axis=-1)


def SVFit_input(dataset, tau_features, MET_features, pairType):
    """
    Prepares input features for SVFit algorithm.
    
    Depending on the pair type ("tau_tau", "ele_tau", or "mu_tau"), the function extracts and formats 
    tau and MET features from the dataset. 

    The function returns:
    - `svfit_data`: A Pandas DataFrame containing the processed features.
    - `ak_array`: An Awkward Array representation of the same data.
    """
    if pairType == "tau_tau":
        tau = np.stack([ak.to_numpy(dataset.Tau[n]) for n in tau_features], axis=1)
        tau = np.transpose(tau, (0, 2, 1))
        tau = np.reshape(tau, (tau.shape[0], tau.shape[1]*tau.shape[2]))
        MET = np.stack([ak.to_numpy(dataset[n]) for n in MET_features], axis=1)
    
        result = np.concatenate((tau, MET), axis=1)
    
        columns = ["tau1_pt", "tau1_eta", "tau1_phi", "tau1_mass", "DM1",            #tau
                   "tau2_pt", "tau2_eta", "tau2_phi", "tau2_mass", "DM2",            #tau
                   "met_pT" , "met_phi", "met_covXX", "met_covXY", "met_covYY"]      #MET
    else: 
        tau = np.stack([ak.to_numpy(dataset.Tau[n]) for n in tau_features], axis=1)
        tau = np.transpose(tau, (0, 2, 1))
        tau = np.reshape(tau, (tau.shape[0], tau.shape[1]*tau.shape[2]))
        p_features = ['pt', 'eta', 'phi', 'mass']
        if pairType == "ele_tau":
            collection = "Electron"
        elif pairType == "mu_tau":
            collection = "Muon"
        p = np.stack([ak.to_numpy(dataset[collection][n]) for n in p_features], axis=1)
        p = np.transpose(p, (0, 2, 1))
        p = np.reshape(p, (p.shape[0], p.shape[1]*p.shape[2]))
        MET = np.stack([ak.to_numpy(dataset[n]) for n in MET_features], axis=1)
    
        result = np.concatenate((p, tau, MET), axis=1)
    
        columns = ["tau1_pt", "tau1_eta", "tau1_phi", "tau1_mass",                   #ele/mu
                   "tau2_pt", "tau2_eta", "tau2_phi", "tau2_mass", "DM2",            #tau
                   "met_pT" , "met_phi", "met_covXX", "met_covXY", "met_covYY"]      #MET

    svfit_data = pd.DataFrame(result, columns = columns)
    ak_array = ak.Array(svfit_data.to_dict(orient='list'))

    return svfit_data, ak_array



def process_feature(data, collection, feature):
    """
    Extracts or assigns default values to a specified feature from a given particle collection.
    It serves for ele_tau/mu_tau training (Electron and Muon collections don't have all the features that Tau has)

    - For certain predefined features (e.g., 'ptCorrPNet', 'rawPNetVSjet'), it returns arrays of zeros.
    - Specific decay mode and tracking-related features are assigned zeros.
    - 'leadTkPtOverTauPt' is assigned ones.
    - All other features are extracted as-is.

    Returns a NumPy array containing the processed feature values.
    """
        
    if feature in ['ptCorrPNet', 'rawPNetVSjet', 'rawDeepTau2018v2p5VSjet']:
        return np.zeros_like(ak.to_numpy(data[collection]['logpt']))
    elif feature == 'charge':
        return ak.to_numpy(data[collection]['charge'])
    elif feature in ['dM_0', 'dM_1', 'dM_2', 'dM_10', 'dM_11', 'leadTkDeltaEta', 'leadTkDeltaPhi']:
        return np.zeros_like(ak.to_numpy(data[collection]['logpt']))
    elif feature == 'leadTkPtOverTauPt':
        return np.ones_like(ak.to_numpy(data[collection]['logpt']))
    else:
        return ak.to_numpy(data[collection][feature])
    


@njit
def match_gen_particle(tau, genparts, delta_r_threshold=0.4):
    """
    Returns the index and deltaR of the closest GenPart to a tau that is either:
    - a b quark from a top quark, or
    - a daughter of a W boson.
    If no match is found within delta_r_threshold, returns (-1, 999.0).
    """
    min_dr = 999.
    min_idx = -1
    for i, gen in enumerate(genparts):
        mother_idx = gen.genPartIdxMother
        if mother_idx < 0:
            continue
        mother = genparts[mother_idx]
        is_b_from_top = abs(gen.pdgId) == 5 and abs(mother.pdgId) == 6
        is_child_of_W = abs(mother.pdgId) == 24
        if is_b_from_top or is_child_of_W:
            dr = delta_r(tau.eta, tau.phi, gen.eta, gen.phi)
            if dr < min_dr:
                min_dr = dr
                min_idx = i
    if (min_idx >= 0) and (min_dr < delta_r_threshold):
        return min_idx, min_dr
    else:
        return -1, 999.
