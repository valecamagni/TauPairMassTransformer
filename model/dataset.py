import torch
from torch.utils.data import Dataset

class DiTauDataset(Dataset):
    """
    Custom implementation of a Dataset in PyTorch.
    
    Args: 
        - ds_inputs : a dictionary where each key corresponds to a specific type of data.
                      The associated value is a torch.Tensor.

    Returns:
        a dictionary with the following keys:
        - tau
        - tauprod (empty tensor if not available)
        - jets
        - met
        - inv_mass_reco
        - inv_mass_tar
        - tar
        - dataset (depending on input sample)
        - pairType ('ele_tau', 'mu_tau', 'tau_tau)
        - padding_masks: a concatenated tensor of padding masks indicating which values are real and which are padded for each feature (tau, tauprod, jets, MET, etc.).
    """

    def __init__(self, ds_input):
        super().__init__()
        self.ds_input = ds_input

    def __len__(self):
        return next(iter(self.ds_input.values())).shape[0]

    def __getitem__(self, idx):
        tar = self.ds_input['tar'][idx]
        inv_mass_tar = self.ds_input['inv_mass_tar'][idx]
        dataset = self.ds_input['dataset'][idx]
        pairtype = self.ds_input['pairType'][idx]

        tau_matrix = self.ds_input['tau'][idx]
        jets_matrix = self.ds_input['jets'][idx]
        met = self.ds_input['met'][idx]
        inv_mass_reco = self.ds_input['inv_mass_reco'][idx]

        # Compute the mandatory zero masks
        zero_mask_taus = (tau_matrix.sum(dim=-1) == 0).to(torch.int)
        zero_mask_jets = (jets_matrix.sum(dim=-1) == 0).to(torch.int)
        zero_mask_met = (met.sum(dim=-1) == 0).to(torch.int)
        zero_mask_mass = (inv_mass_reco.sum(dim=-1) == 0).to(torch.int)

        # Optional: TauProd
        tauprod_matrix = self.ds_input.get('tauprod', None)
        zero_mask_tauprod = None
        if tauprod_matrix is not None:
            tauprod_matrix = tauprod_matrix[idx]
            zero_mask_tauprod = (tauprod_matrix.sum(dim=-1) == 0).to(torch.int)

        # Combine padding masks
        if zero_mask_tauprod is not None:
            padding_masks = torch.concat([zero_mask_taus, zero_mask_tauprod, zero_mask_jets, zero_mask_met, zero_mask_mass], dim=0)
        else:
            padding_masks = torch.concat([zero_mask_taus, zero_mask_jets, zero_mask_met, zero_mask_mass], dim=0)

        # Return the data, ensuring tauprod_matrix can be None
        return {
            "tau": tau_matrix,
            "tauprod": tauprod_matrix if tauprod_matrix is not None else torch.tensor([]),  # Empty tensor if None
            "jets": jets_matrix,
            "met": met,
            "inv_mass_reco": inv_mass_reco,
            "inv_mass_tar": inv_mass_tar,
            "tar": tar.squeeze(),
            "dataset": dataset,
            "pairType": pairtype,
            "padding_masks": padding_masks.float(),
            #"padding_masks": padding_masks.bool()
        }
