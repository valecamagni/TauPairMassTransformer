import onnxruntime as ort
import numpy as np
from utils_datasets import get_test_ds 

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
te, dataset = get_test_ds(base_folder)

session = ort.InferenceSession("tpmt_model.onnx", providers=["CPUExecutionProvider"])

batch = next(iter(te))
taus = batch['tau'].float()
jets = batch['jets'].float()
met = batch['met'].float()
mass = batch['inv_mass_reco'].float()
tauprods = batch['tauprod'].float()
padding_masks = batch['padding_masks']

# dictionary for ONNX Runtime
input_dict = {
    "taus": taus.numpy(),
    "jets": jets.numpy(),
    "met": met.numpy(),
    "mass": mass.numpy(),
    "tauprods": tauprods.numpy(),
    "padding_masks": padding_masks.numpy()
}

# inference
outputs = session.run(None, input_dict)
print("Output ONNX:", np.exp(outputs[0]))


