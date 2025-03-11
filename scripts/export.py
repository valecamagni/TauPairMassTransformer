import torch
import torch.onnx
from config import get_config, get_weights_file_path
from utils_train import get_TPMTmodel
from utils_datasets import get_test_ds 

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"
config = get_config(base_folder)
use_tauprod = config['use_tauprod']
test_pairtype = config['test_pairType']
model_folder = config['model_folder']

# load model and dataset (for input sizes)
te, dataset = get_test_ds(base_folder)
model = get_TPMTmodel(config)

# load weights
epoch = str(config["weight_epoch"])
model_filename = get_weights_file_path(config, epoch)
print(f"Loading weights from: {model_filename}")
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

model.eval()

# Dummy inputs with the same size as real data
batch_size = 1
taus = torch.randn(batch_size, 2, 18)  # (batch, num_taus, features)
jets = torch.randn(batch_size, 3, 4)  # (batch, num_jets, features)
met = torch.randn(batch_size, 1, 8)  # (batch, 1, features)
mass = torch.randn(batch_size, 1, 1)  # (batch, 1, 1)
tauprods = torch.randn(batch_size, 10, 10) if use_tauprod else torch.zeros(batch_size, 1, 1)  # (batch, num_tauprods, features)
padding_masks = torch.randint(0, 2, (batch_size, 17)).float()  # Maschera binaria

input_tensors = (taus, jets, met, mass, tauprods, padding_masks)

torch.onnx.export(
    model, 
    input_tensors, 
    #f"{model_folder}/tpmt_model_{test_pairtype}.onnx",  
    f"{model_folder}/tpmt_model_allinclusive.onnx",  
    export_params=True,  
    opset_version=11,  
    do_constant_folding=True,  
    input_names=['taus', 'jets', 'met', 'mass', 'tauprods', 'padding_masks'],  
    output_names=['output'],  
    dynamic_axes={
        'taus': {0: 'batch_size'}, 
        'jets': {0: 'batch_size'}, 
        'met': {0: 'batch_size'}, 
        'mass': {0: 'batch_size'}, 
        'tauprods': {0: 'batch_size'}, 
        'padding_masks': {0: 'batch_size'},  
        'output': {0: 'batch_size'}
    }
)

print("Model exported")


#import onnx
#import onnxruntime
#
## Carica il modello ONNX
#onnx_model = onnx.load("tpmt_model.onnx")
#onnx.checker.check_model(onnx_model)  # Controlla che il modello sia valido
#
## Test inferenza con ONNX Runtime
#ort_session = onnxruntime.InferenceSession("tpmt_model.onnx")
#
#print(padding_masks.numpy())
## Esegui inferenza
#outputs = ort_session.run(None, {
#    "taus": taus.numpy(),
#    "jets": jets.numpy(),
#    "met": met.numpy(),
#    "mass": mass.numpy(),
#    "tauprods": tauprods.numpy() if use_tauprod else [],
#    "padding_masks": padding_masks.numpy(),
#})
#
#print(outputs)