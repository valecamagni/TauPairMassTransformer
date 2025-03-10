import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import get_config, get_weights_file_path 
from utils_train import NFvalidation, get_NFmodel, plot_losses, compute_mmd, CustomLoss
from utils_datasets import get_ds
import argparse

parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_folder = "/gwpool/users/camagni/Di-tau/pre_processing/"

config = get_config(base_folder)
use_tauprod = config['use_tauprod']

# Get datasets
tr, va = get_ds(config)
print('DATA PROCESSED')

# Initialize model, optimizer, and scheduler
model = get_NFmodel(config).to(device)
print(f"Number of model's parameters: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8, min_lr=1e-6, verbose=True)

criterion = CustomLoss()

train_loss_per_epoch, val_loss_per_epoch = [], []
train_nll_loss_per_epoch, val_nll_loss_per_epoch = [], []
train_mmd_loss_per_epoch, val_mmd_loss_per_epoch = [], []
#train_tau_loss_per_epoch, val_tau_loss_per_epoch = [], []
#train_mass_loss_per_epoch, val_mass_loss_per_epoch = [], []

global_step, best_val_loss, early_stopping_counter = 0, float('inf'), 0

# Training loop
print('START TRAINING LOOP')

for epoch in range(config['num_epochs']):
    model.train()
    running_loss = 0.0 
    running_nll_loss = 0.0
    running_mmd_loss = 0.0
    #running_tau_loss = 0.0
    #running_mass_loss = 0.0

    batch_iterator = tqdm(tr, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:
        # Move tensors to device
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

        nll_loss = - log_prob.mean()
        mmd_loss = compute_mmd(generated_samples.squeeze(0), target)
        _, mass_loss = criterion(generated_samples.squeeze(0), target, taus, mass_target)
        loss = nll_loss + mmd_loss #+ mass_loss

        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        # Backpropagation and optimization
        loss.backward()
        if any(torch.isnan(param.grad).any() for name, param in model.named_parameters()):
            print("Gradient contains NaN, stopping...")
            break
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        running_loss += loss.item()
        running_nll_loss += nll_loss.item()
        running_mmd_loss += mmd_loss.item()
        #running_mass_loss += mass_loss.item()
        #running_tau_loss += tau_loss.item()

    # Calculate mean epoch losses
    epoch_loss = running_loss / len(tr)
    epoch_nll_loss = running_nll_loss / len(tr)
    epoch_mmd_loss = running_mmd_loss / len(tr)
    #epoch_mass_loss = running_mass_loss / len(tr)
    #epoch_tau_loss = running_tau_loss / len(tr)

    train_loss_per_epoch.append(epoch_loss)
    train_nll_loss_per_epoch.append(epoch_nll_loss)
    train_mmd_loss_per_epoch.append(epoch_mmd_loss)
    #train_mass_loss_per_epoch.append(epoch_mass_loss)
    #train_tau_loss_per_epoch.append(epoch_tau_loss)

    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {epoch_loss:.5f}")
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], NLL Loss: {epoch_nll_loss:.5f}")
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], MMD Loss: {epoch_mmd_loss:.5f}")
    #print(f"Epoch [{epoch+1}/{config['num_epochs']}], Mass Loss: {epoch_mass_loss:.5f}")
    #print(f"Epoch [{epoch+1}/{config['num_epochs']}], Tau Loss: {epoch_tau_loss:.5f}")

    epoch_val_loss, epoch_val_nll_loss, epoch_val_mmd_loss, epoch_val_mass_loss = NFvalidation(model, va, device, use_tauprod)
    val_loss_per_epoch.append(epoch_val_loss)
    val_nll_loss_per_epoch.append(epoch_val_nll_loss)
    val_mmd_loss_per_epoch.append(epoch_val_mmd_loss)
    #val_mass_loss_per_epoch.append(epoch_val_mass_loss)
    #val_tau_loss_per_epoch.append(epoch_val_tau_loss)

    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation Loss: {epoch_val_loss:.5f}")
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation NLL Loss: {epoch_val_nll_loss:.5f}")
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation MMD Loss: {epoch_val_mmd_loss:.5f}")
    #print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation Mass Loss: {epoch_val_mass_loss:.5f}")
    #print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation Tau Loss: {epoch_val_tau_loss:.5f}")

    # Learning rate adjustment
    scheduler.step(epoch_val_loss)

    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping: Validation loss did not improve for {config['early_stopping_patience']} epochs.")
            break

    # Save model checkpoint
    os.makedirs(os.getcwd()+ '/' + config['model_folder'] + '/weights', exist_ok=True)
    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)


# Plot loss curves
plot_losses(train_loss_per_epoch, val_loss_per_epoch, 'Epoch', 'Loss', 
            'Training and Validation Loss Over Epochs', config['model_folder'] + '/losses.png')
plot_losses(train_nll_loss_per_epoch, val_nll_loss_per_epoch, 'Epoch', 'NLL Loss', 
            'Training and Validation NLL Loss Over Epochs', config['model_folder'] + '/nll_losses.png')
plot_losses(train_mmd_loss_per_epoch, val_mmd_loss_per_epoch, 'Epoch', 'MMD Loss', 
            'Training and Validation MMD Loss Over Epochs', config['model_folder'] + '/mmd_losses.png')
#plot_losses(train_mass_loss_per_epoch, val_mass_loss_per_epoch, 'Epoch', 'Mass Loss', 
#            'Training and Validation Mass Loss Over Epochs', config['model_folder'] + '/mass_losses.png')
#plot_losses(train_tau_loss_per_epoch, val_tau_loss_per_epoch, 'Epoch', 'Tau Loss', 
#            'Training and Validation Tau Loss Over Epochs', config['model_folder'] + '/tau_losses.png')
# Log-scale plots
#plot_losses(train_loss_per_epoch, val_loss_per_epoch, 'Epoch', 'LogLoss', 
#            'Training and Validation Loss (Log Scale)', config['model_folder'] + '/log_losses.png', log_scale=True)

# Save losses to CSV
pd.DataFrame({'train_loss': train_loss_per_epoch, 'val_loss': val_loss_per_epoch}).to_csv(config['model_folder'] + '/results_losses.csv')








