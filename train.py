import torch
from model.DnCNN import DnCNN
import os
import json
from processing.denoise_loader import get_dn_data_loaders
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import *
from time import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from processing.data_processing import extract_test_patches


def train_step(train_loader,
               model,
               loss_fn,
               optimizer,
               device,
               summary,
               epoch):
    """
    Training step.
    """
    # Initialize the training loss
    train_loss = 0

    # Initialize psnr
    train_psnr = 0

    # Set up the training mode
    model.train()

    # Loop through the data loader
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        # Get the slices, labels and weights, then send them to the desired device
        orig = batch['orig'].to(device).float()
        noisy = batch['noisy'].to(device)

        # Zero the gradients before every batch
        optimizer.zero_grad()

        # Forward pass
        pred = model(noisy)

        # Compute the loss
        loss = loss_fn(pred,
                       orig)

        # Update the running loss:
        train_loss += loss.item()
        train_psnr += psnr(orig,
                           pred)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Compute the mean loss / epoch
    train_loss /= len(train_loader)

    # Compute train_psnr per epoch
    train_psnr /= len(train_loader)

    # Write the loss value
    summary_writer.add_scalar(f'Loss/Train', train_loss, epoch)

    # Write the PSNR
    summary_writer.add_scalar(f'PSNR/Train', train_psnr, epoch)

    # Print results
    print(f"Train Mode. Epoch: {epoch}, loss: {train_loss}, PSNR: {train_psnr}")


def eval_step(val_loader,
              model,
              loss_fn,
              summary,
              epoch):
    """
    Evaluation step
    """
    # Set up a loss list
    eval_loss = 0

    # Eval psnr
    eval_psnr = 0.0

    # Set up the model to eval mode
    model.eval()

    # Turn on the inference mode
    with torch.inference_mode():
        # Loop through the data loader
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            # Get the slices, labels and weights, then send them to the desired device
            orig = batch['orig'].to(device).float()
            noisy = batch['noisy'].to(device).float()

            # Forward pass
            pred = model(noisy)

            # Compute the loss
            loss = loss_fn(pred,
                           orig)

            # Add the running loss
            eval_loss += loss.item()

            # Compute eval psnr
            eval_psnr += psnr(pred,
                              orig).item()

    # Compute the mean loss / epoch
    eval_loss /= len(val_loader)

    # Compute psnr per epoch
    eval_psnr /= len(val_loader)

    # Write the loss value
    summary_writer.add_scalar(f'Loss/Eval', eval_loss, epoch)

    # Write the PSNR
    summary_writer.add_scalar(f'PSNR/Eval', eval_psnr, epoch)

    print(f"Eval Mode. Epoch: {epoch}, loss: {eval_loss}, PSNR: {eval_psnr}")

    # Return
    return eval_loss, eval_psnr


def save_checkpoint(path: str,
                    epoch: int = None,
                    score: float = None,
                    model: torch.nn.Module = None,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler=None,
                    is_best: bool = False,
                    is_latest: bool = False,
                    ):
    """
    Stores/loads checkpoints (model states).
    """
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'score': score,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    if is_best:
        path += '/best.pkl'
        if os.path.exists(path):
            os.remove(path)
    elif is_latest:
        path += '/latest.pkl'
    else:
        path += f'/epoch_{epoch}.pkl'

    torch.save(checkpoint, path)


if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)

    # Read the current configuration
    cfg = json.load(open(current_dir + '/utils/config.json', 'r'))

    # Setup the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the model:
    model = DnCNN()

    # Get number of epochs
    epochs = 50

    # Create the data loaders
    train_loader, val_loader = get_dn_data_loaders(cfg)

    # Loss Function
    loss_fn = nn.L1Loss()

    # Define an optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0002,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0)

    # Train
    model = model.to(device)

    # Initialize bets_psnr
    best_psnr = 0

    EXPERIMENT = datetime.now().strftime("%m-%d-%y_%H-%M")

    # Checkpoint path
    checkpoint_path = cfg['base_path'] + '/experiments/checkpoints/' + EXPERIMENT

    # Start training
    print('==== Started training ====')
    start_time = time()

    # Initialize psnr
    epoch_eval_psnr = 0

    # Writer
    summary_writer = SummaryWriter(cfg['base_path'] + '/experiments/summary/' + EXPERIMENT)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # Perform the training step
        train_step(train_loader=train_loader,
                   model=model,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   device=device,
                   summary=summary_writer,
                   epoch=epoch)

        # Perform the evaluation step
        eval_loss, eval_psnr = eval_step(val_loader=val_loader,
                                         model=model,
                                         loss_fn=loss_fn,
                                         summary=summary_writer,
                                         epoch=epoch)

        # Compare the current dsc with the best dsc
        if eval_psnr > best_psnr:
            best_dsc = eval_psnr
            print(
                f"New best checkpoint at epoch {epoch + 1} | PSNR: {eval_psnr} dB\nSaving new best model."
            )
            save_checkpoint(path=checkpoint_path,
                            epoch=epoch + 1,
                            score=eval_psnr,
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            is_best=True)

    # Save the last state of the network
    save_checkpoint(path=checkpoint_path,
                    epoch=epochs-1,
                    score=eval_psnr,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    is_best=False)

    # Stop training
    end_time = time()
    print(f"==== Stopped training. Total training time: {end_time - start_time:.3f} seconds ===="
                f"\n===========================================")


def test_psnr(orig_path,
              noisy_path):
    orig_images_path = [os.path.join(orig_path, s) for s in os.listdir(orig_path)
                        if s.endswith('.jpg') or s.endswith('.jpeg')]
    noisy_images_path = [os.path.join(noisy_path, s) for s in os.listdir(noisy_path)
                         if s.endswith('.jpg') or s.endswith('.jpeg')]

    for orig_image_path, noisy_image_path in zip(orig_images_path, noisy_images_path):
        test_orig_image, test_noisy_image, initial_shape = extract_test_patches(orig_image_path,
                                                                                noisy_image_path)

        for orig_patch, noisy_patch in zip(test_orig_image, test_noisy_image):




