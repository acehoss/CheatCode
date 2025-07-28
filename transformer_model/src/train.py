import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from cheat_code_model import CheatCodeModel
from dataset import CheatCodeDataset
import argparse
import wandb
import os
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from contextlib import nullcontext
import math
from sklearn.metrics import accuracy_score
import random
from torch.amp import autocast, GradScaler  # Updated import
import sys

# Set seeds at the beginning of your script
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)  # Use any integer seed you like

def compute_day_correlation(train_sin, train_cos, val_sin, val_cos):
    train_angle = math.atan2(train_sin, train_cos)
    val_angle = math.atan2(val_sin, val_cos)
    
    # Convert angles to days of year (0-365)
    train_day = (train_angle + math.pi) / (2 * math.pi) * 365
    val_day = (val_angle + math.pi) / (2 * math.pi) * 365
    
    # Compute circular correlation
    diff = abs(train_day - val_day)
    correlation = 1 - min(diff, 365 - diff) / 182.5  # 182.5 is half of 365
    return correlation

def compute_accuracy(outputs, targets):
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(outputs)
    
    # Convert probabilities to binary predictions using a threshold of 0.5
    preds = (probs >= 0.7).float()
    
    # Calculate accuracy over all predictions
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy.item()

def train(model, train_dataloader, val_dataloader, num_epochs, device, checkpoint_dir, learning_rate, batch_size,
          nhead, d_model, num_encoder_layers, weight_decay, save_steps=1000, run_name=None,
          max_grad_norm=None, steps_per_epoch=None, pos_weight=None, start_epoch=0, global_step=0,
          optimizer=None, scheduler=None, detect_anomaly=False, warmup_steps=0, disable_autocast=False):
    model.to(device)
    
    scaler = None
    if device.type == 'cuda' and not disable_autocast:
        scaler = GradScaler(
            # init_scale=1e14,
            # growth_interval=10,
            # growth_factor=1.0001,
            # backoff_factor=0.9999
        )
    
    # Ensure optimizer and scheduler are provided
    assert optimizer is not None, "Optimizer must be provided"
    assert scheduler is not None, "Scheduler must be provided"
    
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        train_criterion = nn.BCEWithLogitsLoss()
    
    val_criterion = nn.BCEWithLogitsLoss()
    
    # Log hyperparameters
    wandb.config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "d_model_input": d_model,
        "nhead_input": nhead,
        "num_encoder_layers": num_encoder_layers,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps,
        "pos_weight": pos_weight,
        "shuffle_seed": seed,
        "command_line": " ".join(sys.argv)  # Add this line to include the complete command line string
    })
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    val_iter = iter(val_dataloader)
    train_iter = iter(train_dataloader)
    
    # Create overall progress bar
    overall_progress = tqdm(total=num_epochs, desc="Overall Progress", position=0, initial=start_epoch)
    
    start_time = time.time()
    
    # Access the underlying model attributes
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # Define the model configuration dictionary using base_model
    model_config = {
        'num_tickers': base_model.num_tickers,
        'd_model_input': base_model.d_model_input,
        'nhead_input': base_model.nhead_input,
        'd_model_post': base_model.d_model_post,
        'nhead_post': base_model.nhead_post,
        'num_encoder_layers': base_model.ticker_subnets[0].transformer_encoder.num_layers,
        'num_post_layers': base_model.post_transformer.num_layers,
        'seq_len': base_model.ticker_subnets[0].seq_len,
        'dropout': args.dropout,
        'dim_feedforward_input': base_model.dim_feedforward_input,  # Added this line
        'dim_feedforward_post': base_model.dim_feedforward_post,    # Added this line
    }
    
    # Save the initial checkpoint with model_config before training starts
    initial_checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_initial_checkpoint.pt")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': 0,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),  # Added this line
        'scheduler_state_dict': scheduler.state_dict(),  # Added this line
        'model_config': model_config,
        'run_name': run_name,
        'shuffle_seed': seed,
    }, initial_checkpoint_path)
    print(f"Initial checkpoint saved: {initial_checkpoint_path}")
    
    for epoch in range(start_epoch, num_epochs):            
        model.train()
        total_loss = 0
        total_val_loss = 0
        total_accuracy = 0
        total_val_accuracy = 0
        
        epoch_steps = steps_per_epoch if steps_per_epoch is not None else len(train_dataloader)
        
        # Create epoch progress bar
        epoch_progress = tqdm(total=epoch_steps, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        
        # If resuming, set the initial value of the epoch progress bar
        if epoch == start_epoch and global_step > 0:
            initial_step = global_step % epoch_steps
            epoch_progress.update(initial_step)
            
            # # Advance the train_iter to the correct position
            # for _ in range(initial_step):
            #     try:
            #         next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_dataloader)
            #         next(train_iter)
        else:
            initial_step = 0

        val_loss = 0
        val_accuracy = 0
        for i in range(initial_step, epoch_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            # Training step
            minute_data = [tensor.to(device) for tensor in batch['minute_data']]
            labels = batch['labels'].to(device)
            
            outputs = None
            loss = None
            
            # Choose the appropriate context manager
            ctx = torch.autograd.detect_anomaly() if detect_anomaly else nullcontext()
            ac_ctx = autocast(device_type='cuda', dtype=torch.bfloat16) if not disable_autocast and device.type == 'cuda' else nullcontext()
            
            with ctx:
                optimizer.zero_grad()
                with ac_ctx:
                    outputs = model(minute_data)
                    loss = train_criterion(outputs, labels)
                
                # Backward pass and optimization
                if scaler is not None:
                    scaler.scale(loss).backward()
                    
                    # Unscale gradients and perform gradient clipping if max_grad_norm is provided
                    scaler.unscale_(optimizer)
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
                    
                    # Move gradient norm computation here
                    total_norm = 0
                    param_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_norms.append((name, param_norm.item()))
                    total_norm = total_norm ** 0.5
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Perform gradient clipping if max_grad_norm is provided
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
                    
                    # Move gradient norm computation here
                    total_norm = 0
                    param_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_norms.append((name, param_norm.item()))
                    total_norm = total_norm ** 0.5
                    
                    if math.isfinite(total_norm):
                        optimizer.step()
                    else:
                        print(f"Skipping optimizer step due to non-finite gradient norm: {total_norm}")

                scheduler.step()

            # Check if train_loss is non-finite and raise an exception if it is
            if not torch.isfinite(loss):
                raise ValueError(f"Train loss is non-finite: {loss.item()}")
            
            # Calculate training accuracy
            train_accuracy = compute_accuracy(outputs, labels)
            total_accuracy += train_accuracy

            total_loss += loss.item()

            # Only execute validation step every 10 training steps
            if i % 10 == 9:
                # Validation step
                model.eval()
                with torch.no_grad():
                    try:
                        val_batch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dataloader)
                        val_batch = next(val_iter)
                    
                    minute_data = [tensor.to(device) for tensor in val_batch['minute_data']]
                    labels = val_batch['labels'].to(device)
                    
                    outputs = model(minute_data)
                    val_loss = val_criterion(outputs, labels).item()
                    total_val_loss += val_loss

                    # Calculate validation accuracy
                    val_accuracy = compute_accuracy(outputs, labels)
                    total_val_accuracy += val_accuracy

                model.train()
                
                # Log validation metrics
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }, step=global_step)

            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_accuracy": train_accuracy,
                "learning_rate": current_lr,
                'grad_scale': scaler.get_scale() if scaler else 1.0,
                'total_grad_norm': total_norm,
                'max_grad_norm': max(norm for _, norm in param_norms),
                'min_grad_norm': min(norm for _, norm in param_norms)
            }, step=global_step)

            # Update epoch progress bar
            epoch_progress.update(1)
            epoch_progress.set_postfix({
                't_l': f"{loss.item():.4f}",
                'v_l': f"{val_loss:.4f}",
                't_acc': f"{train_accuracy:.4f}",
                'v_acc': f"{val_accuracy:.4f}",
                'lr': f"{current_lr:.2e}",
            })

            global_step += 1

            # Save checkpoint every save_steps
            if global_step % save_steps == 0:
                checkpoint_name = f"{run_name}_checkpoint_step_{global_step}.pt"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                print(f"Saving checkpoint {checkpoint_name}")
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'global_step': global_step,
                    'epoch': epoch + 1,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'model_config': model_config,
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                    'run_name': run_name,
                    'shuffle_seed': seed,
                }, checkpoint_path)
                print(f"Checkpoint saved at step {global_step}: {checkpoint_path}")

            if i >= epoch_steps - 1:
                break

        # Close epoch progress bar
        epoch_progress.close()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(train_dataloader)
        avg_train_accuracy = total_accuracy / len(train_dataloader)
        avg_val_accuracy = total_val_accuracy / len(train_dataloader)

        # Update overall progress bar
        overall_progress.update(1)
        # elapsed_time = time.time() - start_time
        # eta = (elapsed_time / (epoch + 1)) * (num_epochs - epoch - 1)
        # overall_progress.set_postfix({
        #     'avg_train_loss': f"{avg_train_loss:.4f}",
        #     'avg_val_loss': f"{avg_val_loss:.4f}",
        #     'avg_train_acc': f"{avg_train_accuracy:.4f}",
        #     'avg_val_acc': f"{avg_val_accuracy:.4f}",
        #     'elapsed': f"{timedelta(seconds=int(elapsed_time))}",
        #     'eta': f"{timedelta(seconds=int(eta))}"
        # })

    # Close overall progress bar
    overall_progress.close()

    # Save final checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_final_checkpoint_epoch_{epoch+1}.pt")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': model_config,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'run_name': run_name,
        'shuffle_seed': seed,
    }, checkpoint_path)
    print(f"Final checkpoint saved: {checkpoint_path}")

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CheatCode Model",
    )
    parser.add_argument("--start_date", type=str, required=True, help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date for training data (YYYY-MM-DD)")
    parser.add_argument("--frame_period", type=int, default=60, help="Period between training frames in seconds")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for both training and validation")
    parser.add_argument("--nhead", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the current run")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Maximum norm for gradient clipping")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Number of steps per epoch (if None, use full dataset)")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive weight for BCEWithLogitsLoss")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count()//2, 
                        help="Number of worker processes for data loading")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="Number of batches loaded in advance by each worker")
    parser.add_argument("--restart", action='store_true', help="Restart the optimizer and scheduler while resuming training")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint to resume from")
    parser.add_argument("--warmup_factor", type=float, default=0.05, help="Warmup factor of total steps (default: 0.05)")
    parser.add_argument("--num_post_layers", type=int, default=None, help="Number of post-cross-attention transformer layers (default: same as num_encoder_layers)")
    parser.add_argument("--d_model_post", type=int, default=None, help="Dimension of the model for post-transformers (default: same as d_model)")
    parser.add_argument("--nhead_post", type=int, default=None, help="Number of attention heads for post-transformers (default: same as nhead)")
    parser.add_argument("--detect_anomaly", action='store_true', help="Enable torch.autograd.detect_anomaly during training")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the dataset before splitting")
    parser.add_argument("--disable_autocast", action='store_true', help="Disable automatic mixed precision (autocasting)")
    parser.add_argument("--dim_feedforward_input", type=int, default=2048,
                        help="Feedforward network dimension for input transformers")
    parser.add_argument("--dim_feedforward_post", type=int, default=None,  # Changed default to None
                        help="Feedforward network dimension for post transformers (default: same as dim_feedforward_input)")
    
    # Add the new argument for dataset file
    parser.add_argument("--dataset_file", type=str, default=None,
                        help="Path to the dataset file to load instead of connecting to the database")
    parser.add_argument("--save_dataset_file", type=str, default=None,
                        help="Path to save the dataset to an HDF5 file and exit")
    
    args = parser.parse_args()
    
    # Determine the device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using CUDA with {num_gpus} GPUs")
        else:
            print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        raise Exception("No CUDA or MPS device available. This script requires GPU acceleration.")
    
    # Set the default run_name to None
    run_name = args.run_name
    
    # Initialize the model with separate post-transformer hyperparameters
    model = CheatCodeModel(
        num_tickers=2,
        d_model_input=args.d_model,
        nhead_input=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        seq_len=1440,  # Assuming seq_len is 1440
        dropout=args.dropout,
        num_post_layers=args.num_post_layers if args.num_post_layers else args.num_encoder_layers,
        d_model_post=args.d_model_post if args.d_model_post else args.d_model,
        nhead_post=args.nhead_post if args.nhead_post else args.nhead,
        dim_feedforward_input=args.dim_feedforward_input,
        dim_feedforward_post=args.dim_feedforward_post if args.dim_feedforward_post is not None else args.dim_feedforward_input,  # Updated line
    )
    
    # Wrap model with DataParallel if using multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    if args.dataset_file is not None:
        # Load dataset from file
        print(f"Loading dataset from file {args.dataset_file}")
        full_dataset = CheatCodeDataset.from_file(args.dataset_file)
        # Retrieve seed from the dataset if available
        if hasattr(full_dataset, 'seed'):
            seed = full_dataset.seed
            print(f"Using shuffle seed from dataset file: {seed}")
        else:
            print("No shuffle seed found in dataset file.")
        # Update start_date and end_date if necessary
        start_date = full_dataset.start_date
        end_date = full_dataset.end_date
    else:
        # Proceed with dataset creation using the database
        # Database connection parameters
        db_params = {
            "dbname": "trade_data",
            "user": "postgres",
            "password": "mysecretpassword",
            "host": "localhost",
            "port": "5432"
        }
        
        # Create the full dataset
        full_dataset = CheatCodeDataset(start_date, end_date, db_params, args.frame_period)
    
    # Initialize variables for data splits and seed
    train_indices = None
    val_indices = None
    seed = None
    
    start_epoch = 0
    global_step = 0
    
    # Initialize the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    # Compute total steps and warmup steps
    total_steps = (args.steps_per_epoch or len(full_dataset) // args.batch_size) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_factor)
    
    # Initialize the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    
        # Retrieve shuffle seed from checkpoint if it exists
        if 'shuffle_seed' in checkpoint:
            seed = checkpoint['shuffle_seed']
            print(f"Using shuffle seed from checkpoint: {seed}")
        else:
            if args.shuffle:
                seed = random.randint(0, 2**32 - 1)
                print(f"No shuffle seed found in checkpoint; generated new seed: {seed}")
    
        if args.restart:
            print("Restarting training: optimizer and scheduler states are reinitialized")
            # Reset start_epoch and global_step when restarting
            start_epoch = 0
            global_step = 0
            # Generate new run_name if not provided
            if run_name is None:
                run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.uname().nodename
                print(f"Generated new run_name for restart: {run_name}")
            else:
                print(f"Using run_name from command line argument: {run_name}")
        else:
            # Resuming training: load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Ensure optimizer is on the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Set start_epoch and global_step from checkpoint
            start_epoch = checkpoint['epoch'] - 1
            global_step = checkpoint.get('global_step', 0)
            print(f"Resuming training from epoch {start_epoch}, global step {global_step}")

            # Retrieve run_name from the checkpoint if not provided
            if run_name is None:
                run_name = checkpoint.get('run_name', None)
                if run_name is None:
                    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.uname().nodename
                    print(f"No run_name found in checkpoint; generated new run_name: {run_name}")
                else:
                    print(f"Using run_name from checkpoint: {run_name}")
            else:
                print(f"Using run_name from command line argument: {run_name}")
    else:
        print("No checkpoint specified. Starting training from scratch.")
        # Generate run_name if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.uname().nodename
            print(f"Generated new run_name: {run_name}")
    
        # Generate a shuffle seed if shuffling
        if args.shuffle:
            seed = random.randint(0, 2**32 - 1)
            print(f"Generated shuffle seed: {seed}")
    
    # Update checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.save_dataset_file is not None:
        # Generate indices
        indices = list(range(len(full_dataset)))
        if args.shuffle:
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
                print(f"Generated shuffle seed: {seed}")
            random.seed(seed)
            random.shuffle(indices)
            print(f"Shuffled dataset with seed {seed}")
            # Save the seed in the dataset for later use
            full_dataset.seed = seed
        else:
            print("Dataset not shuffled.")
        # Apply the indices to the dataset
        full_dataset = full_dataset.reindex(indices)
        # Save the dataset to the specified HDF5 file
        print(f"Saving dataset to file {args.save_dataset_file}")
        full_dataset.save_to_file(args.save_dataset_file)
        print("Dataset saved. Exiting.")
        exit(0)
    
    # Create data splits
    dataset_size = len(full_dataset)

    # Generate indices
    indices = list(range(dataset_size))

    # No need to shuffle if loading from dataset file
    if args.shuffle and seed is not None:
        random.seed(seed)
        random.shuffle(indices)
        print(f"Shuffled dataset with seed {seed}")
    else:
        print("Dataset not shuffled.")

    # For every 10 frames of training data, take 1 frame for validation data
    train_indices = [i for i in indices if (i % 11) != 10]
    val_indices = [i for i in indices if (i % 11) == 10]
    
    # Create samplers using the indices
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    pin_memory_setting = not isinstance(model, nn.DataParallel)  # Set to False to disable pin_memory

    # Create data loaders with adjusted parameters
    train_dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory_setting,
        prefetch_factor=args.prefetch_factor
    )
    val_dataloader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory_setting,
        prefetch_factor=args.prefetch_factor
    )
    
    # Initialize wandb with the run name
    wandb.init(project="cheatcode", name=run_name, config=args)
    
    # Train the model
    train(
        model,
        train_dataloader,
        val_dataloader,
        args.num_epochs,
        device,
        checkpoint_dir,
        args.learning_rate,
        args.batch_size,
        args.nhead,
        args.d_model,
        args.num_encoder_layers,
        args.weight_decay,
        args.save_steps,
        run_name=run_name,
        max_grad_norm=args.max_grad_norm,
        steps_per_epoch=args.steps_per_epoch,
        pos_weight=args.pos_weight,
        start_epoch=start_epoch,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        detect_anomaly=args.detect_anomaly,
        warmup_steps=warmup_steps,  # Pass warmup_steps to the train function
        disable_autocast=args.disable_autocast,  # Pass the new argument here
    )