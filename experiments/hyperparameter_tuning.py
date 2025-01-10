import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.simplemodel import HCNN
from utils.training import train_and_eval, train_and_eval_with_loss_and_acc
from utils.dataloader import set_all_seeds, create_train_eval_dataloaders
from torch.optim import Optimizer
from typing import List

def checking_num_layers_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different numbers of convolutional layers.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'num_layers_results.txt')
    
    with open(results_file, 'w') as f:   
        for num_layers in range(3, 10):
            EPOCHS = 50
            learning_rate = 0.001
            set_all_seeds(42)
            
            # Initialize the model with varying number of convolutional layers
            model = HCNN(num_of_conv_layers=num_layers).to(device)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            # Train and evaluate the model
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            # Write results to file
            f.write(f"Best Training accuracy when num_conv_layers {num_layers}: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation accuracy when num_conv_layers {num_layers}: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time when num_conv_layers {num_layers}: {total_time:.2f}s\n\n")


def checking_kernel_sizes_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different kernel sizes.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'kernel_size_results.txt')
    best_num_layers = 5
    
    with open(results_file, 'w') as f:
        for kernel_size in range(3, 9, 2):
            EPOCHS = 50
            learning_rate = 0.001
            set_all_seeds(42)
            
            model = HCNN(num_of_conv_layers=best_num_layers, kernel_size=kernel_size).to(device)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Best Training accuracy with kernel size {kernel_size}: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation accuracy with kernel size {kernel_size}: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time with kernel size {kernel_size}: {total_time:.2f}s\n\n")


def checking_skip_connections_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of skip connections.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'skip_connections_results.txt')
    best_num_layers = 5
    best_kernel_size = 3
    
    with open(results_file, 'w') as f:
        EPOCHS = 50
        learning_rate = 0.001
        set_all_seeds(42)
        
        model = HCNN(
            num_of_conv_layers=best_num_layers,
            kernel_size=best_kernel_size,
            skipping=True
        ).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        best_train_acc, best_eval_acc, total_time = train_and_eval(
            model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
        )
        
        f.write(f"Best Training accuracy with skipping: {best_train_acc:.2f}%\n")
        f.write(f"Best Evaluation accuracy with skipping: {best_eval_acc:.2f}%\n")
        f.write(f"Training Time with skipping: {total_time:.2f}s\n\n")


def checking_normalisation_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different normalization techniques.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'normalisation_results.txt')
    best_num_layers = 5
    best_kernel_size = 3
    skipping = False
    normalisations = ["batch", "group", "layer", "instance"]
    
    with open(results_file, 'w') as f:  
        for norm_type in normalisations:
            EPOCHS = 50
            learning_rate = 0.001
            set_all_seeds(42)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                skipping=skipping,
                normalisation=norm_type
            ).to(device)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Best Training accuracy with {norm_type} norm: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation accuracy with {norm_type} norm: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time with {norm_type} norm: {total_time:.2f}s\n\n")


def checking_dropout_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different dropout rates.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'dropout_results.txt')
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    skipping = False
    dropout_vals = torch.arange(0.1, 1.1, 0.1)
    
    with open(results_file, 'w') as f:  
        for dropout_fc in dropout_vals:
            set_all_seeds(42)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                skipping=skipping,
                dropout=True,
                dropout_fc=dropout_fc.item()
            ).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Best Training accuracy with dropout {dropout_fc:.1f}: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation accuracy with dropout {dropout_fc:.1f}: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time with dropout {dropout_fc:.1f}: {total_time:.2f}s\n\n")


def checking_L2_regularization_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different L2 regularization (weight decay) values.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'weight_decay_results.txt')
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    learning_rate = 0.001
    weight_decay_values = [1e-4, 1e-3, 1e-2]
    
    with open(results_file, 'w') as f:  
        for wd in weight_decay_values:
            set_all_seeds(42)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                dropout=True,
                dropout_fc=0.1
            ).to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Weight Decay: {wd}\n")
            f.write(f"Best Training Accuracy: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation Accuracy: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time: {total_time:.2f}s\n")
            f.write("-" * 30 + "\n")


def checking_learning_rates_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different learning rates.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'learning_rates_results.txt')
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    with open(results_file, 'w') as f:
        f.write("\n=== Learning Rate Analysis ===\n")
        for lr in learning_rates:
            set_all_seeds(42)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                dropout=True,
                dropout_fc=0.1
            ).to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=lr)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Best Training Accuracy: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation Accuracy: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time: {total_time:.2f}s\n")
            f.write("-" * 30 + "\n")


def checking_batch_sizes_simplemodel(train_loader_fn, eval_loader_fn, device, results_dir='results'):
    """
    Experiment to check the effect of different batch sizes.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'batch_sizes_results.txt')
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    learning_rate = 0.001
    batch_sizes = [16, 32, 64, 128]
    
    with open(results_file, 'w') as f:
        f.write("\n=== Batch Size Analysis ===\n")
        for batch_size in batch_sizes:
            set_all_seeds(42)
            
            # Recreate data loaders with the new batch size
            train_loader, eval_loader = train_loader_fn(batch_size=batch_size)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                dropout=True,
                dropout_fc=0.1
            ).to(device)
            
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Best Training Accuracy: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation Accuracy: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time: {total_time:.2f}s\n")
            f.write("-" * 30 + "\n")


def checking_optimizer_types_simplemodel(train_loader, eval_loader, device, results_dir='results'):
    """
    Experiment to check the effect of different optimizer types.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'optimizer_types_results.txt')
    EPOCHS = 50
    best_num_layers = 5
    best_kernel_size = 3
    learning_rate = 0.001
    
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=learning_rate),
        'SGD_momentum': lambda params: optim.SGD(params, lr=learning_rate, momentum=0.9),
        'Adam': lambda params: optim.Adam(params, lr=learning_rate),
        'RMSprop': lambda params: optim.RMSprop(params, lr=learning_rate)
    }
    
    with open(results_file, 'w') as f:
        f.write("\n=== Optimizer Types Analysis ===\n")
        for opt_name, opt_func in optimizers.items():
            set_all_seeds(42)
            
            model = HCNN(
                num_of_conv_layers=best_num_layers,
                kernel_size=best_kernel_size,
                dropout=True,
                dropout_fc=0.1
            ).to(device)
            
            optimizer = opt_func(model.parameters())
            
            best_train_acc, best_eval_acc, total_time = train_and_eval(
                model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device
            )
            
            f.write(f"Optimizer: {opt_name}\n")
            f.write(f"Best Training Accuracy: {best_train_acc:.2f}%\n")
            f.write(f"Best Evaluation Accuracy: {best_eval_acc:.2f}%\n")
            f.write(f"Training Time: {total_time:.2f}s\n")
            f.write("-" * 30 + "\n")


def run_all_hyperparameter_tuning(train_loader, eval_loader, device, results_dir='results'):
    """
    Runs all hyperparameter tuning experiments.
    """
    checking_num_layers_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_kernel_sizes_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_skip_connections_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_normalisation_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_dropout_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_L2_regularization_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_learning_rates_simplemodel(train_loader, eval_loader, device, results_dir)
    checking_batch_sizes_simplemodel(
        lambda batch_size: create_train_eval_dataloaders(
            root_dir="./101_ObjectCategories",
            train_samples_per_class=15,
            eval_samples_per_class=15,
            batch_size=batch_size
        ),
        lambda batch_size: create_train_eval_dataloaders(
            root_dir="./101_ObjectCategories",
            train_samples_per_class=15,
            eval_samples_per_class=15,
            batch_size=batch_size
        ),
        device,
        results_dir
    )
    checking_optimizer_types_simplemodel(train_loader, eval_loader, device, results_dir)
