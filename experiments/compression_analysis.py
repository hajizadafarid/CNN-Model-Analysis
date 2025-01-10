import torch.optim as optim
from models.simplemodel import HCNN, test_compression
from utils.training import train_and_eval
from utils.dataloader import set_all_seeds
import os
from torch import nn

def analyze_svd_compression(train_loader, eval_loader, device, compression_ratios=[0.1, 0.2], results_dir='results'):
    """
    Analyze the effect of different SVD compression ratios on model performance.

    Args:
        train_loader (DataLoader): Training DataLoader.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run computations on.
        compression_ratios (list, optional): List of compression ratios to test. Defaults to [0.1, 0.2].
        results_dir (str, optional): Directory to save results. Defaults to 'results'.
    
    Returns:
        list: List of results dictionaries for each compression ratio.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'svd_compression_results.txt')
    
    results = []
    
    # Best hyperparameters from your previous experiments
    best_num_layers = 5
    best_kernel_size = 3
    learning_rate = 0.001
    EPOCHS = 50
    
    with open(results_file, 'a') as f:
        # First evaluate the baseline model
        set_all_seeds(42)
        base_model = HCNN(num_of_conv_layers=best_num_layers,
                          kernel_size=best_kernel_size,
                          dropout=True,
                          dropout_fc=0.1).to(device)
        
        optimizer = optim.SGD(base_model.parameters(), lr=learning_rate, weight_decay=0.01)
        base_train_acc, base_eval_acc, _ = train_and_eval(base_model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device)
        base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        
        # Log baseline model results
        f.write(f"Baseline Model:\n")
        f.write(f"Parameters: {base_params:,}\n")
        f.write(f"Training Accuracy: {base_train_acc:.2f}%\n")
        f.write(f"Evaluation Accuracy: {base_eval_acc:.2f}%\n")
        f.write("-" * 50 + "\n")
        
        results.append({
            'ratio': 1.0,
            'params': base_params,
            'train_acc': base_train_acc,
            'eval_acc': base_eval_acc
        })
        
        # Test different compression ratios
        for ratio in compression_ratios:
            set_all_seeds(42)  # Reset seeds for reproducibility
            
            # Create and compress the model
            model = HCNN(num_of_conv_layers=best_num_layers,
                        kernel_size=best_kernel_size,
                        dropout=True,
                        dropout_fc=0.1).to(device)
            
            compressed_model = test_compression(model, compression_ratio=ratio)
            optimizer = optim.SGD(compressed_model.parameters(), lr=learning_rate)
            
            # Train and evaluate the compressed model
            train_acc, eval_acc, _  = train_and_eval(compressed_model, EPOCHS, optimizer, nn.CrossEntropyLoss(), train_loader, eval_loader, device)
            params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
            
            # Store results
            results.append({
                'ratio': ratio,
                'params': params,
                'train_acc': train_acc,
                'eval_acc': eval_acc
            })
            
            # Log compression results
            f.write(f"\nCompression Ratio: {ratio}\n")
            f.write(f"Parameters: {params:,} ({params/base_params:.2%} of baseline)\n")
            f.write(f"Training Accuracy: {train_acc:.2f}%\n")
            f.write(f"Evaluation Accuracy: {eval_acc:.2f}%\n")
            f.write(f"Accuracy Change: {eval_acc - base_eval_acc:+.2f}%\n")
            f.write("-" * 50 + "\n")
    
    return results
