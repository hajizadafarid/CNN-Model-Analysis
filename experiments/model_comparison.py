import torch.optim as optim
from models.simplemodel import HCNN
from models.pretrainedmodel import PretrainedModel
from utils.training import train_one_epoch, evaluate
from utils.dataloader import set_all_seeds
import matplotlib.pyplot as plt
import os
import time
from torch import nn

def compare_models(scratch_model, pretrained_model, train_loader, eval_loader, device, EPOCHS=50, learning_rate=0.001, results_dir='results', plots_dir='plots'):
    """
    Compare training and evaluation performance between scratch and pretrained models.

    Args:
        scratch_model (nn.Module): The model trained from scratch.
        pretrained_model (nn.Module): The pretrained model.
        train_loader (DataLoader): Training DataLoader.
        eval_loader (DataLoader): Evaluation DataLoader.
        device (torch.device): Device to run computations on.
        EPOCHS (int, optional): Number of training epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        results_dir (str, optional): Directory to save results. Defaults to 'results'.
        plots_dir (str, optional): Directory to save plots. Defaults to 'plots'.
    
    Returns:
        dict: Results containing metrics for both models.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    results = {}
    
    # Define a helper function to train and collect metrics
    def train_model(model, optimizer, criterion):
        train_acc_history = []
        eval_acc_history = []
        train_loss_history = []
        eval_loss_history = []
        training_time = 0
        
        for epoch in range(1, EPOCHS + 1):
            # Train
            start_time = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            epoch_time = time.perf_counter() - start_time
            training_time += epoch_time
            
            # Evaluate
            eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
            
            # Record metrics
            train_acc_history.append(train_acc)
            eval_acc_history.append(eval_acc)
            train_loss_history.append(train_loss)
            eval_loss_history.append(eval_loss)
            
            # Print progress every 5 epochs
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{EPOCHS}")
                print(f"Training Accuracy: {train_acc:.2f}%")
                print(f"Evaluation Accuracy: {eval_acc:.2f}%")
                print(f"Epoch Time: {epoch_time:.2f}s")
                print("-" * 50)
        
        return {
            'train_acc_history': train_acc_history,
            'eval_acc_history': eval_acc_history,
            'train_loss_history': train_loss_history,
            'eval_loss_history': eval_loss_history,
            'training_time': training_time,
            'final_train_acc': train_acc_history[-1],
            'final_eval_acc': eval_acc_history[-1],
            'best_eval_acc': max(eval_acc_history)
        }
    
    # Train Scratch Model
    print("\nTraining Scratch Model...")
    set_all_seeds(42)
    optimizer_scratch = optim.SGD(scratch_model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scratch_results = train_model(scratch_model, optimizer_scratch, criterion)
    results['Scratch'] = scratch_results
    
    # Train Pretrained Model
    print("\nTraining Pretrained Model...")
    set_all_seeds(42)
    optimizer_pretrained = optim.SGD(pretrained_model.parameters(), lr=learning_rate, momentum=0.9)
    pretrained_results = train_model(pretrained_model, optimizer_pretrained, criterion)
    results['Pretrained'] = pretrained_results
    
    # Save final results
    results_file = os.path.join(results_dir, 'model_comparison_results.txt')
    with open(results_file, 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name} Model:\n")
            f.write(f"Final Training Accuracy: {metrics['final_train_acc']:.2f}%\n")
            f.write(f"Final Evaluation Accuracy: {metrics['final_eval_acc']:.2f}%\n")
            f.write(f"Best Evaluation Accuracy: {metrics['best_eval_acc']:.2f}%\n")
            f.write(f"Total Training Time: {metrics['training_time']:.2f}s\n")
    
    # Plot comparison
    plot_comparison(results, plots_dir)
    
    return results

def plot_comparison(results, plots_dir):
    """
    Plot training and evaluation accuracy comparisons between models.

    Args:
        results (dict): Dictionary containing metrics for each model.
        plots_dir (str): Directory to save plots.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for model_name, metrics in results.items():
        plt.plot(range(1, len(metrics['train_acc_history']) + 1), metrics['train_acc_history'], label=f'{model_name} Training')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot evaluation accuracy
    plt.subplot(1, 2, 2)
    for model_name, metrics in results.items():
        plt.plot(range(1, len(metrics['eval_acc_history']) + 1), metrics['eval_acc_history'], label=f'{model_name} Evaluation')
    plt.title('Evaluation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
