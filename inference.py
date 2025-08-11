import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from train_model import SpectrumDataset, CONFIG, setup_seed
from model.network_947331 import Network
import model.network_947331 as network_947331
import model.non_local_embedded_gaussian as non_local_embedded_gaussian
import sys
from math import sqrt
sys.modules['network_947331'] = network_947331
sys.modules['non_local_embedded_gaussian'] = non_local_embedded_gaussian
# Add Network class to safe globals for model loading
#torch.serialization.add_safe_globals([Network])

# Configuration parameters for inference
INFERENCE_CONFIG = {
    "cuda_device": "0",           # GPU device to use
    "checkpoints_dir": "./checkpoints",  # Directory containing model checkpoints
    "results_dir": "./inference_results",  # Directory to save inference results
    "batch_size": 16,           # Batch size for inference
    "random_seed": 2            # Random seed for reproducibility
}

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = INFERENCE_CONFIG["cuda_device"]

# Set random seed for reproducibility
setup_seed(INFERENCE_CONFIG["random_seed"])

def create_model():
    """Create a new model instance for inference.
    
    Returns:
        A new Network model instance
    """
    # Create a new model instance
    model = Network()
    model.eval()  # Set to evaluation mode
    
    return model

def run_inference(model, test_loader):
    """Run inference on the test dataset.
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for the test dataset
        
    Returns:
        Tuple containing lists of predictions and ground truth values
    """
    predictions = []
    ground_truth = []
    metadata = []  # To store additional metadata
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        for _, (img_batch, label_batch, mol_num, Fre_num, Fre_value, mol_kind) in enumerate(test_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()
            
            # Forward pass
            predict, _, _ = model(img_batch)
            
            # Move results back to CPU for analysis
            predict_cpu = predict.cpu().numpy()
            label_cpu = label_batch.cpu().numpy()
            
            # Store predictions and ground truth
            predictions.extend(predict_cpu.flatten().tolist())
            ground_truth.extend(label_cpu.flatten().tolist())
            
            # Store metadata for each sample
            for i in range(len(mol_num)):
                metadata.append({
                    'mol_num': mol_num[i].item(),
                    'Fre_num': Fre_num[i].item(),
                    'Fre_value': Fre_value[i].item(),
                    'mol_kind': mol_kind[i].item()
                })
    
    return predictions, ground_truth, metadata

def calculate_metrics(predictions, ground_truth):
    """Calculate performance metrics.
    
    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth values
        
    Returns:
        Dictionary containing calculated metrics
    """
    mse = mean_squared_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    mae = sqrt(mse)
    return {
        'mae': mae,
        'r2': r2
    }

def create_visualizations(predictions, ground_truth, metrics, save_dir):
    """Create and save visualizations of model performance.
    
    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth values
        metrics: Dictionary of calculated metrics
        save_dir: Directory to save the visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(ground_truth), min(predictions))
    max_val = max(max(ground_truth), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics to the plot
    plt.text(0.05, 0.95, f"MAE: {metrics['mae']:.4f}\nR²: {metrics['r2']:.4f}", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pred_vs_actual.svg'), format='svg')
    plt.close()
    
    # Create histogram of errors
    errors = np.array(predictions) - np.array(ground_truth)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.svg'), format='svg')
    plt.close()

def save_results_to_csv(predictions, ground_truth, metadata, metrics, save_dir):
    """Save inference results to CSV file.
    
    Args:
        predictions: List of model predictions
        ground_truth: List of ground truth values
        metadata: List of dictionaries containing sample metadata
        metrics: Dictionary of calculated metrics
        save_dir: Directory to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'actual': ground_truth,
        'predicted': predictions,
        'error': np.array(predictions) - np.array(ground_truth),
        'mol_num': [m['mol_num'] for m in metadata],
        'Fre_num': [m['Fre_num'] for m in metadata],
        'Fre_value': [m['Fre_value'] for m in metadata],
        'mol_kind': [m['mol_kind'] for m in metadata]
    })
    
    # Save to CSV
    results_df.to_csv(os.path.join(save_dir, 'inference_results.csv'), index=False)
    
    # Save metrics to a separate file
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"R²: {metrics['r2']:.6f}\n")

def load_best_model(checkpoints_dir):
    """Load the best model from checkpoints directory.
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        
    Returns:
        Loaded PyTorch model
    """
    # Check if directory exists
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory {checkpoints_dir} not found")
    
    # Get list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    # Find the best model (assuming naming convention includes loss value)
    # Sort by loss value if available, otherwise use the first checkpoint
    best_checkpoint = checkpoint_files[0]
    
    # Load the model

    checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
    print(f"Loading model from {checkpoint_path}")
    model = torch.load(checkpoint_path)    
    # Load state dict
    # model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Set to evaluation mode
    
    return model

def main():
    """Main function to run the inference process."""
    print('Begin inference')
    
    # Create results directory if it doesn't exist
    os.makedirs(INFERENCE_CONFIG["results_dir"], exist_ok=True)
    
    # Load test dataset
    test_data = SpectrumDataset("test")
    test_loader = Data.DataLoader(dataset=test_data, batch_size=INFERENCE_CONFIG["batch_size"], shuffle=False)
    
    # Load best model
    model = load_best_model(INFERENCE_CONFIG["checkpoints_dir"])
    if torch.cuda.is_available():
        model.cuda()
    
    # Run inference
    predictions, ground_truth, metadata = run_inference(model, test_loader)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    
    # Create visualizations
    create_visualizations(predictions, ground_truth, metrics, INFERENCE_CONFIG["results_dir"])
    
    # Save results to CSV
    save_results_to_csv(predictions, ground_truth, metadata, metrics, INFERENCE_CONFIG["results_dir"])
    
    print(f"Inference completed. Results saved to {INFERENCE_CONFIG['results_dir']}")

if __name__ == '__main__':
    main()