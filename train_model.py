import torch
import torch.utils.data as Data
from model.network_947331 import Network
from torch import nn
import time
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import pandas as pd
import os

# Configuration parameters
CONFIG = {
    "cuda_device": "0",           # GPU device to use
    "results_dir": "./results",   # Directory to save model weights
    "loss_log_file": "./results/train_log.csv",   # File to log loss values
    "max_saved_models": 5,       # Maximum number of models to keep
    "batch_size": 16,           # Batch size for training
    "epochs": 200,              # Number of training epochs
    "learning_rate": 0.001,     # Initial learning rate
    "random_seed": 2            # Random seed for reproducibility
}

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_device"]

class SpectrumDataset(Dataset):
    """Dataset class for loading and processing spectrum data."""

    DATA_COLUMN = 8000  # Length of input feature vector x
    
    def __init__(self, train_or_test: str) -> None:
        """Initialize the dataset.
        
        Args:
            train_or_test: Either 'train' or 'test' to specify the dataset split
        """
        super().__init__()
        self.train_or_test = train_or_test
        self.data_df = self._load_from_csv('data/sign_minmax_4400.csv')
         
    def __getitem__(self, index):
        """Get a single data item from the dataset.
        
        Returns:
            Tuple containing (x, y, mol_num, Fre_num, Fre_value, mol_kind)
        """
        # Load input features
        x = torch.tensor(self.data_df.iloc[index, :self.DATA_COLUMN], dtype=torch.float)
        # Load target value
        y = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN])])
        # Load metadata
        mol_num = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+1])])     # Molecule number
        Fre_num = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+2])])     # Frequency number
        Fre_value = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+3])])   # Frequency value
        mol_kind = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+5])])    # Molecule type
        
        # Reshape input to (2, DATA_COLUMN/2) format for CNN processing
        x = x.reshape(int(self.DATA_COLUMN/2), 2)
        x = x.T
        
        return x, y, mol_num, Fre_num, Fre_value, mol_kind

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data_df)

    def _load_from_csv(self, fp: str):
        """Load data from CSV file and split into train/test sets.
        
        Args:
            fp: File path to the CSV data file
            
        Returns:
            DataFrame containing either training or testing data
        """
        # Load CSV without headers
        df = pd.read_csv(fp, header=None) 

        # Shuffle data with fixed random seed for reproducibility
        transfer_data_new = df.sample(frac=1, random_state=CONFIG["random_seed"]).reset_index(drop=True)
        len_transfer = len(transfer_data_new)

        # Split data into train/test sets (80/20 split)
        if self.train_or_test == 'train':
            train_data = transfer_data_new.iloc[0:int(len_transfer * 0.8)]
            return train_data
        elif self.train_or_test == 'test':
            test_data = transfer_data_new.iloc[int(len_transfer * 0.8):]
            return test_data
        else:
            raise ValueError(f"The value passed to SpectrumDataset must be 'train' or 'test', but got '{self.train_or_test}'")


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
setup_seed(CONFIG["random_seed"])

if __name__ == '__main__':
    print('Begin training')
    
    # Create results directory if it doesn't exist
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    
    # Load datasets
    train_data = SpectrumDataset("train")
    test_data = SpectrumDataset("test")

    # Create data loaders
    train_loader = Data.DataLoader(dataset=train_data, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=CONFIG["batch_size"], shuffle=False)

    # Initialize model
    net = Network()
    if torch.cuda.is_available():
        # Move model to GPU
        # Uncomment the following line to use multiple GPUs
        # net = nn.DataParallel(net)
        net.cuda()

    
    # Initialize optimizer and learning rate scheduler
    opt = torch.optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.8,
        patience=10,
        verbose=False,
        min_lr=0,
        eps=1e-08
    )

    # Define loss function
    loss_func = nn.MSELoss()
    # Training loop
    for epoch_index in range(CONFIG["epochs"]):
        # Print current epoch and learning rate
        print(f"Epoch {epoch_index}, Learning Rate: {opt.param_groups[0]['lr']}")
        start_time = time.time()

        # Training phase
        torch.set_grad_enabled(True)
        net.train()
        for _, (img_batch, label_batch, _, _, _, _) in enumerate(train_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            # Forward pass
            predict, _, _ = net(img_batch)

            # Calculate loss
            try:
                loss = loss_func(predict, label_batch)
            except Exception as e:
                print(f"Error in loss calculation: {e}")
                print(f"Prediction shape: {predict.shape}")
                continue

            # Backward pass and optimization
            net.zero_grad()
            loss.backward()    # Backpropagation
            opt.step()         # Update weights

        # Print epoch training time
        print(f"(LR:{opt.param_groups[0]['lr']:.6f}) Time of epoch: {time.time()-start_time:.4f}s")

        # Evaluation phase
        torch.set_grad_enabled(False)
        net.eval()
        total_loss = []
        total_sample = 0

        for _, (img_batch, label_batch, _, _, _, _) in enumerate(test_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()

            # Forward pass
            predict, _, _ = net(img_batch)
            
            # Calculate loss
            loss = loss_func(predict, label_batch)
            
            # This line seems unnecessary and might cause errors since predict is not a classification output
            # predict = predict.argmax(dim=1)
            
            # Collect statistics
            total_loss.append(loss)
            total_sample += img_batch.size(0)

        # Calculate mean loss
        mean_loss = sum(total_loss) / len(total_loss)

        # Update learning rate based on validation performance
        scheduler.step(mean_loss.item())

        # Print evaluation results
        print(f"Total loss: {sum(total_loss)}, Batches: {len(total_loss)}, Mean loss: {mean_loss}")
        print(f"[Test] epoch[{epoch_index}/{CONFIG['epochs']}] loss:{mean_loss.item():.4f}")

        # Save model
        weight_path = os.path.join(CONFIG["results_dir"], f"netr_mean_loss_{mean_loss.item():.4f}.pth")
        print(f"Saving model to {weight_path}\n")
        
        # Ensure model is on GPU before saving
        if torch.cuda.is_available():
            net.cuda()
        torch.save(net, weight_path)
        
        # Log loss to CSV
        loss_data = pd.DataFrame([mean_loss.item()])
        loss_data.to_csv(CONFIG["loss_log_file"], mode='a+', index=None, header=None)

        # Keep only the best N models and remove others to save disk space
        files = os.listdir(CONFIG["results_dir"])
        filename_pth = []
        for file in files:
            if file.endswith(".pth"):
                filename_pth.append(file)  # Get model filenames
                
        # Sort models by loss value (ascending)
        filename_pth.sort(key=lambda x: float(x[15:-4]))  
        
        # Remove excess models beyond the configured maximum
        if len(filename_pth) > CONFIG["max_saved_models"]:
            for i in range(CONFIG["max_saved_models"], len(filename_pth)):
                os.remove(os.path.join(CONFIG["results_dir"], filename_pth[i]))
                print(f"Removed excess model: {filename_pth[i]}")

