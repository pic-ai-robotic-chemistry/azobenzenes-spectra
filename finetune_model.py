import argparse
import torch
import torch.utils.data as Data
import model.network_947331 as network_947331
import model.non_local_embedded_gaussian as non_local_embedded_gaussian
from model.network_947331_freeze import Network
from torch import nn
import time
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import sys

sys.modules['network_947331'] = network_947331
sys.modules['non_local_embedded_gaussian'] = non_local_embedded_gaussian
# Configuration parameters
CONFIG = {
    "cuda_device": "0",           # GPU device to use
    "data_file": "./data/IR_Raman_azo.csv",  # Input data file
    "pre_weights_dir": "checkpoints",  # Directory with pretrained weights
    "batch_size": 5,             # Batch size for training
    "epochs": 200,              # Number of training epochs
    "learning_rate": 0.005,     # Initial learning rate
    "random_seed": 2,           # Random seed for reproducibility
    "train_size": 10,           # Number of samples for training
    "test_size": 10,            # Number of samples for testing
    "max_saved_models": 2       # Maximum number of models to keep
}

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["cuda_device"]

class SpectrumDataset(Dataset):
    """Dataset class for loading and processing spectrum data."""

    DATA_COLUMN = 8000  # Length of input feature vector x
    
    def __init__(self, train_or_test: str, random_state: int = 1) -> None:
        """Initialize the dataset.
        
        Args:
            train_or_test: Either 'train' or 'test' to specify the dataset split
            random_state: Random seed for data shuffling
        """
        super().__init__()
        self.train_or_test = train_or_test
        self.random_state = random_state
        self.data_df = self._load_from_csv(CONFIG["data_file"])
         
    def __getitem__(self, index):
        """Get a single data item from the dataset.
        
        Returns:
            Tuple containing (x, y, mol_num, Fre_num, Fre_value, mol_kind)
        """
        # Load input features
        x = torch.tensor(self.data_df.iloc[index, :self.DATA_COLUMN], dtype=torch.float)
        # Load target value
        y = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+5])])
        # Load metadata
        mol_num = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+1])])     # Molecule number
        Fre_num = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+2])])     # Frequency number
        Fre_value = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+3])])   # Frequency value
        mol_kind = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN])])      # Molecule type
        
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
        transfer_data_new = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        if self.train_or_test == 'train':
            # Get training data subset
            new_train_data = transfer_data_new.iloc[0:CONFIG["train_size"]]
            return new_train_data
        elif self.train_or_test == 'test':
            # Get testing data subset
            new_test_data = transfer_data_new.iloc[CONFIG["train_size"]:CONFIG["train_size"]+CONFIG["test_size"]]
            return new_test_data
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


def load_pretrained_model():
    """Load the best pretrained model from the specified directory.
    
    Returns:
        Pretrained network with loaded weights
    """
    # Find all model files
    files = os.listdir(CONFIG["pre_weights_dir"])
    filename_pth = []
    for file in files:
        if file.endswith(".pth"):
            filename_pth.append(file)
    
    # Sort models by loss value (ascending)
    filename_pth.sort(key=lambda x: float(x[14:-4]))
    
    # Load the best model (lowest loss)
    model_path = os.path.join(CONFIG["pre_weights_dir"], filename_pth[0])
    print(f"Loading pretrained model from {model_path}")
    
    # Load pretrained model
    old_model = torch.load(model_path)
    
    # Initialize new model
    net = Network()
    
    # Transfer weights from pretrained model to new model
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in old_model.state_dict().items() if k in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    
    return net


if __name__ == '__main__':
    print('Begin fine-tuning')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune a network with a specific random state.')
    parser.add_argument('--random_state', type=int, required=True, help='The random state for shuffling the dataset.')
    args = parser.parse_args()

    random_state = args.random_state

    # Create dynamic weights directory name based on random state
    weights_dir = f'fine_tune_weights_{random_state}'

    # Check if weights directory already exists
    if os.path.exists(weights_dir):
        raise Exception(f"Directory {weights_dir} already exists. Please choose a different random_state value.")
    else:
        os.makedirs(weights_dir)
        
    # Load datasets
    train_data = SpectrumDataset("train", random_state=random_state)
    test_data = SpectrumDataset("test", random_state=random_state)

    # Create data loaders
    train_loader = Data.DataLoader(dataset=train_data, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=CONFIG["batch_size"], shuffle=False)

    # Load pretrained model
    net = load_pretrained_model()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
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

    # Define loss function for evaluation
    loss_func = nn.L1Loss()

    # Training loop
    for epoch_index in range(CONFIG["epochs"]):
        # Print current epoch and learning rate
        print(f"Epoch {epoch_index}, Learning Rate: {opt.param_groups[0]['lr']}")
        start_time = time.time()

        # Training phase
        torch.set_grad_enabled(True)
        net.train()
        for _, (img_batch, _, _, _, _, mol_kind_batch) in enumerate(train_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                mol_kind_batch = mol_kind_batch.cuda()

            # Forward pass
            predict, w_1, w_2 = net(img_batch)

            # Custom loss calculation
            try:
                # Sort predictions by molecule kind
                _, sorted_indices = torch.sort(mol_kind_batch.squeeze())
                sorted_preds = predict[sorted_indices]

                # Calculate custom loss between sorted predictions
                loss = 0
                for i in range(len(sorted_preds)):
                    for j in range(i + 1, len(sorted_preds)):
                        loss += torch.relu(torch.relu(sorted_preds[i]) - sorted_preds[j] + (j-i))
            except Exception as e:
                print(f"Error in loss calculation: {e}")
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

        for _, (img_batch, _, _, _, _, mol_kind_batch) in enumerate(test_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                mol_kind_batch = mol_kind_batch.cuda()

            # Forward pass
            predict, _, _ = net(img_batch)
            
            # Calculate loss
            loss = loss_func(predict, mol_kind_batch)
            
            # Collect statistics
            total_loss.append(loss)

        # Calculate mean loss
        mean_loss = sum(total_loss) / len(total_loss)

        # Update learning rate based on validation performance
        scheduler.step(mean_loss.item())

        # Print evaluation results
        print(f"Total loss: {sum(total_loss)}, Batches: {len(total_loss)}, Mean loss: {mean_loss}")
        print(f"[Test] epoch[{epoch_index}/{CONFIG['epochs']}] loss:{mean_loss.item():.4f}")

        # Save model
        weight_path = os.path.join(weights_dir, f"net_mean_loss_{mean_loss.item():.4f}.pth")
        print(f"Saving model to {weight_path}\n")
        
        torch.save(net, weight_path)

        # Log loss to CSV
        loss_data = pd.DataFrame([mean_loss.item()])
        loss_data.to_csv(os.path.join(weights_dir, "tt.csv"), mode='a+', index=None, header=None)

        # Keep only the best N models and remove others to save disk space
        files = os.listdir(weights_dir)
        filename_pth = [file for file in files if file.endswith(".pth")]
        filename_pth.sort(key=lambda x: float(x[14:-4]))  

        if len(filename_pth) > CONFIG["max_saved_models"]:
            for i in range(CONFIG["max_saved_models"], len(filename_pth)):
                os.remove(os.path.join(weights_dir, filename_pth[i]))

