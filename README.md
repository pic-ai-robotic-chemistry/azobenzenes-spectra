# ATT-CNN Neural Network

This repository contains the implementation of a deep learning model as described in our research article titled **Unlocking azobenzene isomerization mechanisms via an LLM agent-driven workflow integrating simulation, experiment, and machine learning**. The model uses a convolutional neural network with non-local blocks to predict C-N=N-C dihedral angle from IR and Raman spectral data.

## Environment Setup

These packages were used:

```
Python 3.7
PyTorch 1.10.1
Pandas
Scikit-learn
Matplotlib
```

You can set up the environment using pip:

```bash
pip install torch==1.10.1 pandas scikit-learn matplotlib
```

Or create a conda environment:

```bash
conda create -n att_cnn python=3.7
conda activate att_cnn
conda install pandas scikit-learn matplotlib
# For NVIDIA GPU
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# For CPU only
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
```

## Repository Structure

- `train_model.py`: Script for training the model from raw data
- `inference.py`: Script for running inference with a trained model
- `finetune_model.py`: Script for fine-tuning a pre-trained model
- `model/`: Directory containing model architecture definitions
  - `network_947331.py`: Main network architecture
  - `non_local_embedded_gaussian.py`: Implementation of non-local blocks
  - `network_947331_freeze.py`: Modified network for fine-tuning
- `data/`: Directory containing input data files
- `checkpoints/`: Directory for storing model checkpoints

## Training the Model

To train the model from scratch using the raw data, run:

```bash
python train_model.py
```

This script will:
1. Load and preprocess the data from `data/sign_minmax_4400.csv`
2. Initialize the neural network model
3. Train the model for the specified number of epochs
4. Save model checkpoints to the `results` directory
5. Keep track of the best performing models based on validation loss

Training parameters can be adjusted in the `CONFIG` dictionary at the top of the script.

## Running Inference

To run inference using a trained model, execute:

```bash
python inference.py
```

This script will:
1. Load the model we've trained from the `checkpoints` directory
2. Run inference on the test dataset
3. Calculate performance metrics (MAE and R²)
4. Generate visualizations of model performance
5. Save results to the `inference_results` directory

## Fine-tuning the Model

To fine-tune a pre-trained model on a new dataset, run:

```bash
python finetune_model.py --random_state <value>
```

Where `<value>` is an integer used as the random seed for data shuffling. This script will:
1. Load the pre-trained model from the `checkpoints` directory
2. Load and preprocess the fine-tuning data from `data/IR_Raman_azo.csv`
3. Fine-tune the model using a custom loss function
4. Save the fine-tuned model to a new directory named `fine_tune_weights_<random_state>`

## Model Architecture

The model architecture consists of:
- Three convolutional blocks with batch normalization, ReLU activation, and max pooling
- Two non-local blocks for capturing long-range dependencies
- Fully connected layers for final prediction

## Code for preparing GraphRAG input files in the research

Relevant code can be found in directory "text_preprocess". It includes its own README file covering the details.

## Citation

If you use this code in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## License

MIT
