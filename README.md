# Melanoma Classification Project

This project trains a deep learning model to classify skin lesion images as either benign or malignant. It uses the InceptionV3 architecture with transfer learning, implemented in TensorFlow/Keras. Experiment tracking is handled by Weights & Biases (W&B).

## Project Structure

```
.
├── .github/workflows/run.yml   # GitHub Action for automated training
├── data/                       # (Recommended) All data should be placed here
│   ├── train_data/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── validation_data/
│       ├── Benign/
│       └── Malignant/
├── .gitignore                  # Ignores data, models, and cache files
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── train.py                    # Main training script
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Weights & Biases (Optional):**
    If you want to log experiments, you will need a W&B account.
    ```bash
    wandb login
    ```

## Training the Model

The `train.py` script is used to train the model. It accepts several command-line arguments to configure the training process.

### Basic Usage

To run the training with default parameters:
```bash
python train.py
```

### Custom Usage

You can override the default settings using command-line arguments.

**Example:** Train for 20 epochs with a batch size of 16 and a learning rate of 0.001.
```bash
python train.py --epochs 20 --batch_size 16 --learning_rate 0.001 --model_save_path "my_best_model.h5"
```

### All Arguments

- `--project_name`: W&B project name (default: `melanoma`).
- `--entity`: W&B entity (default: `suphawansr20-chiang-mai-university`).
- `--model_name`: Name of the model architecture (default: `InceptionV3`).
- `--train_dir`: Directory for training data (default: `train_data`).
- `--val_dir`: Directory for validation data (default: `validation_data`).
- `--image_size`: Image size (default: `224`).
- `--batch_size`: Batch size (default: `32`).
- `--epochs`: Number of epochs (default: `10`).
- `--learning_rate`: Learning rate (default: `0.0001`).
- `--patience`: Patience for early stopping (default: `3`).
- `--model_save_path`: Path to save the best model (default: `InceptionV3_best.h5`).

## Data

The model expects the data to be organized in the following structure:
- `train_data/Benign/`
- `train_data/Malignant/`
- `validation_data/Benign/`
- `validation_data/Malignant/`

**Note:** The data folders are included in the `.gitignore` file to prevent them from being committed to the repository. You should manage your data separately.