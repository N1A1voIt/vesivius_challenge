# Vesuvius Challenge

A machine learning project for the Vesuvius Challenge - digitally unwrapping and reading ancient scrolls carbonized by the eruption of Mount Vesuvius.

## Project Structure

```
vesuvius/
│
├── notebooks/
│   ├── local_experiments.ipynb  # Local experimentation and debugging
│   └── kaggle_train.ipynb       # Kaggle training notebook
│
├── src/
│   ├── data/
│   │   └── loader.py           # Data loading utilities
│   ├── models/
│   │   └── unet3d.py           # 3D UNet model architecture
│   ├── utils/
│   │   └── helpers.py          # Helper utilities
│   └── train.py                # Main training script
│
├── configs/
│   └── vesuvius.yaml           # Training configuration
│
├── data/
│   ├── samples/                # Small sample data for debugging
│   └── raw/                    # Full dataset (git ignored)
│
├── requirements.txt            # Python dependencies
└── README.md
```

## Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the competition data from Kaggle and extract to `data/raw/`:

```bash
kaggle competitions download -c vesuvius-challenge-ink-detection
unzip vesuvius-challenge-ink-detection.zip -d data/raw/
```

### 3. Local Training

```bash
# Train with debug mode (uses sample data)
python src/train.py --debug

# Train with full data
python src/train.py --config configs/vesuvius.yaml
```

### 4. Kaggle Training

1. Upload the `src/` directory as a Kaggle dataset
2. Create a new notebook and add the dataset
3. Copy/adapt code from `notebooks/kaggle_train.ipynb`

## Model Architecture

The project uses a 3D UNet architecture designed for volumetric segmentation of CT scan data:

- **Input**: 3D volumes of CT scan slices (default: 16 z-slices × 256 × 256)
- **Output**: Binary segmentation mask indicating ink presence
- **Architecture**: Encoder-decoder with skip connections

## Evaluation

The competition uses the F0.5 score, which weighs precision more heavily than recall:

```
F0.5 = (1 + 0.5²) × (precision × recall) / (0.5² × precision + recall)
```

## Configuration

Edit `configs/vesuvius.yaml` to customize:

- Model architecture (features, attention)
- Training hyperparameters (epochs, batch size, learning rate)
- Data augmentation settings
- Early stopping parameters

## License

This project is for the Vesuvius Challenge competition. See competition rules for data usage terms.
