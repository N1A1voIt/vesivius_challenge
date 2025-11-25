# Sample Data Directory

This directory contains small sample data for debugging and local development.

## Structure

Place sample fragments here with the following structure:

```
samples/
├── train/
│   └── fragment_1/
│       ├── surface_volume/
│       │   ├── 00000.tif
│       │   ├── 00001.tif
│       │   └── ...
│       ├── mask.png
│       └── inklabels.png
└── val/
    └── fragment_2/
        └── ...
```

## Creating Sample Data

To create sample data from the full dataset:

1. Copy a small region (e.g., 512x512) from a fragment
2. Include corresponding mask and labels
3. Use for quick iteration during development

## Usage

```python
from src.data.loader import VesuviusDataset

dataset = VesuviusDataset(
    data_dir='data/samples',
    split='train',
)
```
