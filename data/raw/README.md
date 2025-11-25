# Raw Data Directory

This directory is ignored by git. Place the full Vesuvius Challenge dataset here.

## Download Instructions

1. Go to the Kaggle competition page
2. Download the dataset
3. Extract to this directory

## Expected Structure

```
raw/
├── train/
│   ├── 1/
│   │   ├── surface_volume/
│   │   ├── mask.png
│   │   └── inklabels.png
│   ├── 2/
│   └── 3/
└── test/
    ├── a/
    └── b/
```

## Note

The raw data is several GB in size and should not be committed to git.
