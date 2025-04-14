<!--
-*- coding: utf-8 -*-

 Author: [Your Name]
 License: MIT
-->

# Neuron Cell Type Classification with Custom scVI Models

## Table of Contents
1. [General Info](#general-info)
2. [Model and Data](#model-and-data)
3. [Build and Run](#build-and-run)
4. [Results](#results)
5. [License](#license)

<a name="general-info"></a>
## General Info

This repository contains an implementation of custom scVI (single-cell Variational Inference) models based on the paper "Identifying cell types in single-cell transcriptomics data with scVI" by Gayoso et al. The implementation extends the original VAE architecture with additional features and applies it to neuronal cell type classification using single-cell RNA sequencing data from the Allen Brain Atlas.

### Repository Structure

```
vae_scvi_neuro/
├── src/                    # Source code (functions, classes, utilities)
│   ├── __init__.py
│   ├── data/               # Data processing modules
│   │   ├── __init__.py
│   │   ├── loaders.py      # Functions for loading data from Allen Brain Atlas
│   │   └── preprocessing.py # Data cleaning and transformation
│   ├── models/             # Model implementations
│   │   ├── __init__.py
│   │   ├── custom_scvi.py  # Custom scVI model definition
│   │   └── metrics.py      # Model evaluation metrics
│   └── utils/              # Utility functions
│       ├── __init__.py
│       ├── plotting.py     # Visualization functions
│       └── helpers.py      # Misc helper functions
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Initial data exploration
│   ├── 02_model_training.ipynb    # Model training and hyperparameter tuning
│   └── 03_results_analysis.ipynb  # Results visualization and biological interpretation
├── results/                # Output from analysis
│   ├── figures/            # Generated plots and visualizations
│   ├── models/             # Saved model checkpoints
│   └── embeddings/         # Latent space embeddings
├── tests/                  # Unit tests for code in src/
├── configs/                # Configuration files
│   ├── model_params.yaml   # Model hyperparameters
│   └── data_config.yaml    # Data configuration
├── scripts/                # Command-line scripts
│   ├── train_model.py      # Script for model training
│   └── evaluate_model.py   # Script for model evaluation
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── requirements.txt        # Project dependencies
├── setup.py                # Make the project installable
├── README.md               # Project documentation
└── .gitignore              # Include patterns to ignore large data files
```

<a name="model-and-data"></a>
## Model and Data

### Model
This project implements and extends the scVI model from Gayoso et al. (2022), a variational autoencoder (VAE) specifically designed for single-cell RNA sequencing data. The custom implementation includes:

1. **Modified encoder architecture** with deeper neural networks
2. **Layer normalization** for improved training stability
3. **Custom regularization terms** to enhance latent space structure
4. **Hyperparameter optimization** for better cell type separation

### Data
The implementation uses single-cell RNA sequencing data from the Allen Brain Atlas, focusing on cortical neurons. The data includes cells from different brain regions and multiple donors, allowing for comprehensive analysis of neuronal subtypes and batch effect correction.

Data is automatically downloaded and processed from the Allen Brain Institute portal using the data loading utilities in `src/data/loaders.py`.

<a name="build-and-run"></a>
## Build and Run

### How to Build

This repository uses Git Large File Storage (Git LFS) to handle large model checkpoint files. To clone and use this repository:

1. Install Git LFS if you haven't already:
   ```bash
   # For Ubuntu/Debian
   apt-get install git-lfs

   # For macOS with Homebrew
   brew install git-lfs

   # For Windows with Chocolatey
   choco install git-lfs
   ```

2. Enable Git LFS:
   ```bash
   git lfs install
   ```

3. Ensure you have Python 3.8 or higher installed.
4. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scvi_project.git
   cd scvi_project
   ```

5. Pull the LFS files:
   ```bash
   git lfs pull
   ```

6. Create a conda environment:
   ```bash
   conda create -n scvi_env python=3.11
   conda activate scvi_env
   ```

7. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Install the Package

Install the package locally in your conda environment:

```bash
pip install -e .
```

### How to Run

This repository uses Pre-Commit to maintain PEP8 standards. To use Pre-Commit:

1. Activate your conda environment:
   ```bash
   conda activate scvi_env
   ```

2. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

3. Test the Pre-Commit hooks:
   ```bash
   pre-commit run --all-files
   ```

### Training a Model

To train a standard scVI model:

```bash
python scripts/train_model.py --config configs/model_params.yaml --output results/models/standard_model
```

To train a custom model with modified architecture:

```bash
python scripts/train_model.py --config configs/model_params.yaml --model custom --output results/models/custom_model
```

### Jupyter Notebooks

Interactive analysis can be performed using the provided notebooks:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

<a name="results"></a>
## Results

<!-- The implemented custom scVI model demonstrates several improvements over the standard implementation:

1. **Better cell type separation** as measured by silhouette scores
2. **Improved batch effect correction** across different donors
3. **More biologically interpretable latent factors**
4. **Enhanced rare cell type identification**

Detailed analysis results are available in the notebooks and the `results/` directory. -->

<a name="license"></a>
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Gayoso, A., Lopez, R., Xing, G. et al. A Python library for probabilistic analysis of single-cell omics data. Nat Biotechnol 40, 911–920 (2022).
- Allen Brain Map: https://portal.brain-map.org/