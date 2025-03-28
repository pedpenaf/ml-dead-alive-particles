# Voronoi-Based Machine Learning Classifier

## Overview

This project provides a pipeline for extracting Voronoi features from particle simulations and using them to train and analyze machine learning models. The workflow consists of multiple scripts for data extraction, model training, evaluation, and comparison.

## Installation

### Prerequisites

- Python 3.x
- The libraries required can be installed with the requeriments.txt file

## Scripts and Usage

You may have the traj\_phia datasets in folders named `phia\<density>` in the same folder in which you clone this repository and the name of each dataset may be `particles-features-\<density>-Fa\<fa>.txt`, otherwise the voronoi.py script won't be able to read it properly. If you have an .atom dataset with non-float parameters between images you can convert it to a proper .txt file using atom-to-txt.ipynb 

### 1. `voronoi.py`

**Purpose:** Extracts Voronoi parameters from a dataset and saves them for further analysis.

**Usage:**

```bash
python voronoi.py <density> <fa>
```

- **Input:** Reads `phia<density>/traj_phia<density>-T05-Fa<fa>-tau1.dat`, which contains particle ID, type, and x,y coordinates, for more info about the input files read info_dead_or_alive.txt.
- **Output:** Saves extracted features to `phia<density>/particles-features-<density>-Fa<fa>.txt`.

### 2. `binary-classification.py`

**Purpose:** Trains a gradient boosting classifier (LightGBM) on the Voronoi features.

**Usage:**

```bash
python binary-classification.py <n_cores> <density> <fa>
```

- **Input:** Reads `phia<density>/particles-features-<density>-Fa<fa>.txt`.
- **Output:**
  - Displays accuracy and confusion matrix.
  - Saves a feature importance plot as `phia<density>/importance-3D-voronoi-phi<density>-Fa<fa>.png`.

### 3. `export-model.py`

**Purpose:** Similar to `binary-classification.py`, but also saves the trained model and dataset for future analysis.

**Usage:**

```bash
python export-model.py <n_cores> <density> <fa>
```

- **Output:** Saves the trained model and relevant data to `phia<density>/gb-model-<density>-Fa<fa>.pkl`.

### 4. `plot-score-Fa-depend.py`

**Purpose:** Analyzes the performance of trained models across different values of `fa`.

**Usage:**

```bash
python plot-score-Fa-depend.py <density> <fa_list>
```

- **Input:** Reads model files generated by `export-model.py`.
- **Output:** Plots accuracy, ROC AUC score, and F1 score for different `fa` values.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Pedro Peñafiel Bonilla

binary-classification.py script and training data provided by Giulia Janzen

