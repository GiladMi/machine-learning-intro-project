# Machine Learning Intro Project

Final project for an introductory AI course. It compares classical machine learning models with neural networks (PyTorch) on one regression task and one image classification task, with all hyperparameters chosen on a validation set and final results reported on a held-out test set.

![PyTorch regression: predicted vs. true MPG](plots/pytorch_regression_pred_vs_truth.png)

For the full write-up, [the final report is available](assets/ml_intro_project.pdf).

## Tasks and Results

### 1. Regression: Auto MPG ([UCI](https://archive.ics.uci.edu/dataset/9/auto+mpg))

Predicting fuel efficiency (`mpg`) from 7 features. 6 rows with a missing `horsepower` were dropped (398 → 392 records), features were standardized, and the data was split 60/20/20 into train (234), validation (79) and test (79) sets.

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Linear regression (best polynomial degree = 1) | 3.26 | 2.41 | 0.792 |
| KNN regression (k = 39) | 3.23 | 2.38 | 0.796 |
| Neural network (PyTorch, fully connected) | **2.57** | **1.94** | **0.871** |

- **Bias–variance:** polynomial regression of degree 1–5 showed clear overfitting by degree 4 (training MSE near 0, validation MSE up by orders of magnitude), so the linear model was kept as the classical baseline.
- **Optimization:** batch vs. mini-batch (32) gradient descent for linear regression; mini-batch converged in far fewer epochs.
- **Neural network:** grid search over architecture, optimizer (SGD/Adam), learning rate, dropout and weight decay, selected by validation MSE.

### 2. Classification: CIFAR-10

Classical models on raw pixels (3,072 features), using a 3,000-image training and a 1,000-image validation subset for speed, evaluated on the full 10,000-image test set.

| Model | Tuned hyperparameter | Validation accuracy |
|---|---|---|
| Softmax regression | C = 10⁻³ | 35.2% |
| Linear SVM | C = 10⁻⁴ | **35.9%** (test: 36.2%) |
| KNN | k = 31 | 27.8% |

A convolutional neural network (PyTorch) selected via hyperparameter search reaches **83.7–86.7%** test accuracy<sup>1</sup>, far above the classical models, which cannot learn spatial structure from flattened pixels. KNN also suffers from the curse of dimensionality in 3,072-dimensional space.

<sup>1</sup> Neural-network training is not fully deterministic across environments (see the footnote in the report). The report lists 86.7%; the committed notebook output shows 83.7%.

## Repository Structure

| Path | Contents |
|---|---|
| `regression_analysis.py` | Auto MPG: EDA, linear / polynomial / KNN regression, gradient descent comparison |
| `cifar10_classification.py` | CIFAR-10: softmax regression, linear SVM and KNN with hyperparameter tuning |
| `PyTorchNN.ipynb` | PyTorch classification (barebones → `nn.Module` → `nn.Sequential`, FC vs. CNN search) |
| `PyTorchNN_Regression.ipynb` | PyTorch regression on Auto MPG and comparison with the classical baselines |
| `results/`, `plots/` | Saved metrics and figures |

## Build Instructions

### Prerequisites

- [**Python**](https://www.python.org/) 3.12 or 3.13
- [**uv**](https://docs.astral.sh/uv/) (recommended)

### Setup and Execution

1. Clone and navigate to the repository.
2. Run `uv sync` to install the dependencies. `pyproject.toml` points PyTorch at the ROCm wheel index; on NVIDIA or CPU-only machines, change the `pytorch-rocm` index URL to the matching one from [pytorch.org](https://pytorch.org/get-started/locally/).
3. Run `python regression_analysis.py` (the Auto MPG data is downloaded automatically via `ucimlrepo`).
4. Run the `PyTorchNN.ipynb` notebook once to download CIFAR-10 into `intro_to_ai/datasets/`, then run `python cifar10_classification.py`.
5. Run `PyTorchNN_Regression.ipynb` for the neural network regression comparison.
6. Results are written to `results/` and plots to `plots/`.
