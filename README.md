# Regime-Dependent-Behavior-of-Random-Forest-Hyperparameters-under-Correlated-Predictors
This contains the simulation code and experimental framework used for this article for the sake of reproducibility.

---

## 📌 Overview

This project investigates how random forest hyperparameters—particularly feature subsampling (`mtry`)—scale with **effective dimensionality** rather than ambient dimension.

The central finding is that:

> Hyperparameter behavior is governed by the ratio ( n / p_{\text{eff}} ), with regime-dependent effects driven by predictor correlation.

---

## ⚙️ Features

* Simulation framework for correlated predictors
* Multiple covariance structures:

  * **Equicorrelation (compound symmetry)**
  * **AR(1) (decaying correlation)**
* Effective dimensionality computation via eigenvalues
* Random forest hyperparameter tuning (grid-based)
* Robustness checks across:

  * different grids
  * different dependence structures

---

## 📊 Simulation Design

Each simulation run corresponds to a combination of:

* Sample size ( n )
* Number of predictors ( p )
* Correlation level ( \rho )
* Covariance structure
* Replicate index

The framework supports both:

* **Full grid experiments** (main results)
* **Partial grid robustness checks** (e.g., AR(1))

---

## 🚀 Usage

### Example: Equicorrelation (main experiment)

```bash
python rf_simulation1.py \
  --n-values 100,300,500,700,900 \
  --p-values 5,10,20 \
  --rho-values 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --reps 20 \
  --mtry-grid 1,2,3,4,5 \
  --ntree-grid 100,200,300,400,500,600 \
  --n-resamples 20 \
  --output-dir results_full \
  --resume \
  --n-jobs 8
```

---

### Example: AR(1) robustness

```bash
python rf_simulation1.py \
  --n-values 100,300,500 \
  --p-values 5,10,20 \
  --rho-values 0.0,0.1,...,0.9 \
  --cov-structure ar1 \
  --output-dir results_ar1 \
  --resume
```

---

## 📦 Output

Each run produces a dataset containing:

* Optimal `mtry`,'ntree'
* Performance metrics (e.g., AUC or accuracy)
* Confidence interval width (stability measure)
* Derived quantities:

  * ( $p_{\text{eff}}$ )
  * ( $\log(n/p_{\text{eff}})$ )
  * normalized ( mtry )

---
🧠 Key Concept: Effective Dimensionality

Effective dimensionality is computed as:


$p_{\text{eff}} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$


where $( \lambda_i )$ are eigenvalues of the predictor covariance matrix.

📁 Repository Structure
.
├── rf_simulation1.py     # main simulation script
├── selected_results   # output dataset for main
├── selected_result_ar1 #output dataset for ar1
├── Main Results Analysis #analysis of selected_results
├── AR1RobustnessCheck #analysis of ar1_results
├── README.md             # this file
🔁 Reproducibility
All experiments are controlled via command-line arguments
Simulations can be resumed using --resume
Parallel execution supported via --n-jobs

