#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import itertools
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# helpers
# =========================================================
def parse_int_list(x: str):
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def parse_float_list(x: str):
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


import numpy as np


def make_cov_matrix(
    p: int,
    rho: float,
    structure: str = "equicorr",
    block_size: int = 5,
    rho_between: float = 0.1,
) -> np.ndarray:
    """
    Construct a covariance matrix under several dependence structures.

    Parameters
    ----------
    p : int
        Number of predictors.
    rho : float
        Main correlation parameter.
        - equicorr: common off-diagonal correlation
        - ar1: lag-1 decay parameter
        - block: within-block correlation
    structure : str, default="equicorr"
        One of {"equicorr", "ar1", "block"}.
    block_size : int, default=5
        Block size for block covariance.
    rho_between : float, default=0.1
        Between-block correlation for block covariance.

    Returns
    -------
    np.ndarray
        A (p x p) covariance matrix.
    """
    if p <= 0:
        raise ValueError("p must be positive")

    if structure == "equicorr":
        sigma = np.full((p, p), fill_value=rho, dtype=np.float64)
        np.fill_diagonal(sigma, 1.0)
        return sigma

    if structure == "ar1":
        idx = np.arange(p, dtype=np.int64)
        sigma = np.power(rho, np.abs(np.subtract.outer(idx, idx)), dtype=np.float64)
        return sigma.astype(np.float64, copy=False)

    if structure == "block":
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        sigma = np.full((p, p), fill_value=rho_between, dtype=np.float64)
        np.fill_diagonal(sigma, 1.0)

        for start in range(0, p, block_size):
            end = min(start + block_size, p)
            sigma[start:end, start:end] = rho
            np.fill_diagonal(sigma[start:end, start:end], 1.0)

        return sigma

    raise ValueError(f"Unknown covariance structure: {structure}")

def compute_p_eff(Sigma: np.ndarray) -> float:
    """
    Participation-ratio effective dimension:
    p_eff = (sum lambda)^2 / sum(lambda^2)
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    return float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))

def make_beta(
    p: int,
    s: int = 5,
    beta_strength: float = 1.0,
    signal_type: str = "first",
    rng=None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    beta = np.zeros(p, dtype=float)

    if signal_type == "first":
        idx = np.arange(min(s, p))
        beta[idx] = beta_strength

    elif signal_type == "random":
        idx = rng.choice(np.arange(p), size=min(s, p), replace=False)
        beta[idx] = beta_strength

    elif signal_type == "mixed":
        idx = rng.choice(np.arange(p), size=min(s, p), replace=False)
        half = len(idx) // 2
        beta[idx[:half]] = beta_strength
        beta[idx[half:]] = beta_strength / 2.0

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    return beta
def auc_ci_stats(scores, alpha=0.05):
    """
    Empirical percentile CI for AUC scores.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean_auc = float(np.mean(scores))
    lower = float(np.quantile(scores, alpha / 2))
    upper = float(np.quantile(scores, 1 - alpha / 2))
    width = float(upper - lower)

    return mean_auc, lower, upper, width


# =========================================================
# data generation
# =========================================================
def generate_dataset(
    n: int,
    p: int,
    rho: float,
    signal_vars: int,
    beta_strength: float,
    intercept: float,
    seed: int,
    cov_structure: str,
    block_size:int=5,
    rho_between:float=.1,
):
    """
    Generate correlated Gaussian predictors and a binary response.
    """
    rng = np.random.default_rng(seed)

    Sigma = make_cov_matrix(
    p=p,
    rho=rho,
    structure=cov_structure,
    block_size=block_size,
    rho_between=rho_between,
)
    X = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=n
    )

    signal_vars = min(signal_vars, p)

    beta = np.zeros(p)
    beta[:signal_vars] = beta_strength

    eta = intercept + X @ beta
    prob = logistic(eta)
    y = rng.binomial(1, prob, size=n)

    p_eff = compute_p_eff(Sigma)

    return X, y, p_eff


# =========================================================
# hyperparameter evaluation
# =========================================================
def evaluate_hyperparams(
    X,
    y,
    mtry,
    ntree,
    n_resamples,
    test_size,
    alpha,
    seed
):
    """
    Repeated train/validation resampling for one RF hyperparameter pair.
    Returns mean AUC and CI stats.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=n_resamples,
        test_size=test_size,
        random_state=seed
    )

    auc_scores = []

    for split_id, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        rf = RandomForestClassifier(
            n_estimators=ntree,
            max_features=mtry,
            random_state=seed + split_id,
            n_jobs=1,   # avoid nested parallelism
            bootstrap=True
        )

        rf.fit(X_train, y_train)
        prob_pred = rf.predict_proba(X_valid)[:, 1]

        if len(np.unique(y_valid)) < 2:
            auc_scores.append(np.nan)
        else:
            auc_scores.append(roc_auc_score(y_valid, prob_pred))

    mean_auc, ci_lower, ci_upper, ci_width = auc_ci_stats(auc_scores, alpha=alpha)

    return {
        "mean_auc": mean_auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_width
    }


def select_best_hyperparameters(
    X,
    y,
    mtry_grid,
    ntree_grid,
    n_resamples,
    test_size,
    alpha,
    auc_tolerance,
    seed
):
    """
    Two-stage rule:
    1) keep hyperparameters within auc_tolerance of best mean AUC
    2) among them choose smallest CI width
    3) if tie, choose highest AUC
    """
    p = X.shape[1]
    valid_mtry_grid = sorted(set([m for m in mtry_grid if 1 <= m <= p]))
    if not valid_mtry_grid:
        raise ValueError(f"No valid mtry values for p={p}. Got {mtry_grid}")

    rows = []

    for mtry, ntree in itertools.product(valid_mtry_grid, ntree_grid):
        stats = evaluate_hyperparams(
            X=X,
            y=y,
            mtry=mtry,
            ntree=ntree,
            n_resamples=n_resamples,
            test_size=test_size,
            alpha=alpha,
            seed=seed + 10000 * mtry + ntree
        )

        rows.append({
            "mtry": int(mtry),
            "ntree": int(ntree),
            "mean_auc": stats["mean_auc"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "ci_width": stats["ci_width"]
        })

    grid_df = pd.DataFrame(rows)

    best_auc = grid_df["mean_auc"].max()
    candidates = grid_df.loc[grid_df["mean_auc"] >= best_auc - auc_tolerance].copy()

    candidates = candidates.sort_values(
        by=["ci_width", "mean_auc"],
        ascending=[True, False]
    )

    best_row = candidates.iloc[0]

    return best_row.to_dict(), grid_df


# =========================================================
# experiment runner
# =========================================================
def build_tasks(n_values, p_values, rho_values, reps):
    tasks = []
    task_id = 1

    for n, p, rho, rep in itertools.product(n_values, p_values, rho_values, range(1, reps + 1)):
        tasks.append({
            "task_id": task_id,
            "rep": rep,
            "n": n,
            "p": p,
            "rho": rho
        })
        task_id += 1

    return tasks

def run_one_task(task, args):
    task_id = task["task_id"]
    rep = task["rep"]
    n = task["n"]
    p = task["p"]
    rho = task["rho"]
    seed = args.base_seed + task_id
    print(f"Starting task {task['task_id']} on PID {os.getpid()}", flush=True)
    X, y, p_eff = generate_dataset(
        n=n,
        p=p,
        rho=rho,
        signal_vars=args.signal_vars,
        beta_strength=args.beta_strength,
        intercept=args.intercept,
        seed=seed,
        cov_structure=args.cov_structure,
        block_size=args.block_size,
        rho_between=args.rho_between,
    )

    best_row, grid_df = select_best_hyperparameters(
        X=X,
        y=y,
        mtry_grid=args.mtry_grid,
        ntree_grid=args.ntree_grid,
        n_resamples=args.n_resamples,
        test_size=args.test_size,
        alpha=args.alpha,
        auc_tolerance=args.auc_tolerance,
        seed=seed
    )

    selected_result = {
        "task_id": task_id,
        "rep": rep,
        "n": n,
        "p": p,
        "p_eff": p_eff,
        "corr": rho,
        "auc": best_row["mean_auc"],
        "ci_width": best_row["ci_width"],
        "ci_lower": best_row["ci_lower"],
        "ci_upper": best_row["ci_upper"],
        "selected_mtry": int(best_row["mtry"]),
        "selected_ntree": int(best_row["ntree"]),
        "class_balance": float(np.mean(y))
    }

    grid_df = grid_df.copy()
    grid_df["task_id"] = task_id
    grid_df["rep"] = rep
    grid_df["n"] = n
    grid_df["p"] = p
    grid_df["p_eff"] = p_eff
    grid_df["corr"] = rho

    return selected_result, grid_df
def append_row_to_csv(row_dict, csv_path):
    row_df = pd.DataFrame([row_dict])
    write_header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def get_completed_task_ids(results_csv):
    if not os.path.exists(results_csv):
        return set()

    df = pd.read_csv(results_csv, usecols=["task_id"])
    return set(df["task_id"].tolist())


def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)

    results_csv = os.path.join(args.output_dir, "selected_results.csv")
    grid_csv = os.path.join(args.output_dir, "grid_results.csv")

    tasks = build_tasks(
        n_values=args.n_values,
        p_values=args.p_values,
        rho_values=args.rho_values,
        reps=args.reps
    )

    completed = get_completed_task_ids(results_csv) if args.resume else set()
    tasks = [task for task in tasks if task["task_id"] not in completed]

    print(f"Tasks remaining: {len(tasks)}")

    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = [ex.submit(run_one_task, task, args) for task in tasks]

        for fut in as_completed(futures):
            selected_result, grid_df = fut.result()

            append_row_to_csv(selected_result, results_csv)

            write_header = not os.path.exists(grid_csv)
            grid_df.to_csv(grid_csv, mode="a", header=write_header, index=False)

            print(
                f"Finished task {selected_result['task_id']}: "
                f"n={selected_result['n']}, p={selected_result['p']}, corr={selected_result['corr']}"
            )

    print(f"Done. Selected results saved to: {results_csv}")
    print(f"Full tuning grid saved to: {grid_csv}")

# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-values", type=parse_int_list, required=True)
    parser.add_argument("--p-values", type=parse_int_list, required=True)
    parser.add_argument("--rho-values", type=parse_float_list, required=True)

    parser.add_argument("--reps", type=int, default=30)

    parser.add_argument("--mtry-grid", type=parse_int_list, required=True)
    parser.add_argument("--ntree-grid", type=parse_int_list, required=True)

    parser.add_argument("--signal-vars", type=int, default=5)
    parser.add_argument("--beta-strength", type=float, default=1.0)
    parser.add_argument("--intercept", type=float, default=0.0)

    parser.add_argument("--n-resamples", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auc-tolerance", type=float, default=0.01)
    parser.add_argument("--cov-structure", type=str, default="equicorr",choices=["equicorr", "ar1", "block"])
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--rho-between", type=float, default=0.1)
    parser.add_argument("--signal-type", type=str, default="first", choices=["first", "random", "mixed"])
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()

