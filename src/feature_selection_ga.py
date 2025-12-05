# src/feature_selection_ga.py

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline + GA feature selection")
    parser.add_argument("--features_parquet", type=str, required=True,
                        help="Input Parquet file with image_id, label, and feature columns.")
    parser.add_argument("--train_output", type=str, required=True,
                        help="Output Parquet path for train split.")
    parser.add_argument("--test_output", type=str, required=True,
                        help="Output Parquet path for test split.")
    parser.add_argument("--baseline_metrics", type=str, required=True,
                        help="Output JSON file path for baseline metrics.")
    parser.add_argument("--ga_metrics", type=str, required=True,
                        help="Output JSON file path for GA metrics.")
    parser.add_argument("--selected_features", type=str, required=True,
                        help="Output JSON file path for selected feature names.")
    return parser.parse_args()


def baseline_feature_selection(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    vt = VarianceThreshold(threshold=0.0)
    X_train_sel = vt.fit_transform(X_train)
    X_test_sel = vt.transform(X_test)

    baseline_feature_mask = vt.get_support()
    baseline_features = [f for f, keep in zip(feature_cols, baseline_feature_mask) if keep]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        "baseline_accuracy": float(acc),
        "baseline_num_features": int(len(baseline_features)),
    }

    return baseline_features, metrics


def evaluate_mask(mask, X_train, y_train, X_val, y_val, penalty=0.001):
    if mask.sum() == 0:
        idx = np.random.randint(0, len(mask))
        mask[idx] = 1

    selected_indices = np.where(mask == 1)[0]
    X_train_sel = X_train[:, selected_indices]
    X_val_sel = X_val[:, selected_indices]

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_val_sel)
    acc = accuracy_score(y_val, y_pred)

    fitness = acc - penalty * len(selected_indices)
    return fitness, acc, len(selected_indices)


def run_ga_feature_selection(train_df, feature_cols,
                             pop_size=20, n_generations=10,
                             crossover_prob=0.8, mutation_prob=0.1,
                             penalty=0.001):

    X = train_df[feature_cols].values
    y = train_df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_features = len(feature_cols)
    population = np.random.randint(0, 2, size=(pop_size, n_features))

    def tournament_selection(pop, fitnesses, k=3):
        idxs = np.random.choice(len(pop), size=k, replace=False)
        best_idx = idxs[np.argmax(fitnesses[idxs])]
        return pop[best_idx].copy()

    best_individual = None
    best_fitness = -np.inf
    best_acc = 0.0
    best_num_features = 0

    start_time = time.time()

    for gen in range(n_generations):
        fitnesses, gen_accs, gen_nums = [], [], []

        for ind in population:
            f, acc, num = evaluate_mask(ind.copy(), X_train, y_train, X_val, y_val, penalty)
            fitnesses.append(f)
            gen_accs.append(acc)
            gen_nums.append(num)

        fitnesses = np.array(fitnesses)
        gen_accs = np.array(gen_accs)
        gen_nums = np.array(gen_nums)

        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = float(fitnesses[gen_best_idx])
            best_acc = float(gen_accs[gen_best_idx])
            best_num_features = int(gen_nums[gen_best_idx])
            best_individual = population[gen_best_idx].copy()

        print(
            f"[GA] Gen {gen+1}/{n_generations} "
            f"| best_fitness={best_fitness:.4f} "
            f"| best_acc={best_acc:.4f} "
            f"| best_num_features={best_num_features}"
        )

        newpop = [best_individual.copy()]
        while len(newpop) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)

            if np.random.rand() < crossover_prob:
                point = np.random.randint(1, n_features)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            for c in (c1, c2):
                for i in range(n_features):
                    if np.random.rand() < mutation_prob:
                        c[i] = 1 - c[i]
                newpop.append(c)
                if len(newpop) >= pop_size:
                    break

        population = np.array(newpop[:pop_size])

    selected_indices = np.where(best_individual == 1)[0]
    selected_features = [feature_cols[i] for i in selected_indices]

    ga_metrics = {
        "ga_best_fitness": float(best_fitness),
        "ga_accuracy_val": float(best_acc),
        "ga_num_features": int(best_num_features),
        "ga_runtime_seconds": float(time.time() - start_time),
        "population_size": int(pop_size),
        "n_generations": int(n_generations),
    }

    return selected_features, ga_metrics


def main():
    args = parse_args()

    features_path = Path(args.features_parquet)
    train_out_path = Path(args.train_output)
    test_out_path = Path(args.test_output)
    baseline_metrics_path = Path(args.baseline_metrics)
    ga_metrics_path = Path(args.ga_metrics)
    selected_features_path = Path(args.selected_features)

    print(f"Loading Parquet: {features_path}")
    df = pd.read_parquet(features_path)

    if "label" not in df.columns:
        raise ValueError("Parquet must include a 'label' column.")

    feature_cols = [c for c in df.columns if c not in ["image_id", "label"]]
    print(f"Detected {len(feature_cols)} feature columns.")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_df.to_parquet(train_out_path, index=False)
    test_df.to_parquet(test_out_path, index=False)

    baseline_features, baseline_metrics = baseline_feature_selection(
        train_df, test_df, feature_cols
    )
    with open(baseline_metrics_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    selected_features, ga_metrics = run_ga_feature_selection(
        train_df, feature_cols
    )
    ga_metrics["selected_num_features"] = len(selected_features)

    with open(ga_metrics_path, "w") as f:
        json.dump(ga_metrics, f, indent=2)

    with open(selected_features_path, "w") as f:
        json.dump({"selected_features": selected_features}, f, indent=2)

    print("Feature selection + GA completed successfully.")


if __name__ == "__main__":
    main()
