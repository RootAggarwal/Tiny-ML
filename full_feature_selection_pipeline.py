import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    chi2
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("bank-full.csv", sep=";")

df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})

df = df.drop(columns=["duration"])

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

def encode_data(X_train_fold, X_valid_fold):

    X_train_fold = pd.get_dummies(
        X_train_fold,
        drop_first=True
    )

    X_valid_fold = pd.get_dummies(
        X_valid_fold,
        drop_first=True
    )

    X_train_fold, X_valid_fold = (
        X_train_fold.align(
            X_valid_fold,
            join="left",
            axis=1,
            fill_value=0
        )
    )

    return X_train_fold, X_valid_fold

def rank_features(X, y):

    feature_score = defaultdict(float)

    # Pearson
    pearson = (
        X.corrwith(y)
        .abs()
        .sort_values(ascending=False)
    )

    # Spearman
    spearman = pd.Series({
        col: abs(
            X[col].corr(
                y,
                method="spearman"
            )
        )
        for col in X.columns
    }).sort_values(ascending=False)

    # ANOVA
    f_scores, _ = f_classif(X, y)

    anova = pd.Series(
        f_scores,
        index=X.columns
    ).sort_values(ascending=False)

    # Mutual Information
    mi = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    mi = pd.Series(
        mi,
        index=X.columns
    ).sort_values(ascending=False)

    # Chi-square
    X_chi = X.copy()

    for col in X_chi.columns:

        if X_chi[col].min() < 0:

            X_chi[col] = (
                X_chi[col]
                - X_chi[col].min()
            )

    chi_scores, _ = chi2(
        X_chi,
        y
    )

    chi = pd.Series(
        chi_scores,
        index=X.columns
    ).sort_values(ascending=False)

    methods = [
        pearson,
        spearman,
        anova,
        mi,
        chi
    ]

    for ranking in methods:

        for rank, feat in enumerate(
            ranking.index,
            start=1
        ):

            feature_score[feat] += (
                1 / rank
            )

    return (
        pd.Series(feature_score)
        .sort_values(
            ascending=False
        )
    )

def evaluate_top_k(
    X,
    y,
    top_k
):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_idx, valid_idx in skf.split(X, y):

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]

        y_tr = y.iloc[train_idx]
        y_val = y.iloc[valid_idx]

        X_tr, X_val = encode_data(
            X_tr,
            X_val
        )

        ranking = rank_features(
            X_tr,
            y_tr
        )

        selected = (
            ranking
            .head(top_k)
            .index
            .tolist()
        )

        X_tr = X_tr[selected]
        X_val = X_val[selected]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )

        model.fit(
            X_tr,
            y_tr
        )

        pred = model.predict_proba(
            X_val
        )[:, 1]

        auc = roc_auc_score(
            y_val,
            pred
        )

        scores.append(auc)

    return np.mean(scores)

candidate_k = [
    5,
    10,
    15,
    20,
    25,
    30,
    40
]

results = {}

for k in candidate_k:

    score = evaluate_top_k(
        X_train,
        y_train,
        k
    )

    results[k] = score

    print(
        f"K={k} "
        f"AUC={score:.4f}"
    )

best_k = max(
    results,
    key=results.get
)

print(
    "Best K:",
    best_k
)

X_train_enc = pd.get_dummies(
    X_train,
    drop_first=True
)

ranking = rank_features(
    X_train_enc,
    y_train
)

final_features = (
    ranking
    .head(best_k)
    .index
    .tolist()
)

print(final_features)

X_test_enc = pd.get_dummies(
    X_test,
    drop_first=True
)

X_train_enc, X_test_enc = (
    X_train_enc.align(
        X_test_enc,
        join="left",
        axis=1,
        fill_value=0
    )
)

X_train_final = (
    X_train_enc[
        final_features
    ]
)

X_test_final = (
    X_test_enc[
        final_features
    ]
)

final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

final_model.fit(
    X_train_final,
    y_train
)

test_pred = (
    final_model
    .predict_proba(
        X_test_final
    )[:, 1]
)

test_auc = roc_auc_score(
    y_test,
    test_pred
)

print(
    "Final Test AUC:",
    test_auc
)

from sklearn.inspection import permutation_importance

perm = permutation_importance(
    final_model,
    X_test_final,
    y_test,
    n_repeats=20,
    random_state=42
)

importance = pd.Series(
    perm.importances_mean,
    index=final_features
)

print(
    importance.sort_values(
        ascending=False
    )
)
