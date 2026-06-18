"""Train random forest with nested cross-validation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


def mtry_grid(p):
    """Candidate max_features values: log10(p), sqrt(p), p/5, p/2.

    Parameters
    ----------
    p : int
        Number of features.

    Returns
    -------
    list of int
        Sorted unique candidate values, each >= 1.
    """
    return sorted({
        max(1, round(np.log10(p))),
        max(1, round(np.sqrt(p))),
        max(1, round(p / 5)),
        max(1, round(p / 2)),
    })


def nested_cv(X, y, config, rng):
    """Stratified nested CV with OOB MCC hyperparameter tuning.

    Parameters
    ----------
    X : DataFrame
        Feature matrix, n samples x p features.
    y : Series
        Integer-coded binary labels.
    config : dict
        Keys ``cv.n_folds``, ``rf.n_trees``, ``rf.min_samples_leaf``.
    rng : numpy.random.Generator

    Returns
    -------
    predictions : ndarray of int, shape (n,)
    mcc : float
        Matthews correlation coefficient on held-out folds.
    fold_params : list of (min_samples_leaf, max_features)
    """
    n_folds = config["cv"]["n_folds"]
    n_trees = config["rf"]["n_trees"]
    msl_grid = config["rf"]["min_samples_leaf"]
    mf_grid = mtry_grid(X.shape[1])

    param_grid = [(msl, mf) for msl in msl_grid for mf in mf_grid]

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=int(rng.integers(1, 2**31)),
    )

    X_arr, y_arr = X.values, y.values
    predictions = np.empty(len(y_arr), dtype=int)
    fold_params = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]

        best_mcc, best_model, best_p = -np.inf, None, param_grid[0]
        for msl, mf in param_grid:
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_features=mf,
                min_samples_leaf=msl,
                oob_score=True,
                n_jobs=-1,
                random_state=int(rng.integers(1, 2**31)),
            )
            rf.fit(X_train, y_train)
            oob_pred = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
            mcc = matthews_corrcoef(y_train, oob_pred)
            if mcc > best_mcc:
                best_mcc, best_model, best_p = mcc, rf, (msl, mf)

        predictions[test_idx] = best_model.predict(X_arr[test_idx])
        fold_params.append(best_p)

    return predictions, matthews_corrcoef(y_arr, predictions), fold_params


def fit_final(X, y, config, rng):
    """Train RF on all data, selecting hyperparameters by OOB MCC.

    Parameters
    ----------
    X : DataFrame
    y : Series
    config : dict
    rng : numpy.random.Generator

    Returns
    -------
    model : RandomForestClassifier
    best_params : tuple of (min_samples_leaf, max_features)
    """
    msl_grid = config["rf"]["min_samples_leaf"]
    mf_grid = mtry_grid(X.shape[1])
    X_arr, y_arr = X.values, y.values

    best_mcc, best_model, best_params = -np.inf, None, None
    for msl in msl_grid:
        for mf in mf_grid:
            rf = RandomForestClassifier(
                n_estimators=config["rf"]["n_trees"],
                max_features=mf,
                min_samples_leaf=msl,
                oob_score=True,
                n_jobs=-1,
                random_state=int(rng.integers(1, 2**31)),
            )
            rf.fit(X_arr, y_arr)
            oob_pred = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
            mcc = matthews_corrcoef(y_arr, oob_pred)
            if mcc > best_mcc:
                best_mcc, best_model, best_params = mcc, rf, (msl, mf)

    return best_model, best_params
