import xgboost as xgb
import optuna
from train_BDT import load_ds, AMS
import os
import yaml
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import argparse

le = LabelEncoder()

def fmt(x):
        return f"{x:>10.4f}" if np.isfinite(x) else f"{'nan':>10}"


def log_trial_result(study, trial):
    """Print diagnostic information to monitor progress live."""
    value = "None" if trial.value is None else f"{trial.value:.6f}"
    best_value = "None" if study.best_trial.value is None else f"{study.best_trial.value:.6f}"
    ams_ggH_even = trial.user_attrs.get("ams_ggH_even", float("nan"))
    ams_qqH_even = trial.user_attrs.get("ams_qqH_even", float("nan"))
    ams_ggH_odd = trial.user_attrs.get("ams_ggH_odd", float("nan"))
    ams_qqH_odd = trial.user_attrs.get("ams_qqH_odd", float("nan"))
    ams_even = trial.user_attrs.get("ams_even", float("nan"))
    ams_odd = trial.user_attrs.get("ams_odd", float("nan"))
    vetoed = trial.user_attrs.get("vetoed", False)

    sep = "+--------------+------------+------------+"
    print(
        f"\n{'=' * 80}\n"
        f"[Trial {trial.number:04d}] value={value} best={best_value} vetoed={vetoed}\n"
        f"{sep}\n"
        f"| metric       |       EVEN |        ODD |\n"
        f"{sep}\n"
        f"| ams          | {fmt(ams_even)} | {fmt(ams_odd)} |\n"
        f"| ams_ggH      | {fmt(ams_ggH_even)} | {fmt(ams_ggH_odd)} |\n"
        f"| ams_qqH      | {fmt(ams_qqH_even)} | {fmt(ams_qqH_odd)} |\n"
        f"{sep}\n"
        f"{'=' * 80}\n",
        flush=True,
    )

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for XGBoost")
    parser.add_argument('--channel', type=str, help="Channel to optimise", required=True)
    parser.add_argument('--n_trials', type=int, help="Number of trials to attempt")
    parser.add_argument('--n_jobs', type=int, help="Number of cores to use")
    parser.add_argument('--study_name', type=str, help="Name of study (can use to resume)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()


def validation(model, x, y, w_NN, w_phys, parity):
    # Get predictions
    y_pred_proba = model.predict_proba(x) # raw score
    y_pred = y_pred_proba.argmax(axis=1) # predicted label
    y_pred = le.inverse_transform(y_pred) # convert back to original labels (0=genuine, 11=ggH, 12=qqH, 2=fakes)
    # Find events classified as Higgs and get their raw scores, weights and truth labels
    pred_ggH_mask = (y_pred == 11)
    pred_qqH_mask = (y_pred == 12)
    y_pred_ggH = y_pred_proba[:, 2][pred_ggH_mask]
    y_pred_qqH = y_pred_proba[:, 3][pred_qqH_mask]
    w_pred_ggH = w_phys[pred_ggH_mask]
    w_pred_qqH = w_phys[pred_qqH_mask]
    y_ggH = y[pred_ggH_mask]
    y_qqH = y[pred_qqH_mask]
    # Optimised binning (flat signal)
    true_ggH_mask = (y_ggH == 11)
    qqH_but_pred_ggH_mask = (y_ggH == 12)
    true_qqH_mask = (y_qqH == 12)
    ggH_but_pred_qqH_mask = (y_qqH == 11)
    n_bins = 5
    w_perc_ggH = DescrStatsW(y_pred_ggH[true_ggH_mask], weights=w_pred_ggH[true_ggH_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    w_perc_qqH = DescrStatsW(y_pred_qqH[true_qqH_mask], weights=w_pred_qqH[true_qqH_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins_ggH = np.concatenate([[0.25], np.array(w_perc_ggH), [1]])
    bins_qqH = np.concatenate([[0.25], np.array(w_perc_qqH), [1]])
    # Histogram signal and background counts in each bin
    s_counts_ggH = np.histogram(y_pred_ggH[true_ggH_mask], bins=bins_ggH, weights=w_pred_ggH[true_ggH_mask])[0]
    s_counts_qqH = np.histogram(y_pred_qqH[true_qqH_mask], bins=bins_qqH, weights=w_pred_qqH[true_qqH_mask])[0]
    # To prioritise ggH vs qqH separation, we can consider the qqH events classified as ggH as background for the ggH AMS, and vice versa for the qqH AMS
    bkg_counts_ggH = np.histogram(y_pred_ggH[qqH_but_pred_ggH_mask], bins=bins_ggH, weights=w_pred_ggH[qqH_but_pred_ggH_mask])[0]
    bkg_counts_qqH = np.histogram(y_pred_qqH[ggH_but_pred_qqH_mask], bins=bins_qqH, weights=w_pred_qqH[ggH_but_pred_qqH_mask])[0]
    # "AMS" Scores
    if np.any(bkg_counts_ggH == 0):
        print(f"bkg_counts_ggH: {bkg_counts_ggH}")
    ams_bybin_ggH = AMS(s_counts_ggH, bkg_counts_ggH, b0=0)
    ams_bybin_qqH = AMS(s_counts_qqH, bkg_counts_qqH)
    ams_ggH = np.sqrt(np.sum(ams_bybin_ggH**2))
    ams_qqH = np.sqrt(np.sum(ams_bybin_qqH**2))
    
    # Total AMS (using all Higgs as signal and all background as background)
    pred_higgs_mask = (y_pred == 11) | (y_pred == 12)
    y_pred_higgs = y_pred_proba[:, 2][pred_higgs_mask] + y_pred_proba[:, 3][pred_higgs_mask] # sum of ggH and qqH scores
    w_pred_higgs = w_phys[pred_higgs_mask]
    y_higgs = y[pred_higgs_mask]
    true_higgs_mask = (y_higgs == 11) | (y_higgs == 12)
    w_perc_higgs = DescrStatsW(y_pred_higgs[true_higgs_mask], weights=w_pred_higgs[true_higgs_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins_higgs = np.concatenate([[0.25], np.array(w_perc_higgs), [1]])
    s_counts_higgs = np.histogram(y_pred_higgs[true_higgs_mask], bins=bins_higgs, weights=w_pred_higgs[true_higgs_mask])[0]
    bkg_counts_higgs = np.histogram(y_pred_higgs[~true_higgs_mask], bins=bins_higgs, weights=w_pred_higgs[~true_higgs_mask])[0]
    ams_bybin_higgs = AMS(s_counts_higgs, bkg_counts_higgs)
    ams = np.sqrt(np.sum(ams_bybin_higgs**2))
    return ams, ams_ggH, ams_qqH


def train_model(x_train, y_train, w_train, parity, param):
    # Model training
    model = XGBClassifier(**param)

    model.fit(x_train, y_train, sample_weight=w_train)

    return model


def obj(ams, ams_ggH, ams_qqH):
    """
    Objective function to maximise
    """
    alpha = 0.5 # hyper-hyperparameter to tune balance between total AMS and ggH/qqH AMS
    A, G, Q = 3, 16, 4 # put in by hand to make a,g,q on the same scale
    a = ams / A
    g = ams_ggH / G
    q = ams_qqH / Q
    return (a + g + q)/3 - alpha*np.std([a, g, q])


def objective(trial):

    # Optimise the sum of the two obj. scores

    param = {
        "verbosity": 0,
        'objective': 'multi:softmax',
        # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]), # booster: can add , "gblinear"
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000), # n estimators
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True), # L2 regularization weight.
        # "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True), # L1 regularization weight.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0), # sampling
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.7),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10)
    }

    if args.gpu:
        # enable gpu training
        param['device'] = 'cuda'

        # Train even model
        model_even = train_model(x_train_gpu_EVEN, y_train_gpu_EVEN, w_train_gpu_EVEN, "EVEN", param)
        ams_even, ams_ggH_even, ams_qqH_even = validation(model_even, x_val_gpu_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_gpu_ODD, y_train_gpu_ODD, w_train_gpu_ODD, "ODD", param)
        ams_odd, ams_ggH_odd, ams_qqH_odd = validation(model_odd, x_val_gpu_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")

    else:
        # Train even model
        model_even = train_model(x_train_EVEN, y_train_EVEN, w_train_EVEN, "EVEN", param)
        ams_even, ams_ggH_even, ams_qqH_even = validation(model_even, x_val_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_ODD, y_train_ODD, w_train_ODD, "ODD", param)
        ams_odd, ams_ggH_odd, ams_qqH_odd = validation(model_odd, x_val_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")

    trial.set_user_attr("ams_even", float(ams_even))
    trial.set_user_attr("ams_odd", float(ams_odd))
    trial.set_user_attr("ams_ggH_even", float(ams_ggH_even))
    trial.set_user_attr("ams_qqH_even", float(ams_qqH_even))
    trial.set_user_attr("ams_ggH_odd", float(ams_ggH_odd))
    trial.set_user_attr("ams_qqH_odd", float(ams_qqH_odd))

    try:
        obj_even = obj(ams_even, ams_ggH_even, ams_qqH_even)
        obj_odd = obj(ams_odd, ams_ggH_odd, ams_qqH_odd)
    except Exception as e:
        print(f"Error calculating objective: {e}")
        trial.set_user_attr("vetoed", True)
        return 0 # effectively veto this model

    if abs(ams_even - ams_odd)/(ams_even + ams_odd) > 0.04: # allow a 4% difference in total AMS
        trial.set_user_attr("vetoed", True)
        return 0 # effectvely veto this model
    elif abs(ams_ggH_even - ams_ggH_odd)/(ams_ggH_even + ams_ggH_odd) > 0.08: # allow an 8% difference in ggH AMS
        trial.set_user_attr("vetoed", True)
        return 0 # effectvely veto this model
    elif abs(ams_qqH_even - ams_qqH_odd)/(ams_qqH_even + ams_qqH_odd) > 0.08: # allow an 8% difference in qqH AMS
        trial.set_user_attr("vetoed", True)
        return 0 # effectvely veto this model
    else:
        trial.set_user_attr("vetoed", False)
        return obj_even + obj_odd - abs(obj_even - obj_odd)


def main():


    print(f"Optimizing hyperparameters for XGBoost model with {args.n_trials} trials")

    if args.n_trials is not None:

        # Optuna study to optimize hyperparameters
        study = optuna.create_study(direction="maximize", study_name=args.study_name,
                                storage=f"sqlite:///hyperlogs/{args.study_name}.db?timeout=10000", load_if_exists=True)
        # Begin search
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            callbacks=[log_trial_result],
        )

        # Summary
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == "__main__":
    # Configuration of tuning via args
    args = get_args()
    if args.gpu:
        import cupy as cp
    if args.n_jobs is None:
        args.n_jobs = -1

    print("Loading config and datasets")
    # Find correct dataset to use and load config
    if args.channel == 'tt':
        cfg = yaml.safe_load(open("../config/tt/BDTHyperOpt_config.yaml"))
        data_path = 'input_path'
    elif args.channel == 'mt':
        cfg = yaml.safe_load(open("../config/mt/BDTHyperOpt_config.yaml"))
        data_path = 'input_path'
    elif args.channel == 'et':
        cfg = yaml.safe_load(open("../config/et/BDTHyperOpt_config.yaml"))
        data_path = 'input_path'

    # Load datasets
    # EVEN MODEL
    x_train_EVEN, y_train_EVEN, w_train_EVEN = load_ds(os.path.join(cfg['Setup'][data_path], f'ShuffleMerge_EVENmodel_TRAIN.parquet'),
                                                       cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'])
    x_val_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN = load_ds(os.path.join(cfg['Setup'][data_path], 'ShuffleMerge_EVENmodel_VAL.parquet'),
                                                                    cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    y_train_EVEN = le.fit_transform(y_train_EVEN)
    # ODD MODEL
    x_train_ODD, y_train_ODD, w_train_ODD = load_ds(os.path.join(cfg['Setup'][data_path], f'ShuffleMerge_ODDmodel_TRAIN.parquet'),
                                                       cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'])
    x_val_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD = load_ds(os.path.join(cfg['Setup'][data_path], 'ShuffleMerge_ODDmodel_VAL.parquet'),
                                                                    cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    y_train_ODD = le.transform(y_train_ODD)

    if args.gpu:
        print(f"Storing datasets on GPU")
        # Store datasets on gpu
        x_train_gpu_EVEN = cp.array(x_train_EVEN)
        y_train_gpu_EVEN = cp.array(y_train_EVEN)
        w_train_gpu_EVEN = cp.array(w_train_EVEN)
        x_train_gpu_ODD = cp.array(x_train_ODD)
        y_train_gpu_ODD = cp.array(y_train_ODD)
        w_train_gpu_ODD = cp.array(w_train_ODD)
        x_val_gpu_EVEN = cp.array(x_val_EVEN)
        x_val_gpu_ODD = cp.array(x_val_ODD)
        del x_train_EVEN, y_train_EVEN, w_train_EVEN, x_train_ODD, y_train_ODD, w_train_ODD, x_val_EVEN, x_val_ODD





    main()

