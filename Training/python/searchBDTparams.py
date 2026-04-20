import xgboost as xgb
import optuna
from train_BDT import load_ds, AMS
import os
import yaml
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import argparse
le = LabelEncoder()

def log_trial_result(study, trial):
    """Print one concise line for each completed trial to monitor progress live."""
    value = "None" if trial.value is None else f"{trial.value:.6f}"
    best_value = "None" if study.best_trial.value is None else f"{study.best_trial.value:.6f}"
    ams_even = trial.user_attrs.get("ams_even", float("nan"))
    ams_odd = trial.user_attrs.get("ams_odd", float("nan"))
    vetoed = trial.user_attrs.get("vetoed", False)
    print(
        f"[Trial {trial.number:04d}] value={value} "
        f"(ams_even={ams_even:.6f}, ams_odd={ams_odd:.6f}, vetoed={vetoed}) "
        f"best={best_value}",
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
    higgs_mask = (y_pred == 11) | (y_pred == 12)
    y_pred_higgs = y_pred_proba[:, 2][higgs_mask] + y_pred_proba[:, 3][higgs_mask] # combining ggH and qqH scores
    w_pred_higgs = w_phys[higgs_mask]
    y_higgs = y[higgs_mask]
    # Optimised binning (flat signal)
    true_higgs_mask = (y_higgs == 11) | (y_higgs == 12)
    n_bins = 5
    if np.count_nonzero(true_higgs_mask) < n_bins:
        return -9999 # not enough signal events to make a meaningful score
    else:
        w_perc = DescrStatsW(y_pred_higgs[true_higgs_mask], weights=w_pred_higgs[true_higgs_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
        bins = np.concatenate([[0.25], np.array(w_perc), [1]])
        # Histogram signal and background counts in each bin
        s_counts = np.histogram(y_pred_higgs[true_higgs_mask], bins=bins, weights=w_pred_higgs[true_higgs_mask])[0]
        bkg_counts = np.histogram(y_pred_higgs[~true_higgs_mask], bins=bins, weights=w_pred_higgs[~true_higgs_mask])[0]
        # AMS Score
        ams_bybin = AMS(s_counts, bkg_counts)
        ams = np.sqrt(np.sum(ams_bybin**2))
        # print("AMS Score (bin by bin):", ams)
        print(f"AMS {parity}: {ams}")
        return ams


def train_model(x_train, y_train, w_train, parity, param):
    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**param)

    model.fit(x_train, y_train, sample_weight=w_train)

    return model


def objective(trial):
    """
    Objective function for Optuna to optimize. Trains two XGBoost models (even
    and odd) with the same hyperparameters and evaluates their AMS scores on the
    validation set. The final score is a combination of the two AMS scores,
    penalized if they differ significantly to encourage balanced performance
    between the two models.

    In validation()'s current form, this AMS score purely quantifies Higgs
    (ggH+qqH) vs background separation, without distinguishing between ggH and
    qqH. It may be worth investigating the effect of using a more complex score
    that also takes into account the separation between ggH and qqH.
    """

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
        ams_even = validation(model_even, x_val_gpu_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_gpu_ODD, y_train_gpu_ODD, w_train_gpu_ODD, "ODD", param)
        ams_odd = validation(model_odd, x_val_gpu_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")

    else:
        # Train even model
        model_even = train_model(x_train_EVEN, y_train_EVEN, w_train_EVEN, "EVEN", param)
        ams_even = validation(model_even, x_val_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_ODD, y_train_ODD, w_train_ODD, "ODD", param)
        ams_odd = validation(model_odd, x_val_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")

    trial.set_user_attr("ams_even", float(ams_even))
    trial.set_user_attr("ams_odd", float(ams_odd))

    if abs(ams_even - ams_odd)/(ams_even + ams_odd) > 0.05: # allow a 5% difference in total AMS ~ 10% in between the two
        trial.set_user_attr("vetoed", True)
        return 0 # effectvely veto this model
    else:
        trial.set_user_attr("vetoed", False)
        return ams_even + ams_odd - abs(ams_even - ams_odd)


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

