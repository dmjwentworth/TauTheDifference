import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import yaml
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import argparse

le = LabelEncoder()

def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Classifier Training")
    parser.add_argument('--channel', type=str, help="Channel to train", required=True)
    # parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    parser.add_argument('--config', type=str, help='Name of config file (without .yaml extension) in config/ directory', required=False)
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()


def load_ds(path, feat_names, y_name, w_name, eval = False):
    df = pd.read_parquet(path)
    if eval:
        x = df[feat_names]
        y = df[y_name]
        w = df[w_name]
        phys_w = df['weight']
        return x, y, w, phys_w
    else:
        df = df[df['weight']>0] # only positive weights can be used for training
        x = df[feat_names]
        y = df[y_name]
        w = df[w_name]
        return x, y, w
    

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams


def print_label_summary(title, labels, weights=None, classes=None):
    labels = np.asarray(labels)
    label_classes = np.asarray(classes if classes is not None else np.unique(labels))
    counts = pd.Series(labels).value_counts().reindex(label_classes, fill_value=0)

    print(title)
    for label in label_classes:
        line = f"  class {label}: count={int(counts.loc[label])}"
        if weights is not None:
            label_weight = float(np.asarray(weights)[labels == label].sum())
            line += f", weighted_sum={label_weight:.6f}"
        print(line)


def print_train_diagnostics(path, truth_col, train_weight_col):
    df = pd.read_parquet(path, columns=[truth_col, train_weight_col, 'weight'])
    filtered_df = df[df['weight'] > 0]
    classes = np.sort(df[truth_col].unique())

    print("Training label diagnostics:")
    print_label_summary("Raw training class counts:", df[truth_col].to_numpy(), classes=classes)
    print_label_summary("Post-filter class counts (weight > 0):", filtered_df[truth_col].to_numpy(), classes=classes)
    print_label_summary(
        f"Post-filter training weight sums ({train_weight_col}):",
        filtered_df[truth_col].to_numpy(),
        weights=filtered_df[train_weight_col].to_numpy(),
        classes=classes,
    )


def print_confusion_diagnostics(y_true, y_pred, weights=None):
    labels = le.classes_
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    counts_df = pd.DataFrame(counts, index=labels, columns=labels)
    print("Validation confusion matrix (counts):")
    print(counts_df)

    if weights is not None:
        weighted = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=weights)
        weighted_df = pd.DataFrame(weighted, index=labels, columns=labels)
        print("Validation confusion matrix (weighted):")
        print(weighted_df)


def get_class_index(label):
    return int(le.transform([label])[0])


def validation(model, cfg, parity, gpu=False):
    print(f'\033[1mModel performance: \033[0m')
    val_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_VAL.parquet')
    x, y, w_NN, w_phys = load_ds(val_path, cfg['Features']['train'],
                                 cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    ggH_label = 11
    qqH_label = 12
    ggH_idx = get_class_index(ggH_label)
    qqH_idx = get_class_index(qqH_label)

    if gpu:
        if cp is None:
            raise ImportError("CuPy is required for GPU validation. Install cupy or run without --gpu.")
        print(f"Using GPU for validation")
        x_gpu = cp.array(x)
        # Get predictions
        y_pred_proba = model.predict_proba(x_gpu) # raw score
    else:
        y_pred_proba = model.predict_proba(x) # raw score
    y_pred_idx = y_pred_proba.argmax(axis=1) # predicted label index
    y_pred = le.inverse_transform(y_pred_idx)
    accuracy = accuracy_score(y, y_pred, sample_weight=w_NN)
    print("Validation Accuracy:", accuracy)
    print_confusion_diagnostics(y, y_pred, weights=w_NN)

    ggH_mask = y_pred_idx == ggH_idx # boolean mask for events classified as ggH by the BDT
    qqH_mask = y_pred_idx == qqH_idx # boolean mask for events classified as qqH by the BDT

    if not np.any(ggH_mask):
        print(f"No events were classified as ggH. Unique predicted labels: {np.unique(y_pred)}")
    if not np.any(qqH_mask):
        print(f"No events were classified as qqH. Unique predicted labels: {np.unique(y_pred)}")

    # Find events classified as ggH or qqH (Higgs) and get their raw scores, weights and truth labels
    y_pred_ggH = y_pred_proba[:, ggH_idx][ggH_mask]
    y_pred_qqH = y_pred_proba[:, qqH_idx][qqH_mask]
    w_pred_ggH = w_phys[ggH_mask]
    w_pred_qqH = w_phys[qqH_mask]
    y_ggH = y[ggH_mask]
    y_qqH = y[qqH_mask]
    # Optimised binning (flat signal)
    n_bins = 5
    true_ggH_mask = y_ggH == ggH_label
    true_qqH_mask = y_qqH == qqH_label

    if np.any(true_ggH_mask):
        w_perc_ggH = DescrStatsW(y_pred_ggH[true_ggH_mask], weights=w_pred_ggH[true_ggH_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1])
        bins_ggH = np.concatenate([[0.25], np.array(w_perc_ggH), [1]])
        s_counts_ggH = np.histogram(y_pred_ggH[true_ggH_mask], bins=bins_ggH, weights=w_pred_ggH[true_ggH_mask])[0]
        bkg_counts_ggH = np.histogram(y_pred_ggH[~true_ggH_mask], bins=bins_ggH, weights=w_pred_ggH[~true_ggH_mask])[0]
        ams_ggH = AMS(s_counts_ggH, bkg_counts_ggH)
        print("ggH AMS Score (bin by bin):", ams_ggH)
        print(f"\033[1;32mAMS_ggH: {np.sqrt(np.sum(ams_ggH**2))} \033[0m")
    else:
        print("Skipping ggH AMS: no true ggH events among events classified as ggH.")

    if np.any(true_qqH_mask):
        w_perc_qqH = DescrStatsW(y_pred_qqH[true_qqH_mask], weights=w_pred_qqH[true_qqH_mask]).quantile(np.linspace(0, 1, n_bins+1)[1:-1])
        bins_qqH = np.concatenate([[0.25], np.array(w_perc_qqH), [1]])
        s_counts_qqH = np.histogram(y_pred_qqH[true_qqH_mask], bins=bins_qqH, weights=w_pred_qqH[true_qqH_mask])[0]
        bkg_counts_qqH = np.histogram(y_pred_qqH[~true_qqH_mask], bins=bins_qqH, weights=w_pred_qqH[~true_qqH_mask])[0]
        ams_qqH = AMS(s_counts_qqH, bkg_counts_qqH)
        print("qqH AMS Score (bin by bin):", ams_qqH)
        print(f"\033[1;32mAMS_qqH: {np.sqrt(np.sum(ams_qqH**2))} \033[0m")
    else:
        print("Skipping qqH AMS: no true qqH events among events classified as qqH.")
    # AUC Score
    # truth = y_higgs.replace({2:0, 0:0}) # binary Higgs vs all
    # auc = roc_auc_score(truth, y_pred_higgs, sample_weight=w_pred_higgs)
    # print("AUC Score:", auc)
    del x, y, w_NN, w_phys


def train_model(cfg, parity, gpu=False):
    # Input path (depends on even/odd)
    train_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_TRAIN.parquet')

    print_train_diagnostics(train_path, cfg['Features']['truth'], cfg['Features']['weight'])

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])
    y_train = le.fit_transform(y_train) # encode labels to integers for XGBoost
    print(f"LabelEncoder classes_: {le.classes_}")

    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**cfg['param'])

    if gpu:
        if cp is None:
            raise ImportError("CuPy is required for GPU training. Install cupy or run without --gpu.")
        print(f"Using GPU for training")
        # Store datasets on gpu
        x_gpu = cp.array(x_train)
        y_gpu = cp.array(y_train)
        w_gpu = cp.array(w_train)
        model.fit(x_gpu, y_gpu, sample_weight=w_gpu)
    else:
        model.fit(x_train, y_train, sample_weight=w_train)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_dir_name'], parity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{cfg['Setup']['model_prefix']}_{parity}.json")
    model.save_model(save_path)
    # Get Training accuracy
    if gpu:
        y_pred_labels = model.predict(x_gpu)
    else:
        y_pred_labels = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred_labels, sample_weight=w_train)
    print(f"Training Complete! (accuracy: {accuracy}) - Model saved to: {save_path}")
    # Save features used:
    with open(os.path.join(save_dir, 'train_cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    del x_train, y_train, w_train

    return model


def main():
    args = get_args()
    if args.config is None:
        print("No config file specified, using default config name 'BDTconfig.yaml'")
        args.config = 'BDTconfig'
    if args.channel == 'tt': # Use VTight VSjet
        print("Training for tt channel (VTight Vsjet cut)")
        cfg = yaml.safe_load(open(f"../config/tt/{args.config}.yaml"))
        cfg['Setup']['model_prefix'] = 'model' # begining of model json name (add parity after)
    elif args.channel == 'mt':
        print("Training for MuTau channel")
        cfg = yaml.safe_load(open(f"../config/mt/{args.config}.yaml"))
        cfg['Setup']['model_prefix'] = 'model'
    elif args.channel == 'et':
        print("Training for ETau channel")
        cfg = yaml.safe_load(open(f"../config/et/{args.config}.yaml"))
        cfg['Setup']['model_prefix'] = 'model'

    # gpu setup
    if args.gpu:
        import cupy as cp
        cfg['param']['device'] = "gpu" # Use GPU for training

    # Train the model to be applied on EVEN events
    model = train_model(cfg, 'EVEN', gpu=args.gpu)
    validation(model, cfg, 'EVEN', gpu=args.gpu)
    print('---------------------------------- \n')

    # Train the model to be applied on ODD events
    model = train_model(cfg, 'ODD', gpu=args.gpu)
    validation(model, cfg, 'ODD', gpu=args.gpu)
    print('---------------------------------- \n')


if __name__ == "__main__":
    main()
