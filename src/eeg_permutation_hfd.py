
import os
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu, kruskal, shapiro, levene
import random
import argparse

# -------- User parameters (edit) ----------
SEED = 100
K_STATES_LIST_EEG = list(range(6, 12))  # k range is 4-9 for EEG
EEG_data_dir = "data/EEG_data/HFD_PSD_4sec_75overlap_6min"
MRI_csv_path = "/data/s.dharia-ra/PEARL/final_multi_modal/data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv"
RESULTS_DIR = "loso_eeg_permutation_results"
N_PERMUTATIONS = 1000   # set to 1000 or more for final runs (keeps runtime reasonable here)
os.makedirs(RESULTS_DIR, exist_ok=True)
# ------------------------------------------

assert os.path.isdir(EEG_data_dir), f"EEG_data_dir not found: {EEG_data_dir}"
assert os.path.isfile(MRI_csv_path), f"MRI_csv_path not found: {MRI_csv_path}"

# ---------- helpers ----------
def file_to_sid(path_or_name: str, pad: int = 2) -> str:
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    if not digits:
        if base.startswith("sub-"):
            tail = ''.join(ch for ch in base.split('-', 1)[1] if ch.isdigit())
            digits = tail or "0"
        else:
            digits = "0"
    return f"sub-{str(int(digits)).zfill(pad)}"

def _get_col(df, target_name_lower: str):
    for c in df.columns:
        if c.lower() == target_name_lower:
            return c
    raise KeyError(f"Column '{target_name_lower}' not found. Available: {list(df.columns)}")

def choose_statistical_test(groups):
    """Return (test_name, p_value, stat). Fail-safe: returns p=1.0 on error or insufficient data."""
    if len(groups) < 2 or any(g.size == 0 for g in groups):
        return 'None', 1.0, 0.0
    try:
        normality_ok = all(shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
        equal_var_ok = levene(*groups)[1] > 0.05
        if normality_ok and equal_var_ok:
            if len(groups) == 2:
                stat, p_val = ttest_ind(*groups)
                return 't-test', float(p_val), stat
            else:
                stat, p_val = f_oneway(*groups)
                return 'ANOVA', float(p_val), stat
        else:
            if len(groups) == 2:
                stat, p_val = mannwhitneyu(*groups)
                return 'Mann-Whitney', float(p_val), stat
            else:
                stat, p_val = kruskal(*groups)
                return 'Kruskal-Wallis', float(p_val), stat
    except Exception:
        return 'None', 1.0, 0.0

def compute_pvalue_weights(p_values, smoothing=1e-10, power=2.0):
    p = np.array(p_values, dtype=float)
    if p.size == 0:
        return np.array([])
    smoothed = p + smoothing
    inv = (1.0 / smoothed) ** power
    s = np.sum(inv)
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(inv) / float(inv.size)
    return inv / s

def weighted_prediction(probabilities, weights):
    probs = np.array(probabilities)
    weights = np.array(weights)
    if probs.ndim == 1:
        probs = probs[:, None]
    weighted_prob = np.average(probs, axis=1, weights=weights)
    preds = (weighted_prob > 0.5).astype(int)
    return preds, weighted_prob

# ---------- I/O and feature extraction ----------
def load_mri_dataframe(mri_csv_path):
    df = pd.read_csv(mri_csv_path)
    # find name column
    try:
        names_col = _get_col(df, "names")
    except KeyError:
        for alt in ("name", "subject", "participant"):
            try:
                names_col = _get_col(df, alt)
                break
            except KeyError:
                names_col = None
        if names_col is None:
            raise
    # find group column
    group_col = None
    for cand in ("group", "risk_group", "Risk_Group", "Group"):
        try:
            group_col = _get_col(df, cand)
            break
        except KeyError:
            group_col = None
    if group_col is None:
        raise KeyError("Cannot find group column in MRI csv. Look for 'group' or similar.")
    # extract subject_id
    name_series = df[names_col].astype(str)
    subject_raw = name_series.str.extract(r'(sub-\d+)', expand=False).fillna(name_series.str.replace("_T1w", "", regex=False))
    subject_num = subject_raw.str.extract(r'(\d+)', expand=False)
    df["subject_id"] = subject_num.apply(lambda d: f"sub-{str(int(d)).zfill(2)}" if pd.notna(d) and str(d).isdigit() else np.nan)
    return df, names_col, group_col

def build_task_subject_map(mri_df, group_col, task_tuple):
    task_set = set(task_tuple)
    df_task = mri_df[mri_df[group_col].isin(task_set)].copy()
    if df_task.empty:
        raise ValueError(f"No MRI subjects found for task {task_tuple}")
    # mapping: first element -> 0, rest -> 1 (for combined-case)
    first = task_tuple[0]
    mapping = {g: (0 if g == first else 1) for g in task_tuple}
    df_task["numeric_label"] = df_task[group_col].map(mapping)
    subj_ids = df_task["subject_id"].dropna().unique().tolist()
    return {sid: int(df_task[df_task["subject_id"] == sid]["numeric_label"].iloc[0]) for sid in subj_ids}, df_task

def load_eeg_features_for_map(feature_dir: str, risk_group_map: dict):
    all_features, all_labels, all_sub_ids = [], [], []
    npz_paths = sorted(glob.glob(os.path.join(feature_dir, "*.npz")))
    if len(npz_paths) == 0:
        raise ValueError("No .npz files found in EEG_data_dir.")
    for npz_path in npz_paths:
        sid = file_to_sid(npz_path, pad=2)
        if sid not in risk_group_map:
            continue
        label = int(risk_group_map[sid])
        data = np.load(npz_path, allow_pickle=True)
        if "HFD_features" not in data:
            raise KeyError(f"'HFD_features' not found in {npz_path}")
        hfd = data["HFD_features"]
        n_windows = int(hfd.shape[0])
        subj_num = int(sid.split("-")[1])
        for w in range(n_windows):
            all_features.append(hfd[w].flatten())
            all_labels.append(label)
            all_sub_ids.append(subj_num)
    if len(all_features) == 0:
        raise ValueError("No EEG windows loaded for this task. Check mappings.")
    return {"X": np.array(all_features), "y": np.array(all_labels), "sub_ids": np.array(all_sub_ids)}

def compute_std_distance_to_centroid_only(X_data, y_data, sub_ids_data, kmeans_model, k_states):
    centroids = kmeans_model.cluster_centers_
    all_preds = kmeans_model.predict(X_data)
    unique_subjects = np.unique(sub_ids_data)
    subject_features = []
    for subject in unique_subjects:
        mask = (sub_ids_data == subject)
        if not np.any(mask):
            continue
        subj_X = X_data[mask]
        subj_preds = all_preds[mask]
        subj_label = int(y_data[mask][0])
        std_dist = np.full(k_states, np.nan)
        for s in range(k_states):
            state_windows = subj_X[subj_preds == s]
            if state_windows.shape[0] > 0:
                distances = np.linalg.norm(state_windows - centroids[s], axis=1)
                std_dist[s] = float(np.std(distances))
        subject_features.append({'subject_id': int(subject), 'label': subj_label, 'std_distance_to_centroid': std_dist})
    return subject_features

# ---------- LOSO observed (per-task) ----------
def run_task_loso_eeg(task_tuple, k_states_list=K_STATES_LIST_EEG, seed=SEED, save_dir=RESULTS_DIR):
    mri_df, names_col, group_col = load_mri_dataframe(MRI_csv_path)
    risk_map, df_task = build_task_subject_map(mri_df, group_col, task_tuple)
    eeg_data = load_eeg_features_for_map(EEG_data_dir, risk_map)
    unique_subject_nums = np.unique(eeg_data['sub_ids'])
    if unique_subject_nums.size < 2:
        raise ValueError("Need >=2 subjects for LOSO in this task.")
    all_fold_rows = []
    for test_subject in unique_subject_nums:
        train_mask = eeg_data['sub_ids'] != test_subject
        X_train = eeg_data['X'][train_mask]
        subids_train = eeg_data['sub_ids'][train_mask]
        test_mask = eeg_data['sub_ids'] == test_subject
        if X_train.shape[0] == 0 or np.sum(test_mask) == 0:
            continue
        per_K_probs, per_K_pvals, per_K_info = [], [], []
        for K in k_states_list:
            try:
                kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
                kmeans.fit(X_train)
            except Exception:
                continue
            subject_features = compute_std_distance_to_centroid_only(eeg_data['X'], eeg_data['y'], eeg_data['sub_ids'], kmeans, K)
            subj_ids = np.array([s['subject_id'] for s in subject_features])
            labels = np.array([s['label'] for s in subject_features])
            feat_mat = np.vstack([s['std_distance_to_centroid'] for s in subject_features])  # (n_subjects, K)
            train_sub_ids = np.unique(subids_train)
            train_mask_sub = np.isin(subj_ids, train_sub_ids)
            test_mask_sub = subj_ids == int(test_subject)
            if not np.any(test_mask_sub):
                continue
            X_train_sub = feat_mat[train_mask_sub]
            y_train_sub = labels[train_mask_sub]
            X_test_sub = feat_mat[test_mask_sub]
            p_list = []
            for state in range(K):
                vals = X_train_sub[:, state]
                valid_mask_state = ~np.isnan(vals)
                if np.sum(valid_mask_state) < 2:
                    p_list.append(1.0); continue
                groups = [vals[(y_train_sub == lab) & valid_mask_state] for lab in np.unique(y_train_sub)]
                _, p_val, _ = choose_statistical_test(groups)
                p_list.append(float(p_val))
            p_array = np.array(p_list)
            if p_array.size == 0 or np.all(np.isnan(p_array)) or np.all(p_array >= 1.0):
                continue
            best_state = int(np.nanargmin(p_array)); best_p = float(p_array[best_state])
            train_vals_for_state = X_train_sub[:, best_state]
            valid_train_mask = ~np.isnan(train_vals_for_state)
            if np.sum(valid_train_mask) < 2: continue
            if np.isnan(X_test_sub[0, best_state]): continue
            Xtr = train_vals_for_state[valid_train_mask].reshape(-1,1)
            ytr = y_train_sub[valid_train_mask]
            Xte = np.array(X_test_sub[:, best_state]).reshape(-1,1)
            clf = make_pipeline(StandardScaler(), LogisticRegression(penalty=None,
                                                                     class_weight='balanced', random_state=seed))
            clf.fit(Xtr, ytr)
            prob_pos = float(clf.predict_proba(Xte)[:,1][0])
            pred_label = int(clf.predict(Xte)[0])
            per_K_probs.append(prob_pos)
            per_K_pvals.append(best_p)
            per_K_info.append({'K': K, 'best_state': best_state, 'best_p': best_p, 'pred_label': pred_label})
            del clf; del kmeans
        if len(per_K_probs) == 0:
            continue
        weights = compute_pvalue_weights(np.array(per_K_pvals), smoothing=1e-10, power=2.0)
        probs_matrix = np.array(per_K_probs).reshape(1, -1)
        _, ensemble_prob = weighted_prediction(probs_matrix, weights)
        ensemble_prob_scalar = float(ensemble_prob[0])
        ensemble_pred = int(ensemble_prob_scalar > 0.5)
        true_label = int(np.unique(eeg_data['y'][eeg_data['sub_ids'] == test_subject])[0])
        row = {'test_subject': int(test_subject), 'true_label': true_label, 'ensemble_prob': ensemble_prob_scalar,
               'ensemble_pred': ensemble_pred, 'per_K_info': str(per_K_info), 'per_K_pvals': str(per_K_pvals),
               'per_K_probs': str(per_K_probs)}
        all_fold_rows.append(row)
    results_df = pd.DataFrame(all_fold_rows)
    return results_df

# ---------- Permutation testing (EEG only) ----------
def run_task_permutation_eeg(task_tuple, n_permutations=500, k_states_list=K_STATES_LIST_EEG, seed=SEED, save_dir=RESULTS_DIR):
    """
    For each permutation we shuffle training SUBJECT labels inside each LOSO fold and run the same per-K
    selection+logistic pipeline, then ensemble weights by p-values. Returns permutation F1 distribution.
    """
    # prepare deterministic seeds
    random.seed(seed); np.random.seed(seed)
    # prepare locals
    mri_df, names_col, group_col = load_mri_dataframe(MRI_csv_path)
    risk_map, df_task = build_task_subject_map(mri_df, group_col, task_tuple)
    eeg_data = load_eeg_features_for_map(EEG_data_dir, risk_map)
    unique_subject_nums = np.unique(eeg_data['sub_ids'])
    if unique_subject_nums.size < 2:
        raise ValueError("Need >=2 subjects for permutation LOSO in this task.")
    # Prepare permutation storage: list of lists, each inner list collects predicted labels across folds
    permutation_predictions = [ [] for _ in range(n_permutations) ]
    permutation_true_labels = []
    # LOSO folds over subjects
    for test_subject in unique_subject_nums:
        train_mask = eeg_data['sub_ids'] != test_subject
        test_mask = eeg_data['sub_ids'] == test_subject
        X_train = eeg_data['X'][train_mask]
        subids_train = eeg_data['sub_ids'][train_mask]
        if X_train.shape[0] == 0 or np.sum(test_mask) == 0:
            continue
        # Precompute per-K kmeans and subject-level feature matrices (std_distance_to_centroid)
        perK_subject_feats = {}  # K -> (subj_ids_array, labels_array, feat_mat (n_subjects, K_cluster) )
        for K in k_states_list:
            try:
                kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
                kmeans.fit(X_train)
            except Exception:
                continue
            subject_features = compute_std_distance_to_centroid_only(eeg_data['X'], eeg_data['y'], eeg_data['sub_ids'], kmeans, K)
            subj_ids = np.array([s['subject_id'] for s in subject_features])
            labels = np.array([s['label'] for s in subject_features])
            feat_mat = np.vstack([s['std_distance_to_centroid'] for s in subject_features])
            perK_subject_feats[K] = (subj_ids, labels, feat_mat)
            del kmeans
        # If no K worked, skip fold
        if len(perK_subject_feats) == 0:
            continue
        # true label for this test subject (append once per fold)
        true_label = int(np.unique(eeg_data['y'][eeg_data['sub_ids'] == test_subject])[0])
        permutation_true_labels.append(true_label)
        # For each permutation, perform per-K selection with permuted training labels, train logistic on permuted labels, get prob for test subject
        for perm in range(n_permutations):
            per_K_probs = []
            per_K_pvals = []
            for K, (subj_ids, labels, feat_mat) in perK_subject_feats.items():
                # Build train/test subject indices relative to subj_ids array
                train_sub_ids = np.unique(subids_train)
                train_mask_sub = np.isin(subj_ids, train_sub_ids)
                test_mask_sub = subj_ids == int(test_subject)
                if not np.any(test_mask_sub):
                    continue
                X_train_sub = feat_mat[train_mask_sub]
                X_test_sub = feat_mat[test_mask_sub]
                # Permute training subject labels (labels_train are labels aligned with subj_ids)
                y_train_sub_orig = labels[train_mask_sub].copy()
                # Shuffle labels (subject-wise)
                y_train_perm = np.random.permutation(y_train_sub_orig)
                # compute pvals per state using permuted labels
                p_list = []
                for state in range(K):
                    vals = X_train_sub[:, state]
                    valid_mask_state = ~np.isnan(vals)
                    if np.sum(valid_mask_state) < 2:
                        p_list.append(1.0); continue
                    groups = [vals[(y_train_perm == lab) & valid_mask_state] for lab in np.unique(y_train_perm)]
                    _, p_val, _ = choose_statistical_test(groups)
                    p_list.append(float(p_val))
                p_array = np.array(p_list)
                if p_array.size == 0 or np.all(np.isnan(p_array)) or np.all(p_array >= 1.0):
                    continue
                best_state = int(np.nanargmin(p_array)); best_p = float(p_array[best_state])
                train_vals_for_state = X_train_sub[:, best_state]
                valid_train_mask = ~np.isnan(train_vals_for_state)
                if np.sum(valid_train_mask) < 2:
                    continue
                if np.isnan(X_test_sub[0, best_state]):
                    continue
                # Train logistic on permuted labels
                Xtr = train_vals_for_state[valid_train_mask].reshape(-1,1)
                ytr = y_train_perm[valid_train_mask]
                # if permuted labels collapse to single class, produce neutral prob 0.5
                if len(np.unique(ytr)) < 2:
                    prob_pos = 0.5
                else:
                    clf = make_pipeline(StandardScaler(), LogisticRegression(penalty=None,
                                                                            class_weight='balanced', random_state=seed+perm))
                    clf.fit(Xtr, ytr)
                    Xte = np.array(X_test_sub[:, best_state]).reshape(-1,1)
                    prob_pos = float(clf.predict_proba(Xte)[:,1][0])
                    del clf
                per_K_probs.append(prob_pos)
                per_K_pvals.append(best_p)
            # after K loop, ensemble if any K produced result
            if len(per_K_probs) == 0:
                # fallback: predict majority class (use 0) -- but to be consistent, use neutral 0.5->pred0
                pred_label = 0
            else:
                weights = compute_pvalue_weights(np.array(per_K_pvals), smoothing=1e-10, power=2.0)
                probs_matrix = np.array(per_K_probs).reshape(1, -1)
                preds, weighted_prob = weighted_prediction(probs_matrix, weights)
                pred_label = int(preds[0])
            permutation_predictions[perm].append(pred_label)
    # After all folds, compute permutation F1 distribution
    permutation_f1 = []
    # If permutation_true_labels length is 0 then no folds produced data -> error
    if len(permutation_true_labels) == 0:
        raise ValueError("No folds produced predictions during permutation runs (check data/K).")
    for perm in range(n_permutations):
        preds = np.array(permutation_predictions[perm])
        if preds.size != len(permutation_true_labels):
            # If some K failed for some fold, lengths may mismatch; skip this permutation (or pad)
            # We'll skip to keep distribution conservative
            permutation_f1.append(0.0)
            continue
        f1 = f1_score(permutation_true_labels, preds, average='macro', zero_division=0)
        permutation_f1.append(float(f1))
    return {
        'permutation_true_labels': permutation_true_labels,
        'permutation_predictions': permutation_predictions,
        'permutation_f1': permutation_f1
    }

# ---------- Main: run tasks, observed + permutations ----------
def main(run_permutations=True, n_permutations=N_PERMUTATIONS, seed=SEED):
    TASKS = [
        ("N", "A+P+"),
        ("N", "A+P-"),
        ("A+P-", "A+P+"),
        ("N", "A+P-", "A+P+"),  # combined A+P- and A+P+
    ]
    np.random.seed(seed); random.seed(seed)
    summaries = []
    for task in TASKS:
        print("\n" + "="*60)
        print(f"TASK: {task}")
        print("="*60)
        # observed LOSO
        observed_df = run_task_loso_eeg(task, k_states_list=K_STATES_LIST_EEG, seed=seed, save_dir=RESULTS_DIR)
        if observed_df.empty:
            print(f"  No observed folds for task {task}, skipping.")
            continue
        # save observed per-fold
        class_str = "_vs_".join(task).replace("+", "plus").replace("-", "minus")
        task_dir = os.path.join(RESULTS_DIR, f"task_{class_str}")
        os.makedirs(task_dir, exist_ok=True)
        observed_csv = os.path.join(task_dir, "observed_loso_eeg_ensemble_per_fold.csv")
        observed_df.to_csv(observed_csv, index=False)
        print(f"  Saved observed per-fold results: {observed_csv}")
        # observed summary metrics
        y_true = observed_df['true_label'].values
        y_prob = observed_df['ensemble_prob'].values
        y_pred = observed_df['ensemble_pred'].values
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
        f1_obs = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        summary = {'task': class_str, 'n_folds': len(observed_df), 'accuracy': float(acc), 'auc': float(auc), 'f1_macro': float(f1_obs), 'confusion_matrix': cm.tolist()}
        # print the observed summary
        print(f"  Observed results: Accuracy={acc:.4f}, AUC={auc:.4f}, F1-macro={f1_obs:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        
        # permutation testing
        if run_permutations:
            print("  Running permutation testing for EEG (this may take a while)...")
            perm_res = run_task_permutation_eeg(task, n_permutations=n_permutations, k_states_list=K_STATES_LIST_EEG, seed=seed, save_dir=RESULTS_DIR)
            perm_f1 = np.array(perm_res['permutation_f1'])
            # p-value: proportion of permutations with f1 >= observed
            num_ge = np.sum(perm_f1 >= f1_obs)
            p_val = (int(num_ge) + 1) / (len(perm_f1) + 1)
            summary.update({'perm_p_value_f1': float(p_val), 'perm_mean_f1': float(np.mean(perm_f1)), 'perm_std_f1': float(np.std(perm_f1))})
            # save permutation distribution
            np.savez_compressed(os.path.join(task_dir, "permutation_f1_distribution.npz"), perm_f1=perm_f1)
            # save permutation predictions (optionally large)
            # np.savez_compressed(os.path.join(task_dir, "permutation_preds.npz"), preds=perm_res['permutation_predictions'])
            print(f"  Permutation p-value (F1) = {p_val:.4f} (observed F1={f1_obs:.4f})")
        # store summary and save
        pd.DataFrame([summary]).to_csv(os.path.join(task_dir, "loso_eeg_summary.csv"), index=False)
        print(f"  Saved task summary to {os.path.join(task_dir, 'loso_eeg_summary.csv')}")
        summaries.append(summary)
    # save all summaries
    if len(summaries) > 0:
        pd.DataFrame(summaries).to_csv(os.path.join(RESULTS_DIR, "all_tasks_eeg_summary.csv"), index=False)
        print(f"\nAll tasks summary saved to {os.path.join(RESULTS_DIR, 'all_tasks_eeg_summary.csv')}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO EEG ensemble + permutation testing")
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(run_permutations=True, n_permutations=args.permutations, seed=args.seed)
