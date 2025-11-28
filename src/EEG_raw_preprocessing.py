#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resting_state_preprocess.py
—————————————————————————————————————————————————
Pre‑process resting‑state EEG for PEARL dataset and save
per‑subject NumPy archives with risk‑group labels.

Key changes 2025‑06‑06
• Explicit zero‑phase FIR filtering (phase='zero').
• Average reference applied after artefact correction.
• Safer label handling + numeric encoding for machine learning.
• Removed legacy FCz stacking code.
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

# ─────────────────────────────────────────────────────────────
# 0. Small helper
# ─────────────────────────────────────────────────────────────

def get_start_end(events, start_code: int = 4, end_code: int = 11):
    """Return sample indices of the first `start_code` and `end_code` events."""
    event_codes = events[:, 2]
    start_idx = end_idx = None

    if start_code in event_codes:
        start_idx = events[np.where(event_codes == start_code)[0][0], 0]

    if end_code in event_codes:
        end_idx = events[np.where(event_codes == end_code)[0][0], 0]

    return start_idx, end_idx


# ─────────────────────────────────────────────────────────────
# 1. Locate files & determine common subjects
# ─────────────────────────────────────────────────────────────
ROOT = "/data/s.dharia-ra/PEARL"
file_paths_eeg   = glob.glob(f"{ROOT}/original_data/sub-*/eeg/*_task-rest_eeg.vhdr")
file_paths_fmri1 = glob.glob(f"{ROOT}/fMRI-PEARL/derivatives_selected/sub-*/")
file_paths_fmri2 = glob.glob(f"{ROOT}/fMRI-PEARL/PEARL-Neuro_Database_MRI_Stephen_Smith/sub-*/")

subjects_eeg   = {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in file_paths_eeg}
subjects_fmri1 = {os.path.basename(os.path.normpath(p))              for p in file_paths_fmri1}
subjects_fmri2 = {os.path.basename(os.path.normpath(p))              for p in file_paths_fmri2}

common_subjects   = subjects_eeg & subjects_fmri1 & subjects_fmri2
removed_subjects  = (subjects_eeg | subjects_fmri1 | subjects_fmri2) - common_subjects

print(f"Common subjects (EEG ∩ fMRI1 ∩ fMRI2): {sorted(common_subjects)}")
print(f"Removed subjects (missing in ≥1 modality): {sorted(removed_subjects)}\n")

file_paths = file_paths_eeg#sorted([p for p in file_paths_eeg if os.path.basename(os.path.dirname(os.path.dirname(p))) in common_subjects])

# ─────────────────────────────────────────────────────────────
# 2. Load genotype table & map each subject to a risk group
# ─────────────────────────────────────────────────────────────
labels_df = (
    pd.read_csv(f"{ROOT}/fMRI-PEARL/participants.tsv", sep="\t")
      .query("participant_id in @common_subjects")
      .copy()
)

def assign_risk_group(row):
    """Return 'N', 'A+P–', or 'A+P+' according to APOE & PICALM genotype."""
    if "e4" in str(row["APOE_haplotype"]).lower():
        picalm = str(row["PICALM_rs3851179"]).replace("/", "").upper()
        return "A+P+" if picalm == "GG" else "A+P–"
    return "N"

labels_df["risk_group"] = labels_df.apply(assign_risk_group, axis=1)

# Map to integers for ML friendliness
LABEL_MAP = {"N": 0, "A+P–": 1, "A+P+": 2}
group_dict  = labels_df.set_index("participant_id")["risk_group"].to_dict()
label_dict2 = {k: LABEL_MAP[v] for k, v in group_dict.items()}

print(f"Participants in genotype table: {len(labels_df)}")
print(labels_df[["participant_id", "risk_group"]].head(), "\n")

# ─────────────────────────────────────────────────────────────
# 3. Pre‑processing parameters
# ─────────────────────────────────────────────────────────────
LOWCUT = 1.0        # Hz
HIGHCUT = 45.0      # Hz
ADJUST_SAMPLES = 361_291  # ≈ 23.5 min @ 256 Hz – must match protocol
OUTPUT_DIR = "resting_state_preprocessed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 4. Main loop
# ─────────────────────────────────────────────────────────────
print("Starting EEG preprocessing…\n")

for idx, path in enumerate(file_paths, 1):

    participant_id = re.search(r"sub-\d+", path).group(0)
    group_label    = group_dict.get(participant_id, "N")  # default to 'N' if missing
    group_numeric  = LABEL_MAP[group_label]

    out_file = os.path.join(OUTPUT_DIR, f"{participant_id}_preprocessed.npz")
    if os.path.exists(out_file):
        print(f"[{idx:3}/{len(file_paths)}] {participant_id} ({group_label}) – already done, skipping.")
        continue

    print(f"[{idx:3}/{len(file_paths)}] {participant_id} ({group_label}) – processing…")

    try:
        # 4‑A. Load & crop ————————————————————————————————
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        events, _ = mne.events_from_annotations(raw)
        start_smp, end_smp = get_start_end(events, start_code=4, end_code=11)

        if start_smp is not None and end_smp is not None:
            raw.crop(tmin=start_smp / raw.info["sfreq"],
                     tmax=end_smp   / raw.info["sfreq"],)
        elif start_smp is not None:
            new_end = min(start_smp + ADJUST_SAMPLES, raw.n_times)
            raw.crop(tmin=start_smp / raw.info["sfreq"],
                     tmax=new_end  / raw.info["sfreq"])
            print("  • only start event found ➝ padded window.")
        elif end_smp is not None:
            new_start = max(end_smp - ADJUST_SAMPLES, 0)
            raw.crop(tmin=new_start / raw.info["sfreq"],
                     tmax=end_smp   / raw.info["sfreq"])
            print("  • only end event found ➝ back‑filled window.")
        else:
            print("  • no start/end markers – skipping.")
            continue

        # 4‑B. Resample & (HP + Notch) ————————————————————
        raw.resample(256)
        raw_hp = raw.copy().filter(l_freq=LOWCUT, h_freq=None)
        raw_hp.notch_filter(50)

        # 4‑C. ICA (Picard) —————————————————————————————
        ica = ICA(n_components=42, random_state=42, max_iter="auto")
        ica.fit(raw_hp)

        muscle_idx, _ = ica.find_bads_muscle(raw_hp, threshold=0.9)
        eye_idx,    _ = ica.find_bads_eog   (raw_hp, ch_name=["Fp2", "Fp1"],
                                             threshold=0.9, measure="correlation")
        exclude = sorted(set(muscle_idx + eye_idx))
        print(f"  • excluding ICA comps: {exclude}")
        ica.apply(raw, exclude=exclude)

        # 4‑D. Final band‑pass (1–45 Hz) & average reference ———
        raw.filter(l_freq=LOWCUT, h_freq=HIGHCUT)

        # raw, _ = mne.set_eeg_reference(raw, ref_channels="average", projection=False)

        # 4‑E. Export clean data —————————————————————————
        data = raw.get_data()  # channels × time
        np.savez_compressed(out_file, data=data, label=group_numeric, label_str=group_label)
        print(f"  ✔ saved ➝ {out_file}")

    except Exception as err:
        print(f"  ✖ error with {participant_id}: {err}")
        continue

print("\nICA‑based artefact‑removal pipeline complete.")
