import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# --- Configuration: Update these paths ---


def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian (RBF) kernel between x and y"""
    # Ensure inputs are 2D
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)
    
    x_norm = torch.sum(x**2, dim=1, keepdim=True)
    y_norm = torch.sum(y**2, dim=1, keepdim=True) 
    distances = x_norm + y_norm.transpose(0, 1) - 2 * torch.mm(x, y.transpose(0, 1))  # Use transpose instead of .T
    return torch.exp(-distances / (2 * sigma**2))

def mmd_loss(source_features, target_features, sigma=2.0):
    """Compute MMD loss between source and target feature distributions"""
    # Ensure inputs are 2D
    if source_features.dim() > 2:
        source_features = source_features.view(source_features.size(0), -1)
    if target_features.dim() > 2:
        target_features = target_features.view(target_features.size(0), -1)
    
    K_ss = gaussian_kernel(source_features, source_features, sigma)
    K_tt = gaussian_kernel(target_features, target_features, sigma)
    K_st = gaussian_kernel(source_features, target_features, sigma)
    return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()


def multi_scale_mmd(source, target):
    sigmas = [0.1, 1.0, 10.0]  # Multiple scales
    total_mmd = 0
    for sigma in sigmas:
        total_mmd += mmd_loss(source, target, sigma)
    return total_mmd / len(sigmas)


# Base directory for all data
BASE_DATA_DIR = "./data"
SMRI_DATA_DIR = os.path.join(BASE_DATA_DIR, "MRI_data")
EEG_DATA_DIR = os.path.join(BASE_DATA_DIR, "EEG_data/HFD_PSD_4sec_75overlap_6min")
FMRI_DATA_DIR = os.path.join(BASE_DATA_DIR, "fMRI_data")


# --- Step 1: Helper Functions to Load Data for Each Modality ---

def parse_subject_id_from_sMRI(name_string):
    """Extracts the numeric part of the subject ID (e.g., '01' from 'sub-01_T1w')."""
    match = re.search(r'sub-(\d+)', name_string)
    if match:
        return match.group(1)
    return None

def load_sMRI_data(data_dir):
    """
    Loads and combines sMRI features from multiple CSV files.
    Returns a dictionary of features and labels, keyed by standardized subject ID.
    """
    files = [
        os.path.join(data_dir, 'ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv'),
        os.path.join(data_dir, 'ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv'),
        os.path.join(data_dir, 'ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv'),
        os.path.join(data_dir, 'ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv'),
        os.path.join(data_dir, 'ROI_aal3_Vgm.csv_with_groups.csv')
    ]

    X_list = []
    df_for_ids_and_labels = None

    for path in files:
        if not os.path.exists(path):
            print(f"Warning: sMRI file not found at {path}")
            continue
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if '???' in c], errors='ignore')
        feat_cols = [c for c in df.columns if c not in ['names', 'GROUP']]
        X_list.append(df[feat_cols].values)
        if df_for_ids_and_labels is None:
            df_for_ids_and_labels = df

    if not X_list:
        raise FileNotFoundError("No sMRI data files were found or loaded.")

    # Horizontally stack features from all files
    X_combined = np.hstack(X_list)
    
    # Get subject IDs and labels from the first dataframe
    subject_ids_raw = df_for_ids_and_labels['names'].values
    labels = df_for_ids_and_labels['GROUP'].values # Assuming labels are consistent across files for a subject

    # Create the final dictionary
    sMRI_data_dict = {}
    for i, raw_id in enumerate(subject_ids_raw):
        subject_id = parse_subject_id_from_sMRI(raw_id)
        if subject_id:
            sMRI_data_dict[subject_id] = {
                "features": X_combined[i],
                "label": labels[i]
            }
    print(f"Loaded sMRI data for {len(sMRI_data_dict)} subjects.")
    return sMRI_data_dict

# EEG loading function for PSD + HFD features
def load_eeg_data(data_dir, labels = True):
    """
    Loads all EEG .npz files from a directory.
    Returns a dictionary of features keyed by subject ID.
    """
    eeg_data_dict = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"EEG data directory not found: {data_dir}")
        
    for filename in os.listdir(data_dir):
        if filename.endswith('.npz'):
            # Assumes filename is like '1.npz', '2.npz', etc.
            subject_id = filename.split('.')[0]
            # Standardize to two digits with leading zero, e.g., '1' -> '01'
            subject_id_padded = subject_id.zfill(2)
            
            file_path = os.path.join(data_dir, filename)
            eeg_data = np.load(file_path, allow_pickle=True)
            # Concatenate HFD and PSD features
            concate_data = np.concatenate((eeg_data['HFD_features'].mean(0), eeg_data['PSD_features'].mean(0)), axis=1)
            # print(concate_data.shape)
            labels = eeg_data['label']
            eeg_data_dict[subject_id_padded] = concate_data
            eeg_data_dict[subject_id_padded + '_label'] = labels

    # print(f"Loaded EEG data for {len(eeg_data_dict)} subjects.")
    return eeg_data_dict


# def load_eeg_data(data_dir, labels=True):
#     """
#     Loads all EEG .npz files from a directory.
#     Returns a dictionary of features keyed by subject ID.
#     """
#     eeg_data_dict = {}
#     if not os.path.isdir(data_dir):
#         raise FileNotFoundError(f"EEG data directory not found: {data_dir}")
        
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.npz'):
#             # Assumes filename is like '1.npz', '2.npz', etc. (subject numbers)
#             subject_id = filename.split('.')[0]
#             # Standardize to two digits with leading zero, e.g., '1' -> '01'
#             subject_id_padded = subject_id.zfill(2)
            
#             file_path = os.path.join(data_dir, filename)
#             eeg_data = np.load(file_path, allow_pickle=True)
            
#             # Load features from the no-windowing extraction
#             # Shape should be (num_channels, 10) where 10 = 5 PSD + 5 HFD
#             features = eeg_data['features']
#             # unsqueeze at axis 0 to add batch dimension if needed
#             features = np.expand_dims(features, axis=0)  # Shape: (1, num_channels, 10)
#             print(f"Shape of EEG features for {subject_id_padded}: {features.shape}")
            
#             # Handle label - could be scalar or array
#             labels_raw = eeg_data['label']
            
#             # Convert to array format to match expected structure
#             if labels_raw.ndim == 0:  # If it's a scalar (0-dimensional)
#                 # For no-windowing approach, we have one label per subject
#                 # but we need to create an array to match the structure expected by other functions
#                 labels = np.array([labels_raw.item()])  # Single label in array format
#             else:
#                 labels = labels_raw
            
#             print(f"Labels shape for {subject_id_padded}: {labels.shape}, Label: {labels[0] if len(labels) > 0 else 'None'}")
            
#             # Store features and labels
#             eeg_data_dict[subject_id_padded] = features  # Shape: (num_channels, 10)
#             eeg_data_dict[subject_id_padded + '_label'] = labels  # Shape: (1,) for single subject label

#     return eeg_data_dict


### PLV based EEG loading function
# def load_eeg_data(data_dir, labels = True):
#     """
#     Loads all EEG .npz files from a directory.
#     Returns a dictionary of features keyed by subject ID.
#     """
#     eeg_data_dict = {}
#     if not os.path.isdir(data_dir):
#         raise FileNotFoundError(f"EEG data directory not found: {data_dir}")
        
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.npz'):
#             # Assumes filename is like 'sub-01_connectivity.npz', 'sub-02_connectivity.npz', etc.
#             subject_id = filename.split('_')[0].replace('sub-', '')
#             # Standardize to two digits with leading zero, e.g., '1' -> '01'
#             subject_id_padded = subject_id.zfill(2)
            
#             file_path = os.path.join(data_dir, filename)
#             eeg_data = np.load(file_path, allow_pickle=True)
#             # Concatenate HFD and PSD features
#             concate_data = eeg_data['connectivity']
#             print(f"Shape of concatenated data for {subject_id_padded}: {concate_data.shape}")
            
#             # Handle label - could be scalar or array
#             labels_raw = eeg_data['label']
            
#             # Convert to array format to match expected structure
#             if labels_raw.ndim == 0:  # If it's a scalar (0-dimensional)
#                 # Create array with same length as number of samples/windows
#                 num_samples = concate_data.shape[0]
#                 labels = np.full(num_samples, labels_raw.item())  # .item() converts scalar array to Python scalar
#             else:
#                 labels = labels_raw
            
#             print(f"Labels shape for {subject_id_padded}: {labels.shape}, First label: {labels[0] if len(labels) > 0 else 'None'}")
            
#             eeg_data_dict[subject_id_padded] = concate_data
#             eeg_data_dict[subject_id_padded + '_label'] = labels

#     return eeg_data_dict

def load_fmri_data(data_dir, labels = True):
    """
    Loads all fMRI .npz files from a directory.
    Returns a dictionary of features keyed by subject ID.
    """
    # No need for label mapping since labels are already integers
    # label_mapping = {"N": 0, "A+P-": 1, "A+P+": 2}

    fmri_data_dict = {}
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"fMRI data directory not found: {data_dir}")

    label_stats = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_windows.npz'):
            # Extract subject ID and standardize format
            if filename.startswith('sub-'):
                # Handle 'sub-01_windows.npz' format
                subject_id = filename.split('_')[0].replace('sub-', '')
            else:
                # Handle '01_windows.npz' format
                subject_id = filename.split('_')[0]
            
            # Ensure zero-padded format (e.g., '1' -> '01')
            subject_id = subject_id.zfill(2)
            
            file_path = os.path.join(data_dir, filename)
            fmri_data = np.load(file_path, allow_pickle=True)
            # Concatenate PA and AP scans
            concate_data_fMRI = np.concatenate((fmri_data["PA"], fmri_data["AP"]), axis=0)

            # Handle numeric labels - extract the integer value
            labels_raw = fmri_data['label']
            if labels_raw.ndim == 0:  # If it's a scalar array like array(0)
                labels_numeric = labels_raw.item()  # Convert to Python int
            else:
                labels_numeric = int(labels_raw)  # Convert to int if it's already a number
            
            print(f"Subject {subject_id}: raw label = {labels_raw}, numeric = {labels_numeric}")
            
            # Keep track of label statistics
            label_stats[labels_numeric] = label_stats.get(labels_numeric, 0) + 1
            
            # extend the labels to match the number of windows
            labels_array = np.array([labels_numeric] * concate_data_fMRI.shape[0])
            
            fmri_data_dict[subject_id + '_label'] = labels_array
            fmri_data_dict[subject_id] = concate_data_fMRI
    
    print(f"fMRI Label statistics: {label_stats}")        
    print(f"Loaded fMRI data for {len([k for k in fmri_data_dict.keys() if not k.endswith('_label')])} subjects.")
    return fmri_data_dict

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        # Ensure inputs are logits for binary classification
        if inputs.size(1) == 1:
            # Single output neuron
            p = torch.sigmoid(inputs.squeeze(1))
            ce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(1), targets.float(), reduction='none')
        else:
            # Two output neurons
            p = F.softmax(inputs, dim=1)[:, 1]  # Probability of positive class
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_tensor = torch.tensor(self.alpha, device=inputs.device)
                alpha_t = alpha_tensor[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs + 1e-8)  # Add small epsilon for numerical stability

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Single alpha value for all classes
                alpha_t = self.alpha
            else:
                # Per-class alpha values
                if isinstance(self.alpha, torch.Tensor):
                    alpha = self.alpha.to(inputs.device)
                else:
                    alpha = torch.tensor(self.alpha, device=inputs.device)
                alpha_t = alpha.gather(0, targets)
                ce_loss = alpha_t.unsqueeze(1) * ce_loss
        
        # Apply focal loss weight
        if self.alpha is not None and not isinstance(self.alpha, (float, int)):
            loss = focal_weight.unsqueeze(1) * ce_loss
        else:
            loss = focal_weight.unsqueeze(1) * ce_loss
            if isinstance(self.alpha, (float, int)):
                loss = self.alpha * loss

        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()  # Sum over classes, mean over batch
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.sum(dim=1)  # Sum over classes for each sample

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Compute binary cross entropy for each class
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        
        # Compute focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_tensor = torch.tensor(self.alpha, device=inputs.device)
                alpha_t = alpha_tensor.unsqueeze(0).expand_as(targets)
                alpha_t = alpha_t * targets + (1 - alpha_t) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    




# from utils import load_eeg_data, load_fmri_data

# eeg_data_dict = load_eeg_data("HFD_PSD_stats_features_4sec_75overlap")
# fMRI_data_dict = load_fmri_data("fMRI_subject_windows")



def train_model(model, train_loader, test_loader, criterion, optimizer, device='cpu'):
    model.to(device)
    model.train()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_sub_ids = []

    for batch in train_loader:
        inputs, labels, sub_id = batch
        inputs, labels, sub_id = inputs.to(device), labels.to(device), sub_id

        #load out the test_loader one batch
        test_batch = next(iter(test_loader))
        test_inputs, test_labels, test_sub_id = test_batch
        test_inputs, test_labels, test_sub_id = test_inputs.to(device), test_labels.to(device), test_sub_id
        
        # size of train and test batch
        # print(f"Train batch size: {inputs.size()}, Test batch size: {test_inputs.size()}")
        optimizer.zero_grad()
        outputs, source_b2 = model(inputs)
        
        _, target_b2 = model(test_inputs)
    

        # print(labels, outputs)
        loss = criterion(outputs, labels) + multi_scale_mmd(source_b2, target_b2)
        loss.backward()
        optimizer.step()

        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_sub_ids.extend(sub_id)

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss, all_preds, all_labels, all_sub_ids

def subject_level_metrics(preds, labels, sub_ids):
    """
    Calculate subject-level metrics using majority voting.
    
    Args:
        preds: Sample-level predictions (list of tensors or flat array)
        labels: Sample-level labels (list of tensors or flat array)
        sub_ids: Subject IDs for each sample
    
    Returns:
        dict: Subject-level accuracy, f1, and details
    """
    from collections import Counter
    from sklearn.metrics import accuracy_score, f1_score
    
    # Handle different input formats
    if isinstance(preds, list):
        # If preds is a list of tensors, concatenate and convert to predictions
        preds_concat = torch.cat(preds, dim=0)
        if preds_concat.dim() > 1:
            # Convert logits to predictions
            preds_flat = torch.argmax(preds_concat, dim=1).numpy()
        else:
            preds_flat = preds_concat.numpy()
    else:
        preds_flat = np.array(preds)
    
    if isinstance(labels, list):
        # If labels is a list of tensors, concatenate
        labels_flat = torch.cat(labels, dim=0).numpy()
    else:
        labels_flat = np.array(labels)
    
    # Convert sub_ids to numpy array
    sub_ids = np.array(sub_ids)
    
    # Get subject-level predictions via majority voting
    subject_preds, subject_labels = [], []
    
    for subject in np.unique(sub_ids):
        mask = sub_ids == subject
        # Majority vote for predictions
        subject_pred = Counter(preds_flat[mask]).most_common(1)[0][0]
        # True label (should be consistent across samples)
        subject_label = labels_flat[mask][0]
        
        subject_preds.append(subject_pred)
        subject_labels.append(subject_label)
    
    # Calculate metrics
    subj_acc = accuracy_score(subject_labels, subject_preds)
    subj_f1 = f1_score(subject_labels, subject_preds, average='weighted', zero_division=0)
    sample_acc = accuracy_score(labels_flat, preds_flat)
    
    return {
        'subject_accuracy': subj_acc,
        'subject_f1': subj_f1,
        'sample_accuracy': sample_acc,
        'num_subjects': len(subject_preds),
        'num_samples': len(preds_flat)
    }


def test_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_sub_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, sub_id = batch
            inputs, labels, sub_id = inputs.to(device), labels.to(device), sub_id

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_sub_ids.extend(sub_id)

            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss, all_preds, all_labels, all_sub_ids

# def get_X_y(dict, comman_keys, modality="eeg", classes=("N","A+P+")):
#     X = []
#     y = []
#     sub_ids = []
    
#     # Define which classes to include
#     if classes == ("N", "A+P+"):
#         classes_int = [0, 2]
#     elif classes == ("N", "A+P-"):
#         classes_int = [0, 1]
#     elif classes == ("A+P-","A+P+"):
#         classes_int = [1, 2]

#     for subject_id in comman_keys:
#         if subject_id.endswith('_label'):
#             continue
        
#         if modality == "sMRI":
#             if dict[subject_id]['label'] not in classes:
#                 continue
#             X.append(dict[subject_id]['features'])
#             y.append(dict[subject_id]['label'])
#             sub_ids.append(subject_id)
#         else:
#             # Check if subject has the required label key
#             label_key = subject_id + '_label'
#             if label_key not in dict:
#                 continue
                
#             # Get first label to check class membership
#             first_label = dict[label_key][0]
#             if first_label not in classes_int:
#                 continue
                
#             # For EEG/fMRI data with temporal windows
#             X.append(dict[subject_id])
#             y.append(dict[label_key])
#             # Create a list of subject_id repeated for each sample/window
#             num_samples = len(dict[label_key])
#             sub_ids.extend([subject_id] * num_samples)
    
#     if modality == "sMRI":
#         X = np.array(X)
#         y = np.array(y)
#     else:
#         X = np.concatenate(X, axis=0)  # Concatenate along the first axis
#         y = np.concatenate(y, axis=0)  # Concatenate labels along the first axis
#         sub_ids = np.array(sub_ids)

#     # Binary mapping at the end
#     if classes == ("N", "A+P+"):  # 0 vs 2 -> 0 vs 1
#         y = np.where(y == 0, 0, 1)
#     elif classes == ("N", "A+P-"):  # 0 vs 1 -> 0 vs 1 (already binary)
#         pass  # Already 0 and 1
#     elif classes == ("A+P-", "A+P+"):  # 1 vs 2 -> 0 vs 1
#         y = np.where(y == 1, 0, 1)

#     print(f"{modality} - Classes: {classes}, Subjects: {len(np.unique(sub_ids))}, Samples: {len(X)}")
#     # print(f"{modality} - Label distribution after mapping: {np.bincount(y)}")

#     return X, y, sub_ids



def get_X_y(dict, comman_keys, modality="eeg", classes=("N","A+P+")):
    X = []
    y = []
    sub_ids = []
    
    # Define which classes to include
    if classes == ("N", "A+P+"):
        classes_int = [0, 2]
    elif classes == ("N", "A+P-"):
        classes_int = [0, 1]
    elif classes == ("A+P-","A+P+"):
        classes_int = [1, 2]
    elif classes == ("N", "A+P-","A+P+"):
        classes_int = [0, 1, 2]
        #when it is all dont change labels
    elif classes == ("all"):   
        classes_int = [0, 1, 2]


    # Debug: Print first few subjects to see what labels exist
    if modality == "fMRI":
        print(f"DEBUG - {modality}: Checking label values for first few subjects...")
        available_labels = set()
        debug_count = 0
        for subject_id in comman_keys:
            if subject_id.endswith('_label'):
                continue
            label_key = subject_id + '_label'
            if label_key in dict:
                sample_labels = dict[label_key]
                unique_labels = np.unique(sample_labels)
                available_labels.update(unique_labels)
                if debug_count < 5:
                    print(f"  Subject {subject_id}: labels = {unique_labels} (shape: {sample_labels.shape})")
                    debug_count += 1
        print(f"All available labels in {modality}: {sorted(available_labels)}")
        print(f"Looking for classes_int: {classes_int}")

    subjects_with_target_classes = 0
    subjects_checked = 0

    for subject_id in comman_keys:
        if subject_id.endswith('_label'):
            continue
        
        subjects_checked += 1
            
        if modality == "sMRI":
            subject_label_str = dict[subject_id]['label']
            # Check if subject's label is in the requested classes, or if we are loading all classes
            if not (classes == ("all") or subject_label_str in classes):
                continue
            
            # Create proper label mapping based on selected classes
            if classes == ("all"):
                label_mapping = {"N": 0, "A+P-": 1, "A+P+": 2}
            elif len(classes) == 2:
                label_mapping = {classes[0]: 0, classes[1]: 1}
            elif len(classes) == 3:
                label_mapping = {classes[0]: 0, classes[1]: 1, classes[2]: 1} # Maps two classes to 1
            else:
                raise ValueError("Unsupported number of classes for sMRI. Choose 2 or 3 classes.")
            
            # Map string label to integer
            mapped_label = label_mapping[subject_label_str]
            
            X.append(dict[subject_id]['features'])
            y.append(mapped_label)  # Store the mapped integer label
            sub_ids.append(subject_id)
            subjects_with_target_classes += 1
        else:
            # Check if subject has the required label key
            label_key = subject_id + '_label'
            if label_key not in dict:
                if modality == "fMRI":
                    print(f"  WARNING: No label key '{label_key}' found for subject {subject_id}")
                continue
                
            # Get first label to check class membership
            first_label = dict[label_key][0]
            if first_label not in classes_int:
                if modality == "fMRI" and subjects_checked <= 5:  # Only show first few for debugging
                    print(f"  Subject {subject_id}: first_label = {first_label}, not in {classes_int}")
                continue
                
            # For EEG/fMRI data with temporal windows
            X.append(dict[subject_id])
            y.append(dict[label_key])
            # Create a list of subject_id repeated for each sample/window
            num_samples = len(dict[label_key])
            sub_ids.extend([subject_id] * num_samples)
            subjects_with_target_classes += 1
    
    print(f"DEBUG - {modality}: Subjects checked: {subjects_checked}, with target classes: {subjects_with_target_classes}")
    
    if len(X) == 0:
        print(f"ERROR - {modality}: No subjects found with classes {classes} (integer: {classes_int})")
        # For now, let's suggest using different classes
        if modality == "fMRI":
            print("Try using classes=('N', 'A+P-') instead, or check what classes are available in your fMRI data")
        return np.array([]), np.array([]), np.array([])
    
    if modality == "sMRI":
        X = np.array(X)
        y = np.array(y)  # Already mapped to integers
        sub_ids = np.array(sub_ids)
    else:
        X = np.concatenate(X, axis=0)  # Concatenate along the first axis
        y = np.concatenate(y, axis=0)  # Concatenate labels along the first axis
        sub_ids = np.array(sub_ids)
        
        # Binary mapping at the end for EEG/fMRI
        if classes == ("N", "A+P+"):  # 0 vs 2 -> 0 vs 1
            y = np.where(y == 0, 0, 1)
        elif classes == ("N", "A+P-"):  # 0 vs 1 -> 0 vs 1 (already binary)
            pass  # Already 0 and 1
        elif classes == ("A+P-", "A+P+"):  # 1 vs 2 -> 0 vs 1
            y = np.where(y == 1, 0, 1)
        elif classes == ("N", "A+P-", "A+P+"):  # 3-class case, map 0 to 0 and 1 and 2 to 1
            print("Mapping 3 classes to binary: N=0, A+P- and A+P+ = 1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            for i in range(len(y)):
                if y[i] == 0:
                    y[i] = 0
                else:
                    y[i] = 1
        elif classes == ("all"):  # No mapping, keep all three classes
            pass  # Keep original labels 0, 1, 2
    
    print(f"{modality} - Classes: {classes}, Subjects: {len(np.unique(sub_ids))}, Samples: {len(X)}")
    print(f"{modality} - Label distribution after mapping: {np.bincount(y)}")

    return X, y, sub_ids

# def get_X_y(dict, comman_keys, modality="eeg", classes=("N","A+P+")):
#     X = []
#     y = []
#     sub_ids = []
    
#     # Define which classes to include
#     if classes == ("N", "A+P+"):
#         classes_int = [0, 2]
#     elif classes == ("N", "A+P-"):
#         classes_int = [0, 1]
#     elif classes == ("A+P-","A+P+"):
#         classes_int = [1, 2]
#     elif classes == ("N", "all") or classes == ("all", "N"):  # NEW: N vs all others
#         classes_int = [0, 1, 2]  # Include all classes, will map later
#     else:
#         raise ValueError(f"Unsupported class combination: {classes}")

#     # Debug: Print first few subjects to see what labels exist
#     if modality == "fMRI":
#         print(f"DEBUG - {modality}: Checking label values for first few subjects...")
#         available_labels = set()
#         debug_count = 0
#         for subject_id in comman_keys:
#             if subject_id.endswith('_label'):
#                 continue
#             label_key = subject_id + '_label'
#             if label_key in dict:
#                 sample_labels = dict[label_key]
#                 unique_labels = np.unique(sample_labels)
#                 available_labels.update(unique_labels)
#                 if debug_count < 5:
#                     print(f"  Subject {subject_id}: labels = {unique_labels} (shape: {sample_labels.shape})")
#                     debug_count += 1
#         print(f"All available labels in {modality}: {sorted(available_labels)}")
#         print(f"Looking for classes_int: {classes_int}")

#     subjects_with_target_classes = 0
#     subjects_checked = 0

#     for subject_id in comman_keys:
#         if subject_id.endswith('_label'):
#             continue
        
#         subjects_checked += 1
            
#         if modality == "sMRI":
#             # Check if subject's label is in the requested classes
#             subject_label = dict[subject_id]['label']
            
#             # Handle "N vs all" case
#             if classes in [("N", "all"), ("all", "N")]:
#                 # Include all subjects, will map N=0, others=1
#                 mapped_label = 0 if subject_label == "N" else 1
#             else:
#                 # Original logic for specific class pairs
#                 if subject_label not in classes:
#                     continue
                
#                 # Create proper label mapping based on selected classes
#                 if len(classes) == 2:
#                     label_mapping = {classes[0]: 0, classes[1]: 1}
#                 else:
#                     # For 3-class case
#                     label_mapping = {"N": 0, "A+P-": 1, "A+P+": 2}
                
#                 # Map string label to integer
#                 mapped_label = label_mapping[subject_label]
            
#             X.append(dict[subject_id]['features'])
#             y.append(mapped_label)
#             sub_ids.append(subject_id)
#             subjects_with_target_classes += 1
#         else:
#             # Check if subject has the required label key
#             label_key = subject_id + '_label'
#             if label_key not in dict:
#                 if modality == "fMRI":
#                     print(f"  WARNING: No label key '{label_key}' found for subject {subject_id}")
#                 continue
                
#             # Get first label to check class membership
#             first_label = dict[label_key][0]
            
#             # Handle "N vs all" case for EEG/fMRI
#             if classes in [("N", "all"), ("all", "N")]:
#                 # Include all subjects with any of the three labels
#                 if first_label not in [0, 1, 2]:
#                     continue
#             else:
#                 # Original logic for specific class pairs
#                 if first_label not in classes_int:
#                     if modality == "fMRI" and subjects_checked <= 5:
#                         print(f"  Subject {subject_id}: first_label = {first_label}, not in {classes_int}")
#                     continue
                
#             # For EEG/fMRI data with temporal windows
#             X.append(dict[subject_id])
#             y.append(dict[label_key])
#             # Create a list of subject_id repeated for each sample/window
#             num_samples = len(dict[label_key])
#             sub_ids.extend([subject_id] * num_samples)
#             subjects_with_target_classes += 1
    
#     print(f"DEBUG - {modality}: Subjects checked: {subjects_checked}, with target classes: {subjects_with_target_classes}")
    
#     if len(X) == 0:
#         print(f"ERROR - {modality}: No subjects found with classes {classes} (integer: {classes_int})")
#         return np.array([]), np.array([]), np.array([])
    
#     if modality == "sMRI":
#         X = np.array(X)
#         y = np.array(y)  # Already mapped to integers
#         sub_ids = np.array(sub_ids)
#     else:
#         X = np.concatenate(X, axis=0)  # Concatenate along the first axis
#         y = np.concatenate(y, axis=0)  # Concatenate labels along the first axis
#         sub_ids = np.array(sub_ids)
        
#         # Binary mapping at the end for EEG/fMRI
#         if classes == ("N", "A+P+"):  # 0 vs 2 -> 0 vs 1
#             y = np.where(y == 0, 0, 1)
#         elif classes == ("N", "A+P-"):  # 0 vs 1 -> 0 vs 1 (already binary)
#             pass  # Already 0 and 1
#         elif classes == ("A+P-", "A+P+"):  # 1 vs 2 -> 0 vs 1
#             y = np.where(y == 1, 0, 1)
#         elif classes in [("N", "all"), ("all", "N")]:  # N vs all others -> 0 vs 1
#             y = np.where(y == 0, 0, 1)  # N=0, A+P-/A+P+=1

#     print(f"{modality} - Classes: {classes}, Subjects: {len(np.unique(sub_ids))}, Samples: {len(X)}")
#     print(f"{modality} - Label distribution after mapping: {np.bincount(y)}")

    return X, y, sub_ids


import pandas as pd
import numpy as np
import re

def load_psychometric_data(participants_tsv_path):
    """
    Load psychological/cognitive features from participants.tsv file.
    
    Args:
        participants_tsv_path: Path to the participants.tsv file
    
    Returns:
        Dictionary with subject IDs as keys and dict containing features and labels as values
    """
    
    # Define target psychological/cognitive columns
    target_columns = [
        'age', 'sex', 'education', 'BMI',
        'BDI', 'SES', 'RPM', 'EHI', 'NEO_NEU', 'NEO_EXT', 'NEO_OPE', 'NEO_AGR', 'NEO_CON', 
        'AUDIT', 'MINI-COPE_1', 'MINI-COPE_2', 'MINI-COPE_3', 'MINI-COPE_4', 'MINI-COPE_5', 
        'MINI-COPE_6', 'MINI-COPE_7', 'MINI-COPE_8', 'MINI-COPE_9', 'MINI-COPE_10', 
        'MINI-COPE_11', 'MINI-COPE_12', 'MINI-COPE_13', 'MINI-COPE_14', 
        'CVLT_1', 'CVLT_2', 'CVLT_3', 'CVLT_4', 'CVLT_5', 'CVLT_6', 'CVLT_7', 
        'CVLT_8', 'CVLT_9', 'CVLT_10', 'CVLT_11', 'CVLT_12', 'CVLT_13', "leukocytes",
        'erythrocytes', 'hemoglobin', 'hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW-CV',
        'platelets', 'PDW', 'MPV', 'P-LCR',  'neutrophils_%', 'lymphocytes_%',
        'monocytes_%', 'eosinophils_%', 'basophils_%', 'total_cholesterol',
        'cholesterol_HDL', 'non-HDL_cholesterol', 'LDL_cholesterol', 
        'triglycerides', 'HSV_r', 
        'learning_deficits', 'allergies', 'drugs', 'ibuprofen_intake',
        'thyroid_diseases', 'hypertension', 'diabetes', 'other_diseases',
        'smoking_status', 'coffee_status', 'dementia_history_parents'
    ]
    # 'neutrophils', 'lymphocytes',
        # 'monocytes', 'eosinophils', 'basophils',

    # learning_deficits	BMI	allergies	drugs	ibuprofen_intake	thyroid_diseases	hypertension	diabetes	other_diseases	smoking_status	coffee_status	dementia_history_parents

        # leukocytes	erythrocytes	hemoglobin	hematocrit	MCV	MCH	MCHC	RDW-CV	platelets	PDW	MPV	P-LCR	neutrophils	lymphocytes	monocytes	eosinophils	basophils	neutrophils_%	lymphocytes_%	monocytes_%	eosinophils_%	basophils_%	total_cholesterol	cholesterol_HDL	non-HDL_cholesterol	LDL_cholesterol	triglycerides	HSV_r
    
    
    if not os.path.exists(participants_tsv_path):
        raise FileNotFoundError(f"Participants file not found at: {participants_tsv_path}")
    
    print(f"Loading psychometric data from: {participants_tsv_path}")
    
    try:
        # Load the TSV file
        df_participants = pd.read_csv(participants_tsv_path, sep='\t')
        
        # Check for required columns
        if 'participant_id' not in df_participants.columns:
            raise ValueError("'participant_id' column not found in participants file")
        if 'Group' not in df_participants.columns:
            raise ValueError("'Group' column not found in participants file")
        
        # Check which target columns are available
        available_columns = [col for col in target_columns if col in df_participants.columns]
        missing_columns = [col for col in target_columns if col not in df_participants.columns]
        
        print(f"Available psychometric columns ({len(available_columns)}): {available_columns}")
        if missing_columns:
            print(f"Missing columns ({len(missing_columns)}): {missing_columns}")
        
        # Check unique groups/labels
        unique_groups = df_participants['Group'].unique()
        print(f"Found groups: {unique_groups}")
        
        # Create label mapping
        label_mapping = {'N': 0, 'A+P-': 1, 'A+P+': 1}  # Binary classification: N=0, A+P- and A+P+ = 1
        # Alternative mapping for 3-class: {'N': 0, 'A+P-': 1, 'A+P+': 2}
        
        psychometric_data_dict = {}
        subjects_processed = 0
        subjects_skipped = 0
        
        # Process each participant
        for _, row in df_participants.iterrows():
            participant_id = row.get('participant_id', '')
            group_label = row.get('Group', '')
            
            # Parse subject ID to standardized format
            subject_id = parse_subject_id_from_participants(participant_id)
            
            if not subject_id:
                print(f"Warning: Could not parse subject ID from '{participant_id}'")
                subjects_skipped += 1
                continue
            
            # Map group label to numeric
            if group_label not in label_mapping:
                print(f"Warning: Unknown group '{group_label}' for subject {subject_id}")
                subjects_skipped += 1
                continue
            
            numeric_label = label_mapping[group_label]
            
            # Extract available features for this subject
            participant_features = []
            feature_names = []
            
            for col in available_columns:
                value = row.get(col, np.nan)
                feature_names.append(col)
                
                # Handle missing values and different data types
                if pd.isna(value) or value == '' or value == 'n/a' or str(value).strip() == '':
                    participant_features.append(0.0)  # Zero imputation for missing values
                else:
                    try:
                        # Handle special cases
                        if isinstance(value, str):
                            # Handle arrays in string format like "[0. 1 ]"
                            if '[' in value and ']' in value:
                                # Extract first number from array string
                                numbers = re.findall(r'[-+]?\d*\.?\d+', value)
                                if numbers:
                                    participant_features.append(float(numbers[0]))
                                else:
                                    participant_features.append(0.0)
                            else:
                                participant_features.append(float(value))
                        else:
                            participant_features.append(float(value))
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert value '{value}' for column '{col}' in subject {subject_id}. Using 0.0")
                        participant_features.append(0.0)
            
            # Store the processed data
            psychometric_data_dict[subject_id] = {
                "features": np.array(participant_features, dtype=np.float32),
                "label": numeric_label,
                "group_name": group_label,
                "feature_names": feature_names
            }
            
            subjects_processed += 1
        
        # Print summary statistics
        print(f"\n=== Psychometric Data Loading Summary ===")
        print(f"Total subjects processed: {subjects_processed}")
        print(f"Subjects skipped: {subjects_skipped}")
        print(f"Features per subject: {len(available_columns)}")
        
        # Print class distribution
        class_counts = {}
        for subject_data in psychometric_data_dict.values():
            group_name = subject_data['group_name']
            class_counts[group_name] = class_counts.get(group_name, 0) + 1
        
        print(f"Class distribution:")
        for group_name, count in class_counts.items():
            numeric_label = label_mapping.get(group_name, -1)
            print(f"  {group_name} (label={numeric_label}): {count} subjects")
        
        return psychometric_data_dict
        
    except Exception as e:
        print(f"Error loading participants file: {e}")
        raise



def parse_subject_id_from_participants(participant_id):
    """
    Parse subject ID from participants.tsv format (e.g., 'sub-01' -> '01')
    """
    if pd.isna(participant_id) or participant_id == '':
        return None
    
    match = re.search(r'sub-(\d+)', str(participant_id))
    if match:
        return match.group(1)  # Return the number part (e.g., '01', '02', etc.)
    return None





def filter_common_subjects(*data_dicts, verbose=True):
    """
    Filter subjects that exist in ALL provided data dictionaries.
    
    Args:
        *data_dicts: Variable number of data dictionaries from different modalities
        verbose: Whether to print detailed information
    
    Returns:
        tuple: Filtered dictionaries in the same order as input, plus a set of common subject IDs
    """
    if len(data_dicts) < 2:
        raise ValueError("Need at least 2 data dictionaries to find common subjects")
    
    # Extract subject IDs from each dictionary
    subject_sets = []
    modality_names = []
    
    for i, data_dict in enumerate(data_dicts):
        # Filter out label keys for EEG/fMRI data
        subject_ids = set()
        for key in data_dict.keys():
            if not key.endswith('_label'):
                subject_ids.add(key)
        
        subject_sets.append(subject_ids)
        modality_names.append(f"Modality_{i+1}")
    
    if verbose:
        print(f"=== Subject Filtering Analysis ===")
        for i, (subject_set, name) in enumerate(zip(subject_sets, modality_names)):
            print(f"{name}: {len(subject_set)} subjects - {sorted(list(subject_set))}")
    
    # Find intersection of all sets
    common_subjects = subject_sets[0]
    for subject_set in subject_sets[1:]:
        common_subjects = common_subjects.intersection(subject_set)
    
    if verbose:
        print(f"\nCommon subjects across all modalities: {len(common_subjects)}")
        print(f"Common subject IDs: {sorted(list(common_subjects))}")
        
        # Show what's missing from each modality
        for i, (subject_set, name) in enumerate(zip(subject_sets, modality_names)):
            missing = sorted(list(common_subjects - subject_set))
            extra = sorted(list(subject_set - common_subjects))
            if missing:
                print(f"{name} missing: {missing}")
            if extra:
                print(f"{name} extra: {extra}")
    
    # Filter each dictionary to keep only common subjects
    filtered_dicts = []
    for data_dict in data_dicts:
        filtered_dict = {}
        
        for subject_id in common_subjects:
            # Copy subject data
            if subject_id in data_dict:
                filtered_dict[subject_id] = data_dict[subject_id]
            
            # Copy label data for EEG/fMRI
            label_key = subject_id + '_label'
            if label_key in data_dict:
                filtered_dict[label_key] = data_dict[label_key]
        
        filtered_dicts.append(filtered_dict)
    
    if verbose:
        print(f"\n=== Filtering Results ===")
        for i, filtered_dict in enumerate(filtered_dicts):
            # Count actual subjects (exclude label keys)
            actual_subjects = len([k for k in filtered_dict.keys() if not k.endswith('_label')])
            print(f"Modality_{i+1}: {actual_subjects} subjects retained")
    
    return tuple(filtered_dicts) + (common_subjects,)

def load_and_filter_all_modalities(eeg_dir, fmri_dir, smri_dir, psychometric_tsv_path, verbose=True):
    
    if verbose:
        print("Loading all modality data...")
    
    # Load all modalities
    eeg_data_dict = load_eeg_data(eeg_dir)
    fmri_data_dict = load_fmri_data(fmri_dir)  
    smri_data_dict = load_sMRI_data(smri_dir)
    psychometric_data_dict = load_psychometric_data(psychometric_tsv_path)
    
    if verbose:
        print(f"\nOriginal data loaded:")
        print(f"EEG: {len([k for k in eeg_data_dict.keys() if not k.endswith('_label')])} subjects")
        print(f"fMRI: {len([k for k in fmri_data_dict.keys() if not k.endswith('_label')])} subjects") 
        print(f"sMRI: {len(smri_data_dict)} subjects")
        print(f"Psychometric: {len(psychometric_data_dict)} subjects")
    
    # Filter to keep only common subjects
    filtered_eeg, filtered_fmri, filtered_smri, filtered_psychometric, common_subjects = filter_common_subjects(
        eeg_data_dict, fmri_data_dict, smri_data_dict, psychometric_data_dict, verbose=verbose
    )
    
    if verbose:
        print(f"\n=== Final Filtered Results ===")
        print(f"Subjects with ALL four modalities: {len(common_subjects)}")
        print(f"Subject IDs: {sorted(list(common_subjects))}")
        
        # Verify consistency
        eeg_subjects = len([k for k in filtered_eeg.keys() if not k.endswith('_label')])
        fmri_subjects = len([k for k in filtered_fmri.keys() if not k.endswith('_label')])
        smri_subjects = len(filtered_smri)
        psychometric_subjects = len(filtered_psychometric)
        
        print(f"Verification - EEG: {eeg_subjects}, fMRI: {fmri_subjects}, sMRI: {smri_subjects}, Psychometric: {psychometric_subjects}")
        
        if not (eeg_subjects == fmri_subjects == smri_subjects == psychometric_subjects == len(common_subjects)):
            print("⚠️  WARNING: Subject counts don't match after filtering!")
    
    return filtered_eeg, filtered_fmri, filtered_smri, filtered_psychometric, common_subjects

# Convenience function for your specific use case
def load_multimodal_data(classes=("N", "A+P+"), verbose=True):
    """
    Load and filter all modalities, then extract features for specific classes.
    
    Args:
        classes: Tuple of class labels to include (e.g., ("N", "A+P+"))
        verbose: Whether to print detailed information
    
    Returns:
        dict: Contains X, y, sub_ids for each modality plus common subjects
    """
    
    # Define paths (update these as needed)
    eeg_dir = EEG_DATA_DIR
    fmri_dir = FMRI_DATA_DIR
    smri_dir = SMRI_DATA_DIR
    psychometric_tsv = "filtered_participants.tsv"  # Update this path
    
    # Load and filter all modalities
    filtered_eeg, filtered_fmri, filtered_smri, filtered_psychometric, common_subjects = load_and_filter_all_modalities(
        eeg_dir, fmri_dir, smri_dir, psychometric_tsv, verbose=verbose
    )

    print(f"\nExtracting features for classes: {classes}")
    print(f"Total common subjects: {len(common_subjects)}")
    print(filtered_eeg.keys())
    print(filtered_fmri.keys())
    print(filtered_smri.keys())
    print(filtered_psychometric.keys())
    
    # Extract features and labels for each modality
    X_eeg, y_eeg, sub_ids_eeg = get_X_y(filtered_eeg, common_subjects, modality="eeg", classes=classes)
    X_fmri, y_fmri, sub_ids_fmri = get_X_y(filtered_fmri, common_subjects, modality="fMRI", classes=classes)
    X_smri, y_smri, sub_ids_smri = get_X_y(filtered_smri, common_subjects, modality="sMRI", classes=classes)
    
    # For psychometric data, we need a custom extraction since it doesn't follow the same pattern
    X_psychometric, y_psychometric, sub_ids_psychometric = extract_psychometric_features(
        filtered_psychometric, common_subjects, classes=classes
    )
    
    if verbose:
        print(f"\n=== Final Feature Extraction Results ===")
        print(f"Classes selected: {classes}")
        print(f"EEG: {X_eeg.shape} features, {len(np.unique(sub_ids_eeg))} subjects, {len(y_eeg)} samples")
        print(f"fMRI: {X_fmri.shape} features, {len(np.unique(sub_ids_fmri))} subjects, {len(y_fmri)} samples")
        print(f"sMRI: {X_smri.shape} features, {len(np.unique(sub_ids_smri))} subjects, {len(y_smri)} samples")
        print(f"Psychometric: {X_psychometric.shape} features, {len(np.unique(sub_ids_psychometric))} subjects, {len(y_psychometric)} samples")
    
    return {
        'eeg': {'X': X_eeg, 'y': y_eeg, 'sub_ids': sub_ids_eeg},
        'fmri': {'X': X_fmri, 'y': y_fmri, 'sub_ids': sub_ids_fmri}, 
        'smri': {'X': X_smri, 'y': y_smri, 'sub_ids': sub_ids_smri},
        'psychometric': {'X': X_psychometric, 'y': y_psychometric, 'sub_ids': sub_ids_psychometric},
        'common_subjects': common_subjects
    }






def extract_psychometric_features(psychometric_data_dict, common_subjects, classes=("N", "A+P+")):
    """
    Extract features from psychometric data for specific classes using group names.
    
    Args:
        psychometric_data_dict: Dictionary from load_psychometric_data()
        common_subjects: Set of subject IDs to include (MUST be filtered to common subjects only)
        classes: Tuple of group name strings (e.g., ("N", "A+P+"))
    
    Returns:
        tuple: (X, y, sub_ids) arrays
    """
    
    X, y, sub_ids = [], [], []
    
    # Create proper label mapping based on what you're selecting
    if len(classes) == 2:
        label_mapping = {classes[0]: 0, classes[1]: 1}
    else:
        # For 3-class case
        label_mapping = {"N": 0, "A+P-": 1, "A+P+": 1}

    # if classes == "all : no mapping
    if classes == ("all")  :
        label_mapping = {"N": 0, "A+P-": 1, "A+P+": 2}
    
    print(f"Psychometric label mapping: {label_mapping}")
    
    # IMPORTANT: Only iterate through common_subjects, not all subjects in psychometric_data_dict
    for subject_id in common_subjects:
        if subject_id in psychometric_data_dict:
            subject_data = psychometric_data_dict[subject_id]
            group_name = subject_data["group_name"]  # Use group_name instead of numeric label
            
            # Check if the group_name is in our target classes, or if we are loading all classes
            if classes == ("all") or group_name in classes:
                mapped_label = label_mapping[group_name]
                X.append(subject_data["features"])
                y.append(mapped_label)
                sub_ids.append(subject_id)
    
    if len(X) == 0:
        # Debug information
        print(f"DEBUG: Classes requested: {classes}")
        print(f"DEBUG: Label mapping: {label_mapping}")
        print(f"DEBUG: Common subjects count: {len(common_subjects)}")
        print(f"DEBUG: Psychometric dict has subjects: {len(psychometric_data_dict)}")
        
        # Check what group names are available in common subjects
        available_groups = []
        subjects_in_psychometric = []
        
        for subject_id in common_subjects:
            if subject_id in psychometric_data_dict:
                subject_data = psychometric_data_dict[subject_id]
                available_groups.append(subject_data['group_name'])
                subjects_in_psychometric.append(subject_id)
        
        print(f"DEBUG: Common subjects in psychometric data: {len(subjects_in_psychometric)}")
        print(f"DEBUG: Available group names in common subjects: {set(available_groups)}")
        print(f"DEBUG: Subject IDs in common subjects: {sorted(subjects_in_psychometric[:10])}...")  # Show first 10
        
        raise ValueError(f"No subjects found with group names {classes}. Available groups: {set(available_groups)}")
    
    X = np.array(X)
    y = np.array(y)
    sub_ids = np.array(sub_ids)
    
    print(f"Psychometric - Classes: {classes}, Label mapping: {label_mapping}")
    print(f"Loaded {len(X)} subjects")
    print(f"Feature shape: {X.shape}")
    print(f"Labels: {y}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Subjects: {len(np.unique(sub_ids))}, Samples: {len(y)}")
    
    return X, y, sub_ids




import pandas as pd
import numpy as np
import os

def analyze_common_subject_data():
    """
    Finds subjects common to all modalities and calculates demographic statistics
    for that specific cohort.
    """
    # --- Configuration ---
    # These paths should match the configuration in your utils.py
    eeg_dir = os.path.join("data", "EEG_data", "EEG_data_plv")
    fmri_dir = os.path.join("data", "fMRI_data")
    smri_dir = os.path.join("data", "MRI_data")
    participants_tsv_path = "filtered_participants.tsv"

    print("--- Finding Common Subjects Across All Modalities ---")
    
    try:
        # Use the function from utils.py to get the list of common subject IDs
        # We set verbose=False to keep the output clean for this analysis.
        _, _, _, _, common_subjects_ids = load_and_filter_all_modalities(
            eeg_dir, fmri_dir, smri_dir, participants_tsv_path, verbose=False
        )

        if not common_subjects_ids:
            print("No common subjects found across all modalities. Exiting.")
            return

        print(f"\nFound {len(common_subjects_ids)} subjects common to all modalities.")
        
        # Load the participants file to get demographic data
        df = pd.read_csv(participants_tsv_path, sep='\t')

        # The common_subjects_ids are like '01', '02'. The tsv has 'sub-01', 'sub-02'.
        # We need to format them for filtering.
        formatted_ids_to_keep = ['sub-' + str(id).zfill(2) for id in common_subjects_ids]
        
        # Filter the DataFrame to keep only the common subjects
        df_common = df[df['participant_id'].isin(formatted_ids_to_keep)].copy()
        
        print(f"Filtered participants data to {len(df_common)} subjects.")
        
        print("\n--- Analysis of Common Subjects by Group ---")

        # Ensure data types are correct for calculations
        if 'age' in df_common.columns:
            df_common['age'] = pd.to_numeric(df_common['age'], errors='coerce')
        if 'sex' in df_common.columns:
            # We will use value_counts, so original values ('M'/'F' or 0/1) are fine.
            pass

        if 'Group' in df_common.columns:
            # Group by the 'Group' column
            grouped = df_common.groupby('Group')

            # Iterate over each group and print its specific statistics
            for group_name, group_df in grouped:
                print(f"\n--- Group: {group_name} ({len(group_df)} subjects) ---")

                # Age statistics for the current group
                if 'age' in group_df.columns and not group_df['age'].isnull().all():
                    age_mean = group_df['age'].mean()
                    age_std = group_df['age'].std()
                    print("  Age Statistics:")
                    print(f"    - Mean: {age_mean:.2f}")
                    print(f"    - Std Dev: {age_std:.2f}")
                else:
                    print("  Age Statistics: Not available.")

                # Sex distribution for the current group
                if 'sex' in group_df.columns:
                    sex_counts = group_df['sex'].value_counts()
                    print("  Sex Distribution:")
                    # Assuming 'sex' column contains values like 'M'/'F' or 0/1.
                    # The output will show counts for each unique value found.
                    for sex_value, count in sex_counts.items():
                        print(f"    - Count of '{sex_value}': {count}")
                else:
                    print("  Sex Distribution: 'sex' column not found.")
        else:
            print("\n'Group' column not found in the participants file.")
            
        print("\n--- End of Analysis ---")


    except FileNotFoundError as e:
        print(f"\nERROR: A data file was not found.")
        print(f"Details: {e}")
        print("Please ensure all data paths are correct in both this script and utils.py.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == '__main__':
    analyze_common_subject_data()


# # Example usage:
# data = load_multimodal_data(classes=("N", "A+P+"), verbose=True)



