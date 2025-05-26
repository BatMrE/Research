import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate, shift, zoom
from skimage.transform import warp, AffineTransform
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
import kornia  # Added this import
import warnings
import gc
warnings.filterwarnings('ignore')

# Path constants
TRAINING_PATH = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
VALIDATION_PATH = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
OUTPUT_PATH = "/kaggle/working/tumour_augmented/traditional_augmentation"
os.makedirs(OUTPUT_PATH, exist_ok=True)

class BrainTumorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).float().permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_nifti(file_path):
    try:
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata(), nifti_img.affine
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_tumor_regions(training_folder, max_subjects=50):
    tumor_regions = []
    subject_ids = []
    
    training_subjects = [d for d in os.listdir(training_folder) 
                       if os.path.isdir(os.path.join(training_folder, d))]
    training_subjects = training_subjects[:min(len(training_subjects), max_subjects)]
    
    for subject in tqdm(training_subjects, desc="Extracting tumors"):
        subject_folder = os.path.join(training_folder, subject)
        flair_file = os.path.join(subject_folder, f"{subject}_flair.nii")
        t1_file = os.path.join(subject_folder, f"{subject}_t1.nii")
        seg_file = os.path.join(subject_folder, f"{subject}_seg.nii")
        
        if os.path.exists(flair_file) and os.path.exists(t1_file) and os.path.exists(seg_file):
            flair_data, flair_affine = load_nifti(flair_file)
            t1_data, t1_affine = load_nifti(t1_file)
            seg_data, _ = load_nifti(seg_file)
            
            if flair_data is None or t1_data is None or seg_data is None:
                continue
            
            whole_tumor_mask = seg_data > 0
            
            if np.sum(whole_tumor_mask) > 100:
                tumor_regions.append({
                    'flair': flair_data,
                    't1': t1_data,
                    'mask': whole_tumor_mask,
                    'seg': seg_data
                })
                subject_ids.append(subject)
    
    return tumor_regions, subject_ids

def extract_tumor_patches(tumor_regions, patch_size=64, min_tumor_percentage=0.1):
    tumor_patches = []
    
    for subject_data in tqdm(tumor_regions, desc="Extracting tumor patches"):
        # Process one modality at a time
        flair = subject_data['flair']
        t1 = subject_data['t1']
        mask = subject_data['mask']
        seg = subject_data['seg']
        
        # Process slice by slice to reduce memory
        for z in range(mask.shape[2]):
            slice_mask = mask[:, :, z]
            if np.sum(slice_mask) > (patch_size * patch_size * min_tumor_percentage):
                # Process each slice separately
                flair_slice = flair[:, :, z].copy()
                t1_slice = t1[:, :, z].copy()
                seg_slice = seg[:, :, z].copy()
                
                # Process regions
                labeled_mask = label(slice_mask)
                props = regionprops(labeled_mask)
                
                for prop in props:
                    if prop.area > (patch_size * patch_size * min_tumor_percentage):
                        # Extract patch
                        y_min, x_min, y_max, x_max = prop.bbox
                        pad = patch_size // 2
                        y_center = (y_min + y_max) // 2
                        x_center = (x_min + x_max) // 2
                        
                        # Extract patch with bounds checking
                        y_start = max(0, y_center - pad)
                        y_end = min(flair_slice.shape[0], y_center + pad)
                        x_start = max(0, x_center - pad)
                        x_end = min(flair_slice.shape[1], x_center + pad)
                        
                        if (y_end - y_start) >= patch_size//2 and (x_end - x_start) >= patch_size//2:
                            # Extract and resize if needed
                            flair_patch = flair_slice[y_start:y_end, x_start:x_end]
                            if flair_patch.shape[0] != patch_size or flair_patch.shape[1] != patch_size:
                                flair_patch = cv2.resize(flair_patch, (patch_size, patch_size))
                            
                            tumor_patches.append({
                                'flair': flair_patch,
                                'mask': (slice_mask[y_start:y_end, x_start:x_end] > 0.5).astype(np.float32)
                            })
                
                # Clear memory
                del flair_slice, t1_slice, seg_slice
                gc.collect()
    
    return tumor_patches

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

def horizontal_flip(image):
    return np.fliplr(image)

def vertical_flip(image):
    return np.flipud(image)

def random_rotation(image, mask=None, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = rotate(image, angle, reshape=False, mode='nearest', order=1)
    
    if mask is not None:
        rotated_mask = rotate(mask, angle, reshape=False, mode='nearest', order=0)
        return rotated, rotated_mask
    
    return rotated

def random_shift(image, mask=None, max_shift=10):
    dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)
    shifted = shift(image, (dy, dx), mode='nearest')
    
    if mask is not None:
        shifted_mask = shift(mask, (dy, dx), mode='constant', cval=0)
        return shifted, shifted_mask
    
    return shifted

def random_zoom(image, mask=None, max_factor=0.1):
    factor = np.random.uniform(1.0 - max_factor, 1.0 + max_factor)
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    zoomed = zoom(image, factor, order=1)
    
    if factor > 1:
        start_h = (zoomed.shape[0] - h) // 2
        start_w = (zoomed.shape[1] - w) // 2
        zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h = (h - zoomed.shape[0]) // 2
        pad_w = (w - zoomed.shape[1]) // 2
        zoomed_padded = np.zeros_like(image)
        zoomed_padded[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
        zoomed = zoomed_padded
    
    if mask is not None:
        if factor > 1:
            zoomed_mask = zoom(mask, factor, order=0)
            zoomed_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w]
        else:
            zoomed_mask = zoom(mask, factor, order=0)
            mask_padded = np.zeros_like(mask)
            mask_padded[pad_h:pad_h+zoomed_mask.shape[0], pad_w:pad_w+zoomed_mask.shape[1]] = zoomed_mask
            zoomed_mask = mask_padded
        
        return zoomed, zoomed_mask
    
    return zoomed

def random_skew(image, mask=None, intensity=0.1):
    affine_tf = AffineTransform(
        shear=np.random.uniform(-intensity, intensity)
    )
    
    skewed = warp(image, affine_tf, mode='edge', preserve_range=True)
    
    if mask is not None:
        skewed_mask = warp(mask, affine_tf, mode='constant', order=0, preserve_range=True)
        return skewed.astype(image.dtype), skewed_mask.astype(mask.dtype)
    
    return skewed.astype(image.dtype)

def random_intensity(image, factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*factor_range)
    adjusted = image * factor
    return np.clip(adjusted, 0, 1)

def patch_tumor_transfer(healthy_image, tumor_patch, tumor_mask):
    h, w = healthy_image.shape
    ph, pw = tumor_patch.shape
    
    y_pos = np.random.randint(0, h - ph + 1)
    x_pos = np.random.randint(0, w - pw + 1)
    
    augmented = healthy_image.copy()
    blend_mask = tumor_mask.astype(float)
    
    region = augmented[y_pos:y_pos+ph, x_pos:x_pos+pw]
    blended = region * (1 - blend_mask) + tumor_patch * blend_mask
    
    augmented[y_pos:y_pos+ph, x_pos:x_pos+pw] = blended
    
    tumor_transfer_mask = np.zeros_like(healthy_image, dtype=bool)
    tumor_transfer_mask[y_pos:y_pos+ph, x_pos:x_pos+pw] = tumor_mask
    
    return augmented, tumor_transfer_mask

def calculate_metrics(y_true, y_pred, masks):
    """Calculate SSIM, PSNR and SAS metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    y_true_tensor = torch.from_numpy(y_true).float().permute(0, 3, 1, 2).to(device)
    y_pred_tensor = torch.from_numpy(y_pred).float().permute(0, 3, 1, 2).to(device)
    masks_tensor = torch.from_numpy(masks).float().unsqueeze(1).to(device)
    
    ssim_scores = []
    psnr_scores = []
    sas_scores = []
    
    with torch.no_grad():
        for i in range(y_true_tensor.shape[0]):
            true_img = y_true_tensor[i:i+1]
            pred_img = y_pred_tensor[i:i+1]
            mask = masks_tensor[i:i+1]
            
            ssim = kornia.metrics.ssim(true_img, pred_img, window_size=5).mean()
            psnr = kornia.metrics.psnr(true_img, pred_img, max_val=1.0)
            
            tumor_ssim = kornia.metrics.ssim(
                true_img * mask,
                pred_img * mask,
                window_size=5
            )
            
            healthy_ssim = kornia.metrics.ssim(
                true_img * (1 - mask),
                pred_img * (1 - mask),
                window_size=5
            )
            
            real_edges = kornia.filters.sobel(true_img)
            pred_edges = kornia.filters.sobel(pred_img)
            edge_sim = 1 - F.l1_loss(real_edges, pred_edges)
            
            sas_score = 0.4 * tumor_ssim.mean() + 0.3 * healthy_ssim.mean() + 0.3 * edge_sim
            
            ssim_scores.append(ssim.item())
            psnr_scores.append(psnr.item())
            sas_scores.append(sas_score.item())
    
    return {
        'ssim': np.mean(ssim_scores),
        'psnr': np.mean(psnr_scores),
        'sas': np.mean(sas_scores)
    }

def augment_dataset(tumor_patches, validation_folder, num_samples=500, patch_size=64):
    X_augmented = []
    y_augmented = []
    masks_augmented = []
    
    # Process healthy slices in batches
    healthy_slices = []
    validation_subjects = [d for d in os.listdir(validation_folder) 
                         if os.path.isdir(os.path.join(validation_folder, d))][:10]  # Limit subjects
    
    for subject in tqdm(validation_subjects, desc="Processing healthy slices"):
        subject_folder = os.path.join(validation_folder, subject)
        flair_file = os.path.join(subject_folder, f"{subject}_flair.nii")
        
        if os.path.exists(flair_file):
            flair_data, _ = load_nifti(flair_file)
            if flair_data is not None:
                # Process only every 5th slice to reduce memory
                for z in range(0, flair_data.shape[2], 5):
                    flair_slice = normalize_image(flair_data[:, :, z])
                    if np.mean(flair_slice) > 0.01:
                        healthy_slices.append(flair_slice)
                
                del flair_data
                gc.collect()
    
    # Generate augmented samples with memory management
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        if not healthy_slices or not tumor_patches:
            break
            
        # Select random healthy slice
        healthy_idx = np.random.randint(0, len(healthy_slices))
        healthy_flair = healthy_slices[healthy_idx]
        
        # Select random tumor patch
        tumor_idx = np.random.randint(0, len(tumor_patches))
        tumor_patch = tumor_patches[tumor_idx]
        
        # Apply augmentation
        augmented_flair, transferred_mask = patch_tumor_transfer(
            healthy_flair, 
            tumor_patch['flair'], 
            tumor_patch['mask']
        )
        
        # Store results
        X_augmented.append(np.stack([
            healthy_flair, 
            np.zeros_like(healthy_flair),  # Placeholder for T1
            transferred_mask
        ], axis=-1))
        y_augmented.append(augmented_flair[..., np.newaxis])
        masks_augmented.append(transferred_mask)
        
        # Clear memory periodically
        if len(X_augmented) % 100 == 0:
            gc.collect()
    
    return np.array(X_augmented), np.array(y_augmented), np.array(masks_augmented)
    
    # print("\nAugmentation Quality Metrics:")
    # print(f"SSIM: {metrics['ssim']:.4f}")
    # print(f"PSNR: {metrics['psnr']:.2f}")
    # print(f"SAS: {metrics['sas']:.4f}")
    
    # return np.array(X_augmented), np.array(y_augmented), np.array(masks_augmented)

def visualize_augmentations(X, y, masks, num_samples=5):
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(X[idx, :, :, 0], cmap='gray')
        axes[i, 0].set_title('Healthy FLAIR')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(X[idx, :, :, 1], cmap='gray')
        axes[i, 1].set_title('Healthy T1')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(X[idx, :, :, 2], cmap='gray')
        axes[i, 2].set_title('Tumor Mask')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(y[idx, :, :, 0], cmap='gray')
        axes[i, 3].set_title('Augmented FLAIR')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'augmentation_samples.png'))
    plt.close()


print("Extracting tumor regions from training data...")
tumor_regions, _ = extract_tumor_regions(TRAINING_PATH, max_subjects=2)

print("Extracting tumor patches...")
tumor_patches = extract_tumor_patches(tumor_regions, patch_size=64)

print("Creating augmented dataset with traditional methods...")
X_augmented, y_augmented, masks_augmented = augment_dataset(
    tumor_patches, VALIDATION_PATH, num_samples=500
)

print("Visualizing augmentations...")
visualize_augmentations(X_augmented, y_augmented, masks_augmented)

np.save(os.path.join(OUTPUT_PATH, 'X_augmented_samples.npy'), X_augmented[:50])
np.save(os.path.join(OUTPUT_PATH, 'y_augmented_samples.npy'), y_augmented[:50])

print(f"Final dataset shape: X={X_augmented.shape}, y={y_augmented.shape}")
print(f"Results saved to {OUTPUT_PATH}")

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42
)

train_dataset = BrainTumorDataset(X_train, y_train)
val_dataset = BrainTumorDataset(X_val, y_val)

print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
