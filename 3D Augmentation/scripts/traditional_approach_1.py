import os
import numpy as np
import torch
import torch.nn as nn
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
import warnings
warnings.filterwarnings('ignore')

# Path constants
TRAINING_PATH = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
VALIDATION_PATH = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
OUTPUT_PATH = "/kaggle/working/tumour_augmented/traditional_augmentation"
os.makedirs(OUTPUT_PATH, exist_ok=True)

class BrainTumorDataset(Dataset):
    def __init__(self, X, y):
        # Convert to PyTorch tensors and permute to [N,C,H,W]
        self.X = torch.from_numpy(X).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).float().permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_nifti(file_path):
    """Load a NIfTI file and return data and affine"""
    try:
        nifti_img = nib.load(file_path)
        return nifti_img.get_fdata(), nifti_img.affine
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_tumor_regions(training_folder, max_subjects=50):
    """Extract tumor regions from training data"""
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
            
            # Extract different tumor regions based on segmentation labels
            # In BraTS20: 1=necrotic core, 2=edema, 4=enhancing tumor
            whole_tumor_mask = seg_data > 0  # All tumor regions
            
            # Only include subjects with substantial tumor regions
            if np.sum(whole_tumor_mask) > 100:  # Min voxel count threshold  
                # Store both image data and mask
                tumor_regions.append({
                    'flair': flair_data,
                    't1': t1_data,
                    'mask': whole_tumor_mask,
                    'seg': seg_data
                })
                subject_ids.append(subject)
    
    return tumor_regions, subject_ids

def extract_tumor_patches(tumor_regions, patch_size=64, min_tumor_percentage=0.1):
    """Extract patches containing tumors from the tumor regions"""
    tumor_patches = []
    
    for subject_data in tqdm(tumor_regions, desc="Extracting tumor patches"):
        flair = subject_data['flair']
        t1 = subject_data['t1'] 
        mask = subject_data['mask']
        seg = subject_data['seg']
        
        # Find tumor slices with substantial tumor area
        for z in range(mask.shape[2]):
            slice_mask = mask[:, :, z]
            if np.sum(slice_mask) > (patch_size * patch_size * min_tumor_percentage):
                # Get slice data
                flair_slice = flair[:, :, z]
                t1_slice = t1[:, :, z]
                seg_slice = seg[:, :, z]
                
                # Find tumor regions and extract patches
                labeled_mask = label(slice_mask)
                props = regionprops(labeled_mask)
                
                for prop in props:
                    if prop.area > (patch_size * patch_size * min_tumor_percentage):
                        # Get bounding box with some margin
                        y_min, x_min, y_max, x_max = prop.bbox
                        
                        # Add padding for patch extraction
                        pad = patch_size // 2
                        y_center = (y_min + y_max) // 2
                        x_center = (x_min + x_max) // 2
                        
                        # Extract patch ensuring we don't go out of bounds
                        y_start = max(0, y_center - pad)
                        y_end = min(flair_slice.shape[0], y_center + pad)
                        x_start = max(0, x_center - pad)
                        x_end = min(flair_slice.shape[1], x_center + pad)
                        
                        # Skip if patch is too small
                        if (y_end - y_start) < patch_size or (x_end - x_start) < patch_size:
                            continue
                        
                        # Extract patches
                        flair_patch = flair_slice[y_start:y_end, x_start:x_end]
                        t1_patch = t1_slice[y_start:y_end, x_start:x_end]
                        mask_patch = slice_mask[y_start:y_end, x_start:x_end]
                        seg_patch = seg_slice[y_start:y_end, x_start:x_end]
                        
                        # Resize if necessary
                        if flair_patch.shape[0] != patch_size or flair_patch.shape[1] != patch_size:
                            flair_patch = cv2.resize(flair_patch, (patch_size, patch_size))
                            t1_patch = cv2.resize(t1_patch, (patch_size, patch_size))
                            mask_patch = cv2.resize(mask_patch, (patch_size, patch_size)) > 0.5
                            seg_patch = cv2.resize(seg_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
                        
                        # Store patches
                        tumor_patches.append({
                            'flair': flair_patch,
                            't1': t1_patch,
                            'mask': mask_patch,
                            'seg': seg_patch
                        })
    
    print(f"Extracted {len(tumor_patches)} tumor patches")
    return tumor_patches

def normalize_image(image):
    """Normalize image to 0-1 range"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image

# Traditional augmentation functions
def horizontal_flip(image):
    """Horizontal flip augmentation"""
    return np.fliplr(image)

def vertical_flip(image):
    """Vertical flip augmentation"""
    return np.flipud(image)

def random_rotation(image, mask=None, max_angle=15):
    """Random rotation augmentation"""
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = rotate(image, angle, reshape=False, mode='nearest', order=1)
    
    if mask is not None:
        rotated_mask = rotate(mask, angle, reshape=False, mode='nearest', order=0)
        return rotated, rotated_mask
    
    return rotated

def random_shift(image, mask=None, max_shift=10):
    """Random shift augmentation"""
    dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)
    shifted = shift(image, (dy, dx), mode='nearest')
    
    if mask is not None:
        shifted_mask = shift(mask, (dy, dx), mode='constant', cval=0)
        return shifted, shifted_mask
    
    return shifted

def random_zoom(image, mask=None, max_factor=0.1):
    """Random zoom augmentation"""
    factor = np.random.uniform(1.0 - max_factor, 1.0 + max_factor)
    
    # Calculate zoom parameters to keep image centered
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    # Zoom in/out
    zoomed = zoom(image, factor, order=1)
    
    # Crop or pad to maintain original size
    if factor > 1:  # Zoomed in - crop
        start_h = (zoomed.shape[0] - h) // 2
        start_w = (zoomed.shape[1] - w) // 2
        zoomed = zoomed[start_h:start_h+h, start_w:start_w+w]
    else:  # Zoomed out - pad
        pad_h = (h - zoomed.shape[0]) // 2
        pad_w = (w - zoomed.shape[1]) // 2
        zoomed_padded = np.zeros_like(image)
        zoomed_padded[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
        zoomed = zoomed_padded
    
    if mask is not None:
        if factor > 1:  # Zoomed in
            zoomed_mask = zoom(mask, factor, order=0)
            zoomed_mask = zoomed_mask[start_h:start_h+h, start_w:start_w+w]
        else:  # Zoomed out
            zoomed_mask = zoom(mask, factor, order=0)
            mask_padded = np.zeros_like(mask)
            mask_padded[pad_h:pad_h+zoomed_mask.shape[0], pad_w:pad_w+zoomed_mask.shape[1]] = zoomed_mask
            zoomed_mask = mask_padded
        
        return zoomed, zoomed_mask
    
    return zoomed

def random_skew(image, mask=None, intensity=0.1):
    """Random skew/shear augmentation"""
    # Create affine transform
    affine_tf = AffineTransform(
        shear=np.random.uniform(-intensity, intensity)
    )
    
    # Apply transform
    skewed = warp(image, affine_tf, mode='edge', preserve_range=True)
    
    if mask is not None:
        skewed_mask = warp(mask, affine_tf, mode='constant', order=0, preserve_range=True)
        return skewed.astype(image.dtype), skewed_mask.astype(mask.dtype)
    
    return skewed.astype(image.dtype)

def random_intensity(image, factor_range=(0.8, 1.2)):
    """Random intensity scaling"""
    factor = np.random.uniform(*factor_range)
    adjusted = image * factor
    return np.clip(adjusted, 0, 1)

def patch_tumor_transfer(healthy_image, tumor_patch, tumor_mask):
    """Transfer tumor patch to a healthy image"""
    # Get dimensions
    h, w = healthy_image.shape
    ph, pw = tumor_patch.shape
    
    # Randomly position the tumor (ensuring it fits)
    y_pos = np.random.randint(0, h - ph + 1)
    x_pos = np.random.randint(0, w - pw + 1)
    
    # Create a copy of the healthy image
    augmented = healthy_image.copy()
    
    # Apply the tumor mask and blend
    blend_mask = tumor_mask.astype(float)
    
    # Get the region where the tumor will be placed
    region = augmented[y_pos:y_pos+ph, x_pos:x_pos+pw]
    
    # Blend the tumor with the healthy region
    blended = region * (1 - blend_mask) + tumor_patch * blend_mask
    
    # Place back into the image
    augmented[y_pos:y_pos+ph, x_pos:x_pos+pw] = blended
    
    # Create and return the mask for the transferred tumor
    tumor_transfer_mask = np.zeros_like(healthy_image, dtype=bool)
    tumor_transfer_mask[y_pos:y_pos+ph, x_pos:x_pos+pw] = tumor_mask
    
    return augmented, tumor_transfer_mask

def augment_dataset(tumor_patches, validation_folder, num_samples=1000, patch_size=64):
    """Create augmented dataset with traditional augmentation methods"""
    X_augmented = []
    y_augmented = []
    masks_augmented = []
    
    # Get healthy brain slices
    healthy_slices = []
    
    validation_subjects = [d for d in os.listdir(validation_folder) 
                         if os.path.isdir(os.path.join(validation_folder, d))]
    
    print("Collecting healthy slices...")
    for subject in tqdm(validation_subjects):
        subject_folder = os.path.join(validation_folder, subject)
        flair_file = os.path.join(subject_folder, f"{subject}_flair.nii")
        t1_file = os.path.join(subject_folder, f"{subject}_t1.nii")
        
        if os.path.exists(flair_file) and os.path.exists(t1_file):
            flair_data, _ = load_nifti(flair_file)
            t1_data, _ = load_nifti(t1_file)
            
            if flair_data is None or t1_data is None:
                continue
            
            # Collect middle slices which typically have more brain tissue
            z_indices = range(flair_data.shape[2]//4, 3*flair_data.shape[2]//4)
            
            for z in z_indices:
                flair_slice = normalize_image(flair_data[:, :, z])
                t1_slice = normalize_image(t1_data[:, :, z])
                
                # Check if slice has sufficient tissue (not just background)
                if np.mean(flair_slice) > 0.01:
                    healthy_slices.append({
                        'flair': flair_slice,
                        't1': t1_slice
                    })
    
    print(f"Collected {len(healthy_slices)} healthy slices")
    
    # Generate augmented samples
    print("Generating augmented samples...")
    num_generated = 0
    
    while num_generated < num_samples and healthy_slices and tumor_patches:
        # Randomly select a healthy slice
        healthy_idx = np.random.randint(0, len(healthy_slices))
        healthy_flair = healthy_slices[healthy_idx]['flair']
        healthy_t1 = healthy_slices[healthy_idx]['t1']
        
        # Randomly select a tumor patch
        tumor_idx = np.random.randint(0, len(tumor_patches))
        tumor_patch = tumor_patches[tumor_idx]
        
        # Apply tumor transfer
        tumor_flair = normalize_image(tumor_patch['flair'])
        tumor_mask = tumor_patch['mask'].astype(bool)
        tumor_seg = tumor_patch['seg']
        
        # Apply patch tumor transfer
        augmented_flair, transferred_mask = patch_tumor_transfer(
            healthy_flair, tumor_flair, tumor_mask
        )
        
        # Apply random traditional augmentations with probability
        if np.random.random() < 0.5:
            # Horizontal flip
            if np.random.random() < 0.5:
                augmented_flair = horizontal_flip(augmented_flair)
                transferred_mask = horizontal_flip(transferred_mask)
                healthy_t1 = horizontal_flip(healthy_t1)
            
            # Vertical flip
            if np.random.random() < 0.5:
                augmented_flair = vertical_flip(augmented_flair)
                transferred_mask = vertical_flip(transferred_mask)
                healthy_t1 = vertical_flip(healthy_t1)
            
            # Random rotation
            if np.random.random() < 0.7:
                augmented_flair, transferred_mask = random_rotation(
                    augmented_flair, transferred_mask, max_angle=15
                )
                healthy_t1 = random_rotation(healthy_t1, max_angle=15)
            
            # Random shift
            if np.random.random() < 0.7:
                augmented_flair, transferred_mask = random_shift(
                    augmented_flair, transferred_mask, max_shift=10
                )
                healthy_t1 = random_shift(healthy_t1, max_shift=10)
            
            # Random zoom
            if np.random.random() < 0.5:
                augmented_flair, transferred_mask = random_zoom(
                    augmented_flair, transferred_mask, max_factor=0.1
                )
                healthy_t1 = random_zoom(healthy_t1, max_factor=0.1)
            
            # Random skew
            if np.random.random() < 0.3:
                augmented_flair, transferred_mask = random_skew(
                    augmented_flair, transferred_mask, intensity=0.05
                )
                healthy_t1 = random_skew(healthy_t1, intensity=0.05)
            
            # Random intensity for FLAIR tumor only
            if np.random.random() < 0.8:
                # Apply intensity change only to the tumor region
                intensity_factor = np.random.uniform(0.8, 1.5)
                tumor_region = augmented_flair * transferred_mask
                augmented_flair = augmented_flair * (1 - transferred_mask) + tumor_region * intensity_factor
                augmented_flair = np.clip(augmented_flair, 0, 1)
        
        # Stack channels and create labels
        # Input: stacked FLAIR, T1 and tumor mask
        # Output: augmented FLAIR with tumor (ground truth)
        input_stack = np.stack([healthy_flair, healthy_t1, transferred_mask], axis=-1)
        output_stack = np.expand_dims(augmented_flair, axis=-1)
        
        X_augmented.append(input_stack)
        y_augmented.append(output_stack)
        masks_augmented.append(transferred_mask)
        
        num_generated += 1
        if num_generated % 100 == 0:
            print(f"Generated {num_generated} augmented samples")
    
    return np.array(X_augmented), np.array(y_augmented), np.array(masks_augmented)

def visualize_augmentations(X, y, masks, num_samples=5):
    """Visualize some augmented samples"""
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, idx in enumerate(indices):
        # Original FLAIR (healthy)
        axes[i, 0].imshow(X[idx, :, :, 0], cmap='gray')
        axes[i, 0].set_title('Healthy FLAIR')
        axes[i, 0].axis('off')
        
        # Original T1 (healthy)
        axes[i, 1].imshow(X[idx, :, :, 1], cmap='gray')
        axes[i, 1].set_title('Healthy T1')
        axes[i, 1].axis('off')
        
        # Tumor mask
        axes[i, 2].imshow(X[idx, :, :, 2], cmap='gray')
        axes[i, 2].set_title('Tumor Mask')
        axes[i, 2].axis('off')
        
        # Augmented FLAIR with tumor
        axes[i, 3].imshow(y[idx, :, :, 0], cmap='gray')
        axes[i, 3].set_title('Augmented FLAIR')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'augmentation_samples.png'))
    plt.close()

def main():
    print("Extracting tumor regions from training data...")
    tumor_regions, _ = extract_tumor_regions(TRAINING_PATH, max_subjects=50)
    
    print("Extracting tumor patches...")
    tumor_patches = extract_tumor_patches(tumor_regions, patch_size=64)
    
    print("Creating augmented dataset with traditional methods...")
    X_augmented, y_augmented, masks_augmented = augment_dataset(
        tumor_patches, VALIDATION_PATH, num_samples=500
    )
    
    print("Visualizing augmentations...")
    visualize_augmentations(X_augmented, y_augmented, masks_augmented)
    
    # Save some samples for later use
    np.save(os.path.join(OUTPUT_PATH, 'X_augmented_samples.npy'), X_augmented[:50])
    np.save(os.path.join(OUTPUT_PATH, 'y_augmented_samples.npy'), y_augmented[:50])
    
    print(f"Final dataset shape: X={X_augmented.shape}, y={y_augmented.shape}")
    print(f"Results saved to {OUTPUT_PATH}")
    
    # Create train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_augmented, y_augmented, test_size=0.2, random_state=42
    )
    
    # Create PyTorch datasets
    train_dataset = BrainTumorDataset(X_train, y_train)
    val_dataset = BrainTumorDataset(X_val, y_val)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

if __name__ == "__main__":
    main()