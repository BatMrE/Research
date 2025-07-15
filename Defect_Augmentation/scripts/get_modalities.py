import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os

index = "021"  # change as needed

base_path = r"D:\MS\Research\Thesis\Code\github_MVtec3D\Data\bagel\test\combined"
rgb_path = os.path.join(base_path, "rgb", f"{index}.png")
xyz_path = os.path.join(base_path, "xyz", f"{index}.tiff")
mask_path = os.path.join(base_path, "gt", f"{index}.png")

print("RGB Path:", rgb_path)
print("Exists?", os.path.exists(rgb_path))


rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

xyz = tiff.imread(xyz_path)  # Shape: (H, W, 3)
Z = xyz[..., 2]  # Depth map

depth_norm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX)
depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

overlay = rgb.copy()
overlay[mask > 0] = [255, 0, 0]  # Red overlay
blended = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

# Plot everything
plt.figure(figsize=(14, 6))

plt.subplot(1, 4, 1)
plt.imshow(rgb)
plt.title("RGB")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(depth_color)
plt.title("Depth (Z colormap)")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title("Anomaly Mask")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(blended)
plt.title("RGB + Mask Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()
