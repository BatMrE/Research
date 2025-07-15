import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

xyz = tiff.imread(
    r"D:\MS\Research\Thesis\Code\github_MVtec3D\Data\bagel\test\combined\xyz\021.tiff"
)  # (H, W, 3)
points = xyz.reshape(-1, 3)
mask = np.isfinite(points).all(axis=1)
points = points[mask]
if points.shape[0] > 200_000:
    idx = np.random.choice(points.shape[0], 200_000, replace=False)
    points = points[idx]

view_angles = {
    "front":  (30,   0),
    "side":   (30,  90),
    "top":    (90,   0),
    "iso1":   (45,  45),
    "iso2":   (20, 135),
}

n = len(view_angles)
cols = 3
rows = (n + cols - 1) // cols
fig = plt.figure(figsize=(4*cols, 4*rows))

for i, (name, (elev, azim)) in enumerate(view_angles.items(), start=1):
    ax = fig.add_subplot(rows, cols, i, projection="3d")
    ax.scatter(
        points[:,0], points[:,1], points[:,2],
        s=0.5,  # point size
        depthshade=False,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name)
    ax.set_axis_off()

plt.tight_layout()
plt.show()
