import subprocess

subprocess.run(
    'tar --use-compress-program="xz -T0" -xvf "D:/MS/Research/Thesis/Code/github_MVtec3D/Data/full_data/mv_3d_anomaly.tar.xz" -C "D:/MS/Research/Thesis/Code/github_MVtec3D/Data/full_data"',
    shell=True,
    check=True
)
