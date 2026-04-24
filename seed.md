# Seed notes

## GPU / PyTorch

This environment ships a CUDA build of PyTorch on `linux-64` (via the
`pytorch-gpu` conda-forge metapackage, pinned through `[system-requirements]
cuda = "12"` in `pixi.toml`). On `osx-arm64` and `linux-aarch64` the plain CPU
build is used.

Verified working on this host: Tesla P100-PCIE-12GB, driver 570.133.07,
CUDA runtime 12.9, cuDNN 9.10.2, compute capability 6.0.
