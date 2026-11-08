"""Supervised segmentation with a small U-Net, using the optional PyTorch extra.

Torch-free modules: ``config``, ``manifest``, ``splitting``, ``preparation``, ``augment``,
``metrics``, ``synthetic``, ``runtime``. They work without PyTorch installed.

Torch modules: ``unet``, ``losses``, ``torch_data``, ``checkpoint``, ``inference``, ``train``,
``evaluate``. They are imported only after :func:`microflow.ml.runtime.require_torch` succeeds.
See docs/machine-learning.md.
"""
