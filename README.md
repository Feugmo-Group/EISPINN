# EISPINN

This repository contains a Jupyter Notebook implementing a Physics-Informed Neural Network (PINN) to model the alkaline Oxygen Evolution Reaction (OER). The PINN leverages Neural Tangent Kernel (NTK) weighting to adaptively balance the loss terms, improving convergence and accuracy in solving the governing partial differential equations.

# Repo Structure
├── NTKRYU.ipynb   # Main Jupyter notebook implementation
├── conf/
│   ├── config.yaml
├── ryu/             # Work-in-progress modules for future package
│   ├── analysis/       # Post-processing, metrics, and evaluation
│   ├── gradients/      # Automatic differentiation utilities
│   ├── losses/         # PINN and PDE loss functions (outdated)
│   ├── networks/       # Neural network architectures
│   ├── physics/        # Governing equations & operators (outdated)
│   ├── sampling/       # Training data / collocation point sampling (outdated)
│   ├── training/       # Training loop and optimizers (outdated)
│   └── weighting/      # NTK-based and adaptive weighting strategies (outdated)
├── .gitignore
├── LICENSE
└── README.md            # Project description

Prerequisites:
  Python 3.9+
  Jupyter Notebook
  Dependencies:
    pip install torch numpy matplotlib hydra-core

