import torch
from typing import Dict, Any, Tuple
from physics.physics import ElectrochemicalPhysics
from losses.losses import compute_boundary_residuals_for_adaptive
from losses.losses import compute_initial_residuals_for_adaptive
from losses.losses import compute_film_physics_loss
import torch.nn as nn

torch.manual(613)

