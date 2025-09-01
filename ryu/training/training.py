import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import os
import time

from networks.networks import NetworkManager
from physics.physics import ElectrochemicalPhysics
from losses.losses import compute_total_loss
from losses.losses import compute_total_loss_al
from losses.losses import compute_interior_loss,compute_boundary_loss,compute_initial_loss,compute_film_physics_loss
from losses.losses import _extract_constraint_violations_al

torch.manual_seed(613)

class PINNTrainer:
    """
    PINN trainer for electrochemical systems
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, output_dir: Optional[str] = None):
        """
        Initialize the PINN trainer with configurable loss weighting.

        Args:
            config: Complete configuration dictionary
            device: PyTorch device for computation
            output_dir: Directory for saving outputs and checkpoints
        """
        self.config = config
        self.device = device
        self.output_dir = output_dir or "outputs"
        self.al_config = ""
        self.use_al = False
        self.constraint_history = {}
        #initialize core components
        print("Initializing Ryu")
        self.networks = NetworkManager(config, device)
        self.physics = ElectrochemicalPhysics(config, device)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_step = 0
        self.best_loss = float('inf')
        self.best_checkpoint_path = None

        # Setup directories
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Loss history tracking
        self.loss_history = {
            'total': [], 'interior': [], 'boundary': [], 'initial': [],
        }

        # Training configuration
        self.max_steps = config['training']['max_steps']
        self.print_freq = config['training']['rec_results_freq']
        self.save_freq = config['training']['save_network_freq']

        # Training statistics
        self.start_time = None
        self.total_params = sum(p.numel() for p in self.networks.get_all_parameters() if p.requires_grad)

        print(f"✅ Initialization complete!")
        print(f"📊 Total parameters: {self.total_params:,}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create and configure the optimizer."""
        params = self.networks.get_all_parameters()
        optimizer_config = self.config['optimizer']['adam']

        optimizer = optim.AdamW(
            params,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )

        # Set initial_lr for scheduler compatibility
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create and configure the learning rate scheduler."""
        scheduler_config = self.config['scheduler']

        if scheduler_config['type'] == "RLROP":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config['RLROP']['factor'],
                patience=scheduler_config['RLROP']['patience'],
                threshold=scheduler_config['RLROP']['threshold'],
                min_lr=scheduler_config['RLROP']['min_lr'],
            )
        elif scheduler_config['type'] == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['tf_exponential_lr']['decay_rate'],
                last_epoch=scheduler_config['tf_exponential_lr']['decay_steps']
            )
        elif scheduler_config['type'] == "None":
            return optim.lr_scheduler.ConstantLR(
                self.optimizer,1.0,self.config.training.max_steps,self.config.training.max_steps
            )
        
    def compute_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses for current step with dynamic weighting.

        Returns:
            Dictionary of all loss components
        """
        # Update weights if using dynamic strategy
        self._update_loss_weights()

        # Sample training points
        (x_interior, t_interior, E_interior,
        x_boundary, t_boundary, E_boundary,
        x_initial, t_initial, E_initial,) = self.sample_training_points()

        if self.use_al:
            loss_dict, _, constraint_violations = compute_total_loss_al(
            x_interior, t_interior, E_interior,
            x_boundary, t_boundary, E_boundary,
            x_initial, t_initial, E_initial,
            self.networks, self.physics,
            self.al_manager
        )
            # Store AL metrics for plotting
            if 'penalty' in loss_dict:
                self.al_metrics_history['penalty_term'].append(loss_dict['penalty'].item())
            if 'lagrangian' in loss_dict:
                self.al_metrics_history['lagrangian_term'].append(loss_dict['lagrangian'].item())

            for name, violation in constraint_violations.items():
                if name not in self.constraint_history:
                    self.constraint_history[name] = []
                self.constraint_history[name].append(violation.item())

        # Compute all losses with current weights
        elif self.ntk_weights is not None:
            # Use NTK weights (granular component weighting)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                self.networks, self.physics,
                weights=None,  # Don't use standard weights
                ntk_weights=self.ntk_weights  # Use NTK component weights
            )
        else:
            # Use standard weights (uniform, batch_size, manual)
            loss_dict = compute_total_loss(
                x_interior, t_interior, E_interior,
                x_boundary, t_boundary, E_boundary,
                x_initial, t_initial, E_initial,
                self.networks, self.physics,
                weights=self.loss_weights,
                ntk_weights=None
            )

        return loss_dict
    
    def collect_current_points_and_residuals(self, networks):
        """
        Collect current collocation points and compute their residuals
        """

        #sample current points
        if self.config.sampling.strat == "Uniform":
            x_interior, t_interior, E_interior = self.sampler.sample_interior_points(networks)
            x_boundary, t_boundary, E_boundary = self.sampler.sample_boundary_points(networks)
            x_initial, t_initial, E_initial = self.sampler.sample_initial_points(networks)
        else: # Adaptive
            x_interior, t_interior, E_interior = self.sampler.get_interior_points()
            x_boundary, t_boundary, E_boundary = self.sampler.get_boundary_points()
            x_initial, t_initial, E_initial = self.sampler.get_initial_points()

        #compute residuals directly using existing loss functions
        _,_, interior_residuals = compute_interior_loss(
        x_interior, t_interior, E_interior, networks, self.physics, return_residuals=True)

        _,_, boundary_residuals = compute_boundary_loss(
        x_interior, t_interior, E_boundary, networks, self.physics, return_residuals=True)

        _,_, initial_residuals = compute_initial_loss(
        x_interior, t_interior, E_initial, networks, self.physics, return_residuals=True)

        # Package for plotting
        points_dict = {
            'interior': torch.cat([x_interior, t_interior, E_interior], dim=1),
            'boundary': torch.cat([x_boundary, t_boundary, E_boundary], dim=1),
            'initial': torch.cat([x_initial, t_initial, E_initial], dim=1)
        }

        residuals_dict = {
        **interior_residuals,  # cv_pde, av_pde, h_pde, poisson_pde
        'boundary': boundary_residuals,
        'initial': initial_residuals,
        }
        
        return points_dict, residuals_dict

    def training_step(self) -> Dict[str, float]:
        """
        Perform one complete training step with automatic weight balancing.

        Returns:
            Dictionary of loss values (as floats)
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute losses (includes weight updates)
        loss_dict = self.compute_losses()

        # Backward pass
        total_loss = loss_dict['total']
        total_loss.backward()

        #gradient clipping
        torch.nn.utils.clip_grad_norm_(self.networks.get_all_parameters(), max_norm=1.0) 

        if self.use_al:
            # Flip gradients for multipliers (ascent)
            torch.nn.utils.clip_grad_norm_(self.al_manager.get_multiplier_parameters(), max_norm=1.0)
            for param in self.al_manager.get_multiplier_parameters():
                if param.grad is not None:
                    param.grad *= -1

        # Optimizer step
        self.optimizer.step()
        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()

        # Convert tensors to floats for logging
        loss_dict_float = {k: v.item() if torch.is_tensor(v) else v
                        for k, v in loss_dict.items()}

        # Update training state
        self.current_step += 1

        return loss_dict_float

    def update_loss_history(self, loss_dict: Dict[str, float]) -> None:
        """
        Update the loss history tracking
        """
        for key in self.loss_history.keys():
            if key in loss_dict:
                self.loss_history[key].append(loss_dict[key])
            else:
                self.loss_history[key].append(0.0) #default for missing keys

    def print_progress(self, loss_dict: Dict[str, float]) -> None:
        """
        Print detailed training information
        """
        if self.current_step % self.print_freq == 0 or self.current_step==0:
            print(f"\n=== Step {self.current_step} ===")
            print(f"Total Loss: {loss_dict['total']:.6f}")
            
    def train(self, manual_loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        Run the complete training process with automatic balancing.
        """

        