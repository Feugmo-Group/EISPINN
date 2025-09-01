"""
Loss function calculations for RYU

Simple functional approach for PINNs
"""

import torch
from typing import Dict, Tuple, Any, Union, Optional
torch.manual_seed(613)

def compute_interior_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                          networks, physics,
                          return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                                                                   Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Computes interior PDE residual losses.

    see compute_pde_residuals for full calculations

    Args:
        x: spatial coordinates
        t: temporal coordinates
        E: applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: if True, also return raw residuals for NTK

    returns:
        If return_residuals = False: Tuple of (total_interior_loss, individual_losses_dict)
        If return_residuals=True: Tuple of (total_interior_loss, individual_losses_dict, residuals_dict)
    """

    #compute PDE residuals using physics module
    k_residual, oh_residual, o_residual, poisson_residual = physics.compute_pde_residuals(x, t, E, networks)

    # Calculate individual losses
    k_pde_loss = torch.mean(k_residual ** 2)
    oh_pde_loss = torch.mean(oh_residual ** 2)
    if physics.config.pde.physics.include_holes:
        o_pde_loss = torch.mean(o_residual ** 2)
    else:
        o_pde_loss = torch.mean(torch.zeros_like(o_residual))
    poisson_pde_loss = torch.mean(poisson_residual ** 2)

    # Total interior loss
    total_interior_loss = k_pde_loss + oh_pde_loss + o_pde_loss + poisson_pde_loss

    individual_losses = {
        'k_pde': k_pde_loss,
        'oh_pde': oh_pde_loss,
        'o_pde': o_pde_loss,
        'poisson_pde': poisson_pde_loss
    }

    if return_residuals:
        residuals = {
            'k_pde': k_residual,
            'oh_pde': oh_residual,
            'o_pde': o_residual,
            'poisson_pde': poisson_residual
        }
        return total_interior_loss, individual_losses, residuals
    else:
        return total_interior_loss, individual_losses
    
def compute_boundary_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                          networks, physics,
                          return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Compute boundary losses

    Args:
        x
        t
        E
        networks
        physics
        return_residuals
    """
    batch_size = x.shape[0]
    half_batch = batch_size // 2

    # Total boundary loss
    total_boundary_loss = 0

    individual_losses = {
        'cv_mf_bc': 0,
        'av_mf_bc': 0,
        'u_mf_bc': 0,
        'cv_fs_bc': 0,
        'av_fs_bc': 0,
        'u_fs_bc': 0,
        'h_fs_bc': 0,
    }
    if return_residuals:
        # Combine all residuals into single tensor for NTK computation

        residuals_dict = {'cv_mf_bc':0, 'av_mf_bc':0, 'u_mf_bc':0, 
                    'cv_fs_bc':0, 'av_fs_bc':0, 'u_fs_bc':0, 'h_fs_bc':0}
        combined_residuals = torch.cat([
            0,0,
            0, 0,
            0, 0, 0
        ])
        return total_boundary_loss, individual_losses, combined_residuals,residuals_dict
    else:
        return total_boundary_loss, individual_losses

def compute_initial_loss(x: torch.Tensor, t: torch.Tensor, E: torch.Tensor,
                         networks, physics,
                         return_residuals: bool = False) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]],
Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Computes initial condition losses

    Args:
        x: Spatial coordinates
        t: Time coordinates (should be zeros)
        E: Applied potential
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Tuple of (total_initial_loss, individual_losses_dict)
        If return_residuals=True: Tuple of (total_initial_loss, individual_losses_dict, combined_residuals)
    """
    L_input = torch.cat([t, E], dim=1)
    inputs = torch.cat([x, t, E], dim=1)

    # K initial conditions
    k_initial_pred = networks['k'](inputs)
    k_initial_t = physics.grad_computer.compute_derivative(k_initial_pred, t)
    k_initial_residual = k_initial_pred + k_initial_t
    k_initial_loss = torch.mean(k_initial_pred**2) + torch.mean(k_initial_t**2)

    # Anion vacancy initial conditions
    oh_initial_pred = networks['oh'](inputs)
    oh_initial_t = physics.grad_computer.compute_derivative(oh_initial_pred, t)
    oh_initial_residual = oh_initial_pred + oh_initial_t
    oh_initial_loss = torch.mean(oh_initial_pred**2) + torch.mean(oh_initial_t**2)

    # Anion vacancy initial conditions
    o_initial_pred = networks['o'](inputs)
    o_initial_t = physics.grad_computer.compute_derivative(o_initial_pred, t)
    o_initial_residual = o_initial_pred + o_initial_t
    o_initial_loss = torch.mean(o_initial_pred**2) + torch.mean(o_initial_t**2)

    # Potential initial conditions
    u_initial_pred = networks['potential'](inputs)
    u_initial_t = physics.grad_computer.compute_derivative(u_initial_pred, t)
    poisson_initial_residual = (u_initial_pred - (
            (E / physics.scales.phic) - (1e7 * (physics.scales.lc / physics.scales.phic) * x))) + u_initial_t
    poisson_initial_loss = torch.mean((u_initial_pred - (
            (E / physics.scales.phic) - (1e7 * (physics.scales.lc / physics.scales.phic) * x)))**2) + torch.mean(u_initial_t**2)

    # Total initial loss
    total_initial_loss = k_initial_loss + oh_initial_loss + poisson_initial_loss + o_initial_loss

    individual_losses = {
        'k_ic': k_initial_loss,
        'oh_ic': oh_initial_loss,
        'poisson_ic': poisson_initial_loss,
        'o_ic': o_initial_loss,
    }

    if return_residuals:
        # Combine all residuals into single tensor for NTK computation

        residual_dict = {'k_ic':k_initial_residual, 'oh_ic':oh_initial_residual, 'o_ic':o_initial_residual, 'poisson_ic':poisson_initial_residual}
        combined_residuals = torch.cat([
            k_initial_residual,
            oh_initial_residual,
            poisson_initial_residual,
            o_initial_residual
        ])
        return total_initial_loss, individual_losses, combined_residuals,residual_dict
    else:
        return total_initial_loss, individual_losses
    
def compute_total_loss(x_interior: torch.Tensor, t_interior: torch.Tensor, E_interior: torch.Tensor,
                       x_boundary: torch.Tensor, t_boundary: torch.Tensor, E_boundary: torch.Tensor,
                       x_initial: torch.Tensor, t_initial: torch.Tensor, E_initial: torch.Tensor,
                       t_film: torch.Tensor, E_film: torch.Tensor,
                       networks, physics,
                       weights: Optional[Dict[str, float]] = None,
                       ntk_weights: Optional[Dict[str, float]] = None,
                       return_residuals: bool = False) -> Union[Dict[str, torch.Tensor],
Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    Compute all losses and return detailed breakdown.

    Args:
        x_interior, t_interior, E_interior: Interior points
        x_boundary, t_boundary, E_boundary: Boundary points
        x_initial, t_initial, E_initial: Initial points
        t_film, E_film: Film physics points
        networks: NetworkManager instance
        physics: ElectrochemicalPhysics instance
        weights: Standard loss weights dictionary (uniform/batch_size weighting)
        ntk_weights: NTK component weights dictionary (takes precedence over weights)
        return_residuals: If True, also return raw residuals for NTK computation

    Returns:
        If return_residuals=False: Dictionary with all loss components for monitoring and optimization
        If return_residuals=True: Tuple of (loss_dict, residuals_dict)
    """
    # Compute individual loss components
    if return_residuals:
        interior_loss, interior_breakdown, interior_residuals = compute_interior_loss(x_interior, t_interior,
                                                                                      E_interior, networks, physics,
                                                                                      return_residuals=True)
        boundary_loss, boundary_breakdown, boundary_residuals,_ = compute_boundary_loss(x_boundary, t_boundary,
                                                                                      E_boundary, networks, physics,
                                                                                      return_residuals=True)
        initial_loss, initial_breakdown, initial_residuals,_ = compute_initial_loss(x_initial, t_initial, E_initial,
                                                                                  networks, physics,
                                                                                  return_residuals=True)

        all_residuals = {
            **interior_residuals,  # K_pde, OH_pde, O_pde, poisson_pde
            'boundary': boundary_residuals,
            'initial': initial_residuals,
        }
    else:
        interior_loss, interior_breakdown = compute_interior_loss(x_interior, t_interior, E_interior, networks, physics)
        boundary_loss, boundary_breakdown = compute_boundary_loss(x_boundary, t_boundary, E_boundary, networks, physics)
        initial_loss, initial_breakdown = compute_initial_loss(x_initial, t_initial, E_initial, networks, physics)

    # Apply weights - NTK weights take precedence
    if ntk_weights is not None:
        # Use NTK's granular component weights
        weighted_cv_pde = ntk_weights.get('k_pde') * interior_breakdown['k_pde']
        weighted_av_pde = ntk_weights.get('oh_pde') * interior_breakdown['oh_pde']
        weighted_h_pde = ntk_weights.get('o_pde') * interior_breakdown['o_pde']
        weighted_poisson_pde = ntk_weights.get('poisson_pde') * interior_breakdown['poisson_pde']

        weighted_interior = weighted_cv_pde + weighted_av_pde + weighted_h_pde + weighted_poisson_pde
        weighted_boundary = ntk_weights.get('boundary') * boundary_loss
        weighted_initial = ntk_weights.get('initial') * initial_loss

        # Individual boundary and initial components with NTK weighting
        boundary_weight = ntk_weights.get('boundary')
        initial_weight = ntk_weights.get('initial')

    else:
        # Use standard weights (uniform, batch_size, or manual)
        if weights is None:
            weights = {
                'interior': 1.0,
                'boundary': 1.0,
                'initial': 1.0,
            }

        weighted_interior = weights['interior'] * interior_loss
        weighted_boundary = weights['boundary'] * boundary_loss
        weighted_initial = weights['initial'] * initial_loss

        # Individual components with standard weighting
        weighted_k_pde = weights['interior'] * interior_breakdown['k_pde']
        weighted_oh_pde = weights['interior'] * interior_breakdown['oh_pde']
        weighted_o_pde = weights['interior'] * interior_breakdown['o_pde']
        weighted_poisson_pde = weights['interior'] * interior_breakdown['poisson_pde']

        weighted_interior = weights['interior']*interior_loss 
        boundary_weight = weights['boundary']
        initial_weight = weights['initial']

    # Total loss
    total_loss = weighted_interior + weighted_boundary + weighted_initial

    # Combine all losses into one dictionary
    all_losses = {
        'total': total_loss,
        'interior': weighted_interior,
        'boundary': weighted_boundary,
        'initial': weighted_initial,

        # Individual PDE components
        'weighted_k_pde': weighted_k_pde,
        'weighted_oh_pde': weighted_oh_pde,
        'weighted_o_pde': weighted_o_pde,
        'weighted_poisson_pde': weighted_poisson_pde,

        # Individual boundary components
        **{f"weighted_{k}": boundary_weight * v for k, v in boundary_breakdown.items()},

        # Individual initial components
        **{f"weighted_{k}": initial_weight * v for k, v in initial_breakdown.items()}
    }

    if return_residuals:
        return all_losses, all_residuals
    else:
        return all_losses