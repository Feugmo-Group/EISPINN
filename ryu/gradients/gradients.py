"""
Gradient Computation for PINNs
"""

import torch
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass
import warnings


class GradientContainer(NamedTuple):
    """
    Container for gradient computation results.

    Organizes all computed derivatives in a structured way for easy access.
    """

    #Network Predicitons
    phi: torch.Tensor # Potential
    c_K: torch.Tensor # Consentration of K+
    c_OH: torch.Tensor # Concentration of OH-
    c_O: torch.Tensor # Concentration of O2

    #Time derivatives
    c_K_t: torch.Tensor #dc_K/dt
    c_OH_t: torch.Tensor #dc_OH/dt
    c_O_t: torch.Tensor #dc_O/dt

    # First Spatial derivatives
    phi_x: torch.Tensor #dphi/dx
    c_K_x: torch.Tensor
    c_OH_x: torch.Tensor
    c_O_x: torch.Tensor

    #Second Spatial derivatives
    phi_xx: torch.Tensor
    c_K_xx: torch.Tensor
    c_OH_xx: torch.Tensor
    c_O_xx: torch.Tensor


@dataclass
class GradientConfig:
    """
    Configuration for gradient computations
    """
    create_graph: bool = True #for higeher order derivatives
    retain_graph: bool = True # Keep computation graph
    validate_inputs: bool = True # Check tensor properties



class GradientComputer:
    """
    Efficient gradient computation for PINNs

    Provides methods to compute gradients needed for PDE residuals
    """
    def __init__(self, config: Optional[GradientConfig] = None, device: Optional[torch.device] = None):
        """
        Initialize gradient computer.

        Args:
            config: Configuration for gradient computations
            device: PyTorch device for computations
        """
        self.config = config or GradientConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_derivative(
            self,
            output: torch.Tensor,
            input_var: torch.Tensor,
            order: int = 1,
            create_graph: Optional[bool] = None,
            retain_graph: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Compute derivative of output with respect to input_var.

        Args:
            output: Network output tensor
            input_var: Input variable to differentiate with respect to
            order: Order of derivative (1 or 2)
            create_graph: Override config setting for create_graph
            retain_graph: Override config setting for retain_graph

        Returns:
            Derivative tensor
        """

        create_graph = create_graph if create_graph is not None else self.config.create_graph
        retain_graph = retain_graph if retain_graph is not None else self.config.retain_graph

        if order == 1:
            return self._first_derivative(output, input_var, create_graph, retain_graph)
        elif order == 2:
            # Compute first derivative, then take derivative again
            first_deriv = self._first_derivative(output, input_var, create_graph=True, retain_graph=True)
            return self._first_derivative(first_deriv, input_var, create_graph, retain_graph)
        else:
            raise ValueError(f"Derivative order {order} not supported. Use 1 or 2.")

    def _first_derivative(
            self,
            output: torch.Tensor,
            input_var: torch.Tensor,
            create_graph: bool,
            retain_graph: bool
    ) -> torch.Tensor:
        """Compute first-order derivative using autograd"""
        grad_outputs = torch.ones_like(output)

        try:
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=input_var,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            return gradients

        except RuntimeError as e:
            if "does not require grad" in str(e):
                raise ValueError(
                    f"Input tensor must have requires_grad=True. Got requires_grad={input_var.requires_grad}")
            else:
                raise e
    def compute_electrochemistry_gradients(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            E: torch.Tensor,
            networks: Dict[str, torch.nn.Module]
    ) -> GradientContainer:
        """
        Compute all gradients needed for electrochemistry PINNs.

        Args:
            x: Spatial coordinates (requires_grad=True)
            t: Time coordinates (requires_grad=True)
            E: Applied potential
            networks: Dictionary of networks or NetworkManager instance

        Returns:
            GradientResults with all computed derivatives
        """
        # Forward pass through networks
        inputs_3d = torch.cat([x, t, E], dim=1)

        # Get network predictions
        phi = networks['potential'](inputs_3d)
        c_K_raw = networks['K'](inputs_3d)
        c_OH_raw = networks['OH'](inputs_3d)
        c_O_raw = networks['O'](inputs_3d)


        # Networks predict concentrations directly
        c_K = c_K_raw
        c_OH = c_OH_raw
        c_O = c_O_raw


        # Direct derivatives
        c_K_t = self.compute_derivative(c_K, t)
        c_OH_t = self.compute_derivative(c_OH, t)
        c_O_t = self.compute_derivative(c_O, t)

        # Compute first spatial derivatives
        phi_x = self.compute_derivative(phi, x)

        c_K_x = self.compute_derivative(c_K, x)
        c_OH_x = self.compute_derivative(c_OH, x)
        c_O_x = self.compute_derivative(c_O, x)

        # Compute second spatial derivatives
        phi_xx = self.compute_derivative(phi_x, x)

        c_K_xx = self.compute_derivative(c_K_x, x)
        c_OH_xx = self.compute_derivative(c_OH_x, x)
        c_O_xx = self.compute_derivative(c_O_x, x)

        return GradientContainer(
            phi=phi, c_K=c_K, c_OH=c_OH, c_O=c_O,
            c_K_t=c_K_t, c_OH_t=c_OH_t, c_O_t=c_O_t,
            phi_x=phi_x, c_K_x=c_K_x, c_OH_x=c_OH_x, c_O_x=c_O_x,
            phi_xx=phi_xx, c_K_xx=c_K_xx, c_OH_xx=c_OH_xx, c_O_xx=c_O_xx
        )