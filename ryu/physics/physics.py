import torch
from typing import Dict, Any, NamedTuple, Tuple
from dataclasses import dataclass
from gradients.gradients import GradientComputer, GradientConfig


@dataclass
class PhysicsConstants:
    """Fundamental physical constants"""
    F: float = 96485.0  # Faraday constant [C/mol]
    R: float = 8.3145  # Gas constant [J/(mol·K)]
    T: float = 293.0  # Temperature [K]
    k_B: float = 1.3806e-23  # Boltzmann constant [J/K]
    eps0: float = 8.85e-12  # Vacuum permittivity [F/m]
    electron_charge: float = 1.6e-19  # Elementary charge [C]

@dataclass
class TransportProperties:
    """Transport properties for different species"""
    # Diffusion coefficients [m²/s]
    D_OH: float = 5.3e-9    # OH⁻ diffusion coefficient
    D_K: float = 1.96e-9    # K⁺ diffusion coefficient  
    D_O2: float = 2.0e-9    # O₂ diffusion coefficient

    # Species charges
    z_k: float = 1.0  
    z_oh: float = -1.0
    z_o: float = 0.0

@dataclass
class MaterialProperties:
    """Solution and electrode properties"""
    # Relative permittivity of electrolyte
    eps_r: float = 78.5     # Water at 25°C
    
    # Bulk concentrations [mol/m³]
    c_OH_bulk: float = 1000.0   # 1 M KOH
    c_K_bulk: float = 1000.0    # 1 M KOH
    c_O2_bulk: float = 0.26     # Saturated O₂ in water

@dataclass
class ReactionKinetics:
    """Reaction rate constants and kinetic parameters"""
    # Standard rate constants
    k0: float = 1e-6        # Exchange current density [A/m²]
    alpha: float = 0.5      # Transfer coefficient [dimensionless]
    gamma: float = 1.0      # Reaction order in OH⁻ [dimensionless]
    phi_eq: float = 1.23    # Equilibrium potential [V vs RHE]
    # Number of electrons
    n: float = 4.0          # Electrons per O₂ molecule

@dataclass
class GeometryParameters:
    """Geometric parameters"""
    L_domain: float = 1e-5  # Domain length [m]
    
@dataclass
class CharacteristicScales:
    """Characteristic scales for non-dimensionalization"""
    lc: float = 1e-5        # Length scale [m]
    cc: float = 1000.0      # Concentration scale [mol/m³]  
    tc: float = None        # Time scale [s] - computed
    phic: float = None      # Potential scale [V] - computed


class ElectrochemicalPhysics:
    """
    Main physics manager that handles all electrochemical calculations.

    This class consolidates all physics parameters and provides methods
    for PDE calculations, rate constant computations, and scaling.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        Initialize physics parameters from configuration.

        Args:
            config: Configuration dictionary
            device: PyTorch device for computations
        """
        self.device = device
        self.config = config

        # Load all parameter groups from config
        self._load_parameters_from_config()

        # Set up characteristic scales
        self._setup_characteristic_scales()

        # Initialize gradient computer
        grad_config = GradientConfig(
            create_graph=True,
            retain_graph=True,
            validate_inputs=True
        )

        self.grad_computer = GradientComputer(grad_config, device)

    def _load_parameters_from_config(self):
        """Load all physics parameters from configuration"""
        pde_config = self.config['pde']

        self.constants = PhysicsConstants()
        self.transport = TransportProperties()
        self.materials = MaterialProperties()
        self.kinetics = ReactionKinetics()
        self.geometry = GeometryParameters()
        self.scales = CharacteristicScales()
    
    def _setup_characteristic_scales(self):
        """Setup characteristic scales for non-dimensionalization"""
        # Compute derived scales
        self.scales.tc = self.scales.lc**2 / self.transport.D_OH
        self.scales.phic = self.constants.R * self.constants.T / self.constants.F

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Get a summary of all physics parameters"""
        return {
            'constants': self.constants.__dict__,
            'transport': self.transport.__dict__,
            'materials': self.materials.__dict__,
            'kinetics': self.kinetics.__dict__,
            'geometry': self.geometry.__dict__,
            'scales': self.scales.__dict__
        }

    def to_tensor(self, value: float) -> torch.Tensor:
        """Convert scalar to tensor on correct device"""
        return torch.tensor(value, device=self.device, dtype=torch.float32)
    
    def compute_rate_constants(self, t: torch.Tensor, E: torch.Tensor, networks, single: bool = False):
        """
        Compute electrochemical rate constants using BV kinetics

        R_OH = -k₀ * c_OH^γ * exp(α*F*(φ-φ_eq)/(R*T))
        R_O2 = (1/4) * k₀ * c_OH^γ * exp(α*F*(φ-φ_eq)/(R*T))

        Args:
            t: Time tensor (dimensionless)
            E: Applied potential tensor
            networks: NetworkManager instance
            single: Whether computing for single point or batch

        Returns:
            Tuple of (R_OH, R_O2) reaction rates
        """
        if single:
            batch_size = 1
            x_mf = torch.zeros(1, 1, device=self.device)
        else:
            batch_size = t.shape[0]
            x_mf = torch.zeros(batch_size, 1, device=self.device)

        # Get potentials at interfaces
        inputs_mf = torch.cat([x_mf, t, E], dim=1)
        phi = networks['potential'](inputs_mf)  # φ̂_mf
        c_OH = networks['c_oh'](inputs_mf)

        # Overpotential
        eta = phi - self.kinetics.phi_eq
        
        # Butler-Volmer exponential term
        exp_term = torch.exp(self.kinetics.alpha * self.constants.F * eta / 
                           (self.constants.R * self.constants.T))
        
        # Concentration dependence
        conc_term = torch.pow(c_OH, self.kinetics.gamma)
        
        # OH⁻ consumption rate (negative because consumed)
        R_OH = -self.kinetics.k0 * conc_term * exp_term
        
        # O₂ production rate (positive because produced)  
        R_O2 = (1.0/4.0) * self.kinetics.k0 * conc_term * exp_term
        
        return R_OH, R_O2
    
    def compute_current_density(self, c_OH: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute current density from reaction rate.
        
        j = n * F * |R_OH|
        
        Args:
            c_OH: OH⁻ concentration tensor
            phi: Electric potential tensor
            
        Returns:
            Current density tensor [A/m²]
        """
        R_OH, _ = self.compute_reaction_rates(c_OH, phi)
        j = self.kinetics.n * self.constants.F * torch.abs(R_OH)
        return j

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor, E: torch.Tensor, networks):
        """
        Compute gradients of all network outputs using the gradient computer.

        Args:
            x: Spatial coordinates (requires_grad=True)
            t: Time coordinates (requires_grad=True)
            E: Applied potential
            networks: NetworkManager instance

        Returns:
            GradientResults namedtuple with all gradients
        """
        return self.grad_computer.compute_electrochemistry_gradients(x, t, E, networks)
    
    def compute_pde_residuals(self, x: torch.Tensor, t: torch.Tensor, E: torch.Tensor, networks):
        """
        Compute PDE residuals for all governing equations

        Args:
            x: Spatial coordinates (dimensionless)
            t: Time coordinates (dimensionless)
            E: Applied potential
            networks: NetworkManager instance

        Returns:
            Tuple of residuals: (cv_residual, av_residual, h_residual, poisson_residual)
        """
        #Get all gradients using the gradient computer
        grads = self.compute_gradients(x, t, E, networks)

        # Compute reaction rates
        R_OH, R_O2 = self.compute_reaction_rates(grads['c_OH'], grads['phi'])

        # Constants for migration terms
        F_RT = self.constants.F / (self.constants.R * self.constants.T)

        # K resiudal
        k_migration = (self.transport.D_K * self.transport.z_k * F_RT *
                        (grads.c_K_x * grads.phi_x +grads.c_K * grads.phi_xx)
                        )

        k_residual = (grads.c_K_t - self.transport.D_K * grads.c_K_xx - k_migration)

        # OH residual
        oh_migration = self.transport.D_OH * self.transport.z_oh * F_RT * (
            grads.c_OH_x * grads.phi_x + grads.c_OH * grads.phi_xx
        )

        oh_residual = (grads.c_OH_t - self.transport.D_OH * grads.c_oh_xx -
                        oh_migration - R_OH)

        # O residual
        o_residual = (grads.c_O_t - 
                      self.transport.D_O2 * grads.c_O_xx + 
                      R_O2)

        # Poisson equation residual
        permittivity = self.materials.eps_r * self.constants.eps0
        charge_density = self.constants.F * (grads.c_K - grads.c_OH)
        
        poisson_residual = (grads.phi_xx + charge_density / permittivity)

        return k_residual, oh_residual, o_residual, poisson_residual