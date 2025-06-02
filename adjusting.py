import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    """Fully Connected Feed Forward Neural Net"""
    def __init__(self,cfg,input_dim=3,output_dim=2,hidden_layers=5,layer_size=20):
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = hidden_layers
        self.layer_size = layer_size

        self.activation = Swish()

        # Input layer
        self.input_layer = nn.Linear(input_dim, self.layer_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.layer_size, self.layer_size)
            for _ in range(self.num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.layer_size, output_dim)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for i, layer in enumerate(self.hidden_layers):
                x = self.activation(layer(x))
                
        return self.output_layer(x)
    

class PNP():
     """
     Reimplementation of the work from Mayen-Mondragon et al. (2019) as a PINN
     Inputs: x, w (omega - frequency)
     Outputs:
        C_ox
        C_red
        phi
     """
     def __init__(self,cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks
        self.C_red_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=2).to(self.device)
        self.C_ox_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=2).to(self.device)
        self.phi_net = FFN(cfg.arch.fully_connected, input_dim=3, output_dim=2).to(self.device)

        # Physics parameters
        self.d = cfg.pnp.geometry.d
        self.D_red = cfg.pnp.physics.D_red
        self.D_ox = cfg.pnp.physics.D_ox
        self.k0 = cfg.pnp.physics.k0
        self.alpha_c = cfg.pnp.physics.alpha_c
        self.alpha_a = cfg.pnp.physics.alpha_a
        self.C_bulk = cfg.pnp.physics.C_bulk
        self.C_dl = cfg.pnp.physics.C_dl
        self.n = cfg.pnp.physics.n
        self.F = cfg.pnp.physics.F
        self.R = cfg.pnp.physics.R
        self.T = cfg.pnp.physics.T
        self.eta_amp = cfg.pnp.physics.eta_amp

        # Optimizer
        params = (list(self.C_red_net.parameters()) +
                 list(self.C_ox_net.parameters()) +
                 list(self.phi_net.parameters()))
        self.optimizer = optim.Adam(
            params,
            lr=cfg.optimizer.adam.lr,
            betas=cfg.optimizer.adam.betas,
            eps=cfg.optimizer.adam.eps,
            weight_decay=cfg.optimizer.adam.weight_decay
        )

        # Scheduler/ Same as in PhysicsNEMO for now
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.scheduler.tf_exponential_lr.decay_steps,
            gamma=cfg.scheduler.tf_exponential_lr.decay_rate
        )

        # Loss weights
        self.poisson_weight = cfg.pnp.weights.poisson_weight
        self.nernst_weight = cfg.pnp.weights.nernst_weight
        self.bc_weight = cfg.pnp.weights.bc_weight
        self.ic_weight = cfg.pnp.weights.ic_weight

     def forward_pass(self, x, w, E):
        '''
        Forward pass with positivity constraints
        '''
        # ****** Establishes x, w, E as inputs ********
        inputs = torch.cat([x, w, E], dim=1)

        C_red_output = self.C_red_net(inputs)
        C_ox_output = self.C_ox_net(inputs)
        phi_output = self.phi_net(inputs)

        C_red_real = C_red_output[:,0:1]
        C_red_imag = C_red_output[:, 1:2]
        C_ox_real = C_ox_output[:,0:1]
        C_ox_imag = C_ox_output[:, 1:2]
        phi_real = phi_output[:,0:1]
        phi_imag = phi_output[:,1:2]        

        C_red = torch.complex(C_red_real, C_red_imag)
        C_ox = torch.complex(C_ox_real, C_ox_imag)
        phi = torch.complex(phi_real, phi_imag)
        
        return C_red, C_ox, phi
     
     def gradient_computation(self, x, w, E):
         """
         Computes gradients of concentrations and Poisson
         """
         x.requires_grad_(True)
         w.requires_grad_(True)
         E.requires_grad_(True)

         C_red, C_ox, phi = self.forward_pass(x, w, E)

         #Spatial Gradients
         C_red_x = torch.complex(torch.autograd.grad(C_red.real, x, grad_outputs=torch.ones_like(C_red.real),
                                         create_graph=True, retain_graph=True)[0], 
                                 torch.autograd.grad(C_red.imag, x, grad_outputs=torch.ones_like(C_red.imag),
                                         create_graph=True, retain_graph=True)[0])
         C_ox_x = torch.complex(torch.autograd.grad(C_ox.real, x, grad_outputs=torch.ones_like(C_ox.real),
                                         create_graph=True, retain_graph=True)[0], 
                                 torch.autograd.grad(C_ox.imag, x, grad_outputs=torch.ones_like(C_ox.imag),
                                         create_graph=True, retain_graph=True)[0])
         
         C_red_xx = torch.complex(torch.autograd.grad(C_red_x.real, x, grad_outputs=torch.ones_like(C_red_x.real),
                                         create_graph=True, retain_graph=True)[0], 
                                 torch.autograd.grad(C_red_x.imag, x, grad_outputs=torch.ones_like(C_red_x.imag),
                                         create_graph=True, retain_graph=True)[0])
         C_ox_xx = torch.complex(torch.autograd.grad(C_ox_x.real, x, grad_outputs=torch.ones_like(C_ox_x.real),
                                         create_graph=True, retain_graph=True)[0], 
                                 torch.autograd.grad(C_ox_x.imag, x, grad_outputs=torch.ones_like(C_ox_x.imag),
                                         create_graph=True, retain_graph=True)[0])
        
         return C_red, C_ox, phi, C_red_x, C_ox_x, C_red_xx, C_ox_xx 

     def BV_rates(self, C_red_surf, C_ox_surf, eta):
        '''
        Compute Butler-Volmer reaction rates based on the rates in the paper
        '''
        #exponential terms
        exp_cathodic = torch.exp(-self.alpha_c * self.n *self.F * eta / (self.R * self.T))
        exp_anodic = torch.exp(-self.alpha_a * self.n *self.F * eta / (self.R * self.T))
        
        #BV current density
        J_R = self.n * self.F * self.k0 * (C_red_surf*exp_cathodic - C_ox_surf*exp_anodic)

        return J_R
    
     def pde_residuals(self, x, w, E):
         """
         Nernst-Planck equations + Poisson
         Takes x and w as inputs to output tensors representing the concentration and potential
         """
         C_red, C_ox, phi, C_red_x, C_ox_x, C_red_xx, C_ox_xx = self.gradient_computation(x, w, E)

         #complex term
         i_w = 1j * w.squeeze(-1).unsqueeze(-1)

         #iwC_red = D_red * nabla^2 * C_red
         np_red = i_w * C_red - self.D_red * C_red_xx

         #iwC_ox = D_ox * nabla^2 * C_ox
         np_ox = i_w * C_ox - self.D_ox * C_ox_xx

         return np_red, np_ox

     def interior_loss(self, w, E):
         """
         Physics based loss from governing equations
         """
         x = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * self.d
         w_interior = w[:self.cfg.batch_size.interior] if w.size(0) >= self.cfg.batch_size.interior else w 
         E_interior = E[:self.cfg.batch_size.interior] if E.size(0) >= self.cfg.batch_size.interior else E

         #compute residuals
         np_red, np_ox = self.pde_residuals(x, w_interior, E_interior)

         # loss for complex equations
         red_loss = torch.mean(torch.abs(np_red)**2)
         ox_loss = torch.mean(torch.abs(np_ox)**2)

         total_int_loss = red_loss + ox_loss
         
         return total_int_loss, red_loss, ox_loss


     def boundary_loss(self, w, E):
         """
         Boundary loss from domains of physical system
         """
         bc_loss = 0.0
         x_left = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
         w_bc = w[:self.cfg.batch_size.BC] if w.size(0) >= self.cfg.batch_size.BC else w
         E_bc = E[:self.cfg.batch_size.BC] if E.size(0) >= self.cfg.batch_size.BC else E

         C_red, C_ox, phi, C_red_x, C_ox_x, C_red_xx, C_ox_xx = self.gradient_computation(x_left, w_bc, E_bc)

         #overpotential
         eta = torch.complex(
             E_bc,
             torch.zeros_like(E_bc)
         )
         
         #BV current at electrode surface
         J_R = self.BV_rates(C_red, C_ox, eta)

         # DL current
         i_w_scalar = 1j * w_bc.squeeze(-1).unsqueeze(-1)
         J_dl = self.C_dl * i_w_scalar * eta

         #Total current
         J_total = J_R + J_dl

         impedance_loss = torch.mean(torch.abs(J_total - (self.C_dl * 1j * w_bc * eta))**2)

         bc_loss+=torch.mean(torch.abs(impedance_loss)**2)

         # BCs: -D * nablaC = J_R/(n*F)
         flux_red = -self.D_red * C_red_x + J_R / (self.n * self.F)
         flux_ox = -self.D_ox * C_ox_x + J_R / (self.n * self.F)
         
         bc_loss += torch.mean(torch.abs(flux_red)**2 + torch.abs(flux_ox)**2)

         #RBC
         x_right = torch.full((self.cfg.batch_size.BC, 1), self.d, device=self.device)
         C_red_bulk, C_ox_bulk, _, _, _, _, _ = self.gradient_computation(x_right, w_bc, E_bc)

         #Bulk concentrations
         bulk_bc_red = C_red_bulk - torch.complex(
             torch.full_like(C_red_bulk.real, self.C_bulk),
             torch.zeros_like(C_red_bulk.imag)
         )

         bulk_bc_ox = C_ox_bulk - torch.complex(
             torch.full_like(C_ox_bulk.real, self.C_bulk),
             torch.zeros_like(C_ox_bulk.imag)
         )
         
         bc_loss+= torch.mean(torch.abs(bulk_bc_red)**2 + torch.abs(bulk_bc_ox)**2)

         return bc_loss
    
     def total_loss(self, log_freq, E):
        """Compute total loss."""
        interior_loss, red_loss, ox_loss = self.interior_loss(log_freq, E)
        bc_loss = self.boundary_loss(log_freq, E)

        total = interior_loss+bc_loss*self.bc_weight

        return total, interior_loss, red_loss, ox_loss, bc_loss

     def train_step(self, log_freq, E):
        """
        Perform one training step.
        """
        self.optimizer.zero_grad()
        loss, interior_loss, red_loss, ox_loss, bc_loss = self.total_loss(log_freq, E)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return (loss.item(), interior_loss.item(), red_loss.item(), ox_loss.item(), bc_loss.item()
        )
     def train(self):
        """Train the model."""
        losses = []
        interior_losses = []
        red_losses = []
        ox_losses = []
        bc_losses = []

        # Training loop
        for step in range(self.cfg.training.max_steps):
            # Sample frequency for this step
            log_freq = torch.rand(max(self.cfg.batch_size.interior, self.cfg.batch_size.BC,), 1, device = self.device) * 6 - 2
            E = torch.rand(max(self.cfg.batch_size.interior, self.cfg.batch_size.BC,), 1, device = self.device) * 5e-3 * torch.sin(10**(log_freq))
            w_log = (log_freq+2)/6
            w = 2 * np.pi * (10**log_freq)

            loss, interior_loss, red_loss, ox_loss, bc_loss = self.train_step(log_freq, E)
            losses.append(loss)
            interior_losses.append(interior_loss)
            red_losses.append(red_loss)
            ox_losses.append(ox_loss)
            bc_losses.append(bc_loss)
            

            # Print progress
            if step % self.cfg.training.rec_results_freq == 0:
                print(f"Step {step}, Loss: {loss:.6f}, Interior: {interior_loss:.6f}, "
                      f"Red: {red_loss:.6f}, Ox: {ox_loss:.6f}, BC: {bc_loss:.6f}")
                
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    self.save_model(f"outputs/checkpoints/model_step_{step}")
                    
        # Final save
        print(f"Step {step}, Loss: {loss:.6f}")
        self.save_model("outputs/checkpoints/model_final")

        return losses, interior_losses, red_losses, ox_losses, bc_losses

     def save_model(self, name):
        """
        Save model state
        """
        torch.save({
            'C_red_net': self.C_red_net.state_dict(),
            'C_ox_net': self.C_ox_net.state_dict(),
            'phi_net': self.phi_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{name}.pt")

     def compute_impedance(self, frequencies):
         """
         Computes the impedance
         """
         impedances = []        
         for freq in frequencies:
            omega = torch.tensor([[2 * torch.pi * freq]], device=self.device, dtype=torch.float32)
            E = torch.tensor([[5e-3 * np.sin(2 * np.pi * freq)]], device=self.device, dtype=torch.float32)
            # Electrode surface (x=0)
            x_electrode = torch.zeros(1, 1, device=self.device, dtype=torch.float32)
            x_electrode.requires_grad_(True)

            # Enable gradients here
            C_red, C_ox, phi, C_red_x, C_ox_x, _, _ = self.gradient_computation(x_electrode, omega, E)

            # Complex overpotential
            eta = torch.complex(
                torch.tensor([[self.eta_amp]], device=self.device, dtype=torch.float32),
                torch.zeros(1, 1, device=self.device, dtype=torch.float32)
            )

            # Total current
            J_R = self.BV_rates(C_red, C_ox, eta)
            J_dl = self.C_dl * 1j * omega * eta
            J_total = J_R + J_dl

            with torch.no_grad():
                Z = eta / J_total
                impedances.append(Z.item())

         return np.array(impedances)


@hydra.main(config_path="conf", config_name="electrochemical_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = PNP(cfg)
    
    # Train
    losses, interior_losses, red_losses, ox_losses, bc_losses = model.train()
    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.semilogy(losses, label='Total Loss', linewidth=2)
    plt.semilogy(interior_losses, label='Interior Loss', linewidth=2)
    plt.semilogy(red_losses, label='Reduction Loss', linewidth=2)
    plt.semilogy(ox_losses, label='Oxidation Loss', linewidth=2)
    plt.semilogy(bc_losses, label='Boundary Loss', linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Loss Evolution')
    plt.savefig("outputs/plots/training_losses.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute and plot impedance spectrum
    frequencies = np.logspace(-2, 4, 100)  # 1e-2 to 1e4 Hz as in paper
    E = np.linspace(0, 5e-3, 100)
    impedances = model.compute_impedance(frequencies)
    
    # Nyquist plot
    plt.figure(figsize=(10, 8))
    plt.plot(impedances.real, -impedances.imag, 'b-', linewidth=2)
    plt.xlabel('Zreal (Ω·m²)')
    plt.ylabel('-Zimag (Ω·m²)')
    plt.title('Nyquist Plot')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("outputs/plots/nyquist_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bode plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Magnitude
    ax1.loglog(frequencies, np.abs(impedances))
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ω·m²)')
    ax1.set_title('Impedance Magnitude')
    ax1.grid(True)
    
    # Phase
    phase = np.angle(impedances) * 180 / np.pi
    ax2.semilogx(frequencies, phase)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Impedance Phase')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/plots/bode_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()