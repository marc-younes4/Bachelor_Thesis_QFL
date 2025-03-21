"""
This script performs a simulation using a vortex injection only.
It first runs a background simulation and then a simulation with a vortex
(via multiplying the background by an ideal vortex factor).
At each saved propagation step (delta_z), it computes the “unwanted perturbation”
as the difference between the vortex simulation and the expected ideal field,
which is defined as the background simulation multiplied by the initial vortex factor.
We mask out the central 4 pixels (the vortex core) so that the vortex itself is not
considered a perturbation.
Finally, the unwanted perturbations are saved (here in a dictionary) to a file.
"""

# %% Import libraries and functions
import os
import pickle
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from NLSE import NLSE
from scipy.ndimage import map_coordinates
from skimage import restoration
import scipy.ndimage as ndi
from scipy.constants import c, epsilon_0
from cycler import cycler
from tqdm import tqdm
import numba


# %% Define helper functions

def cartesian_to_polar(image, origin=None, dr=1, dt=None):
    """
    Converts a Cartesian image to polar coordinates.
    """
    ny, nx = image.shape
    if origin is None:
        origin = (ny // 2, nx // 2)

    x, y = np.meshgrid(np.arange(nx) - origin[1], np.arange(ny) - origin[0])
    r, theta = np.hypot(x, y), np.arctan2(y, x)

    nr = int(np.ceil(r.max() / dr))
    if dt is None:
        nt = max(nx, ny)
    else:
        nt = int(np.ceil((2 * np.pi) / dt))

    r_i = np.linspace(0, r.max(), nr)
    theta_i = np.linspace(-np.pi, np.pi, nt)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    X_cart, Y_cart = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
    row_idx = origin[0] + Y_cart
    col_idx = origin[1] + X_cart
    coords = np.vstack((row_idx.ravel(), col_idx.ravel()))

    polar_image = map_coordinates(image, coords, order=1, mode='nearest').reshape(nr, nt)
    return polar_image, r_grid, theta_grid

def plot_radial_profile(field, name, window, N):
    """
    Plots the radial profile of the intensity sqrt(|field|^2)
    """
    plt.figure(figsize=(10, 6))
    intensity_cart  = np.abs(field) ** 2
    polar, r_grid, _ = cartesian_to_polar(intensity_cart)
    mid = polar.shape[1] // 2
    window_size = 10
    start = mid - window_size // 2
    end = mid + window_size // 2 + (window_size % 2)
    I_r = np.mean(polar[:, start:end], axis=1)
    sqrt_I_r = np.sqrt(I_r)
    pixel_size = window / N
    r_values = r_grid[:, 0]
    plt.plot(r_values, sqrt_I_r, linewidth=3)
    plt.xlabel(r"$r$ (mm)", fontsize=22)
    plt.ylabel(r"$\sqrt{I}$", fontsize=22)
    plt.title(name, fontsize=24)
    plt.grid(True)
    plt.savefig(f"{name}.png", dpi=800)
    plt.close()

def Vortex(XX, YY, pos=(0, 0), xi=40e-6, ell=1):
    """
    Computes the vortex injection factor.
    Returns the amplitude and phase to be multiplied with the field.
    Note: Using xi=40e-6 so that the vortex core spans ~4 pixels.
    """
    # Shift the coordinate system if needed
    YY_shift = cp.asarray(YY) + pos[1]
    XX_shift = cp.asarray(XX) + pos[0]
    r = cp.hypot(YY_shift, XX_shift)
    theta = cp.arctan2(YY_shift, XX_shift)
    # Ideal vortex: smooth core (with healing length xi) and charge ell.
    amp = r / cp.sqrt(r**2 + (xi / 0.83)**2)
    phase = ell * theta
    return amp, phase

# %% Simulation Parameters and Setup
ell = 1
xi = 17e-6        # set healing length so that the vortex core spans ~4 pixels
N = 2048
window = 20e-3    # cell size in meters
L_cm = 21  #1e-3     # cell length in centimeters
L = L_cm * 1e-2   # convert to meters

# Other simulation parameters
noise_amp = 0.0
puiss = 3.2
waist = 1.7e-3
n20 = -9.5e-11
Isat0 = 6e5
d_real = 2 * 3.15e-6
n2 = n20
Isat = Isat0
k0 = 2 * np.pi / 780e-9
T = 0.07
alpha = -np.log(T) / L
nl_length = 0

# Define the spatial grid through the NLSE simulation instance.
simu = NLSE(alpha, power=puiss, window=window, n2=n2, V=None, L=L, NX=N, NY=N, nl_length=nl_length)
simu.delta_z = 1e-4 # 1e-7  # propagation step
simu.n2 = n2
simu.I_sat = Isat

# Create output folder for saving data
base_path = "MarcY_Leon/DATA/Atoms/2025/Multicharge/Simulations/0226_multi_quench"
folder_name = f"vortex_extraction_l{ell}_L{L}"
full_path = os.path.join(base_path, folder_name)
os.makedirs(full_path, exist_ok=True)

# Number of saved samples along z.
N_samples = L_cm #3 #here one sample per cm
z_samples = cp.zeros(N_samples + 1)
E0_samples = cp.zeros((N_samples + 1, N, N), dtype=cp.complex64)
E_vortex_samples = cp.zeros((N_samples + 1, N, N), dtype=cp.complex64)
N_steps = int(round(L / simu.delta_z))
save_every = N_steps // N_samples
tau_samples = cp.zeros(N_samples + 1)
# %% Run background simulation (without vortex injection)
print("Simulating background field...")

# E0 = cp.exp(-((simu.XX**2 + simu.YY**2) / waist**2))
# E0 += cp.random.normal(0, noise_amp/2, E0.shape) + 1j*cp.random.normal(0, noise_amp/2, E0.shape)

# %% Initialize simulation
LL = L
E0 = (
    cp.exp(
        -(cp.asarray(simu.XX) ** exponent + cp.asarray(simu.YY) ** exponent)
        / waist**exponent)
    + 1j * 0
)

E0 += cp.random.normal(0, noise_amp / 2, E0.shape) + 1j * cp.random.normal(
    0, noise_amp / 2, E0.shape   ) #add noise 

def callback_sample_2(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    if i % save_every == 0:
        #print(f"Saving sample at step {i} (z = {z})")
        rho_ref = np.abs(A.get())
        phi_ref = np.angle(A.get())
        threshold = 2e-2
        mask = rho_ref < threshold * np.max(rho_ref)
        phi_ref_masked = np.ma.array(phi_ref, mask=mask)
        phi_ref_unwrapped = restoration.unwrap_phase(
            phi_ref_masked, wrap_around=(True, True)
        )
        tau = np.abs(np.nanmax(phi_ref_unwrapped) - np.nanmin(phi_ref_unwrapped))
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z
        tau_samples[i // save_every] = tau

E_background = simu.out_field(
    E0,
    LL,
    callback=callback_sample_2,
    callback_args=(E0_samples, z_samples),
    plot=False,
    precision="single",
)


# %% Compute the ideal vortex injection factor at z=0
amp_ideal, phase_ideal = Vortex(simu.XX, simu.YY, ell=ell, xi=xi)
vortex_factor = amp_ideal * cp.exp(1j * phase_ideal)
# For a quick check, you can plot the radial profile:
plot_radial_profile(vortex_factor.get(), "Ideal_Vortex", window, N)

# %% Run vortex simulation (background multiplied by vortex factor)
print("Simulating vortex field (vortex injection only)...")
E1 = E0.copy()
E1 *= vortex_factor  # apply the vortex injection factor

E_vortex = simu.out_field(
    E1,
    L,
    callback=lambda simu, A, z, i: (
        E_vortex_samples.__setitem__(i // save_every, A) or None
        if (i % save_every == 0) else None
    ),
    plot=False,
    precision="single",
    normalize=True,
)

# %% Extract unwanted perturbations at each saved z.
# The idea is: in an ideal simulation the vortex field would be E_ideal = (background evolution) * vortex_factor.
# In practice, numerical imperfections cause extra (unwanted) contributions.
# We compute: delta = E_vortex - E_ideal, and we set the vortex core (central 2x2 pixels) to zero.
unwanted_perturbations = {}  # dictionary keyed by z (in meters)

# Convert saved samples from GPU to CPU (if needed)
E0_samples_cpu = cp.asnumpy(E0_samples)
E_vortex_samples_cpu = cp.asnumpy(E_vortex_samples)
z_samples_cpu = cp.asnumpy(z_samples)

# Compute the ideal field at each step:
vortex_factor_cpu = cp.asnumpy(vortex_factor)

# Define the indices for the vortex core (central 2x2 pixels)
cx, cy = N // 2, N // 2
core_slice = (slice(cx-1, cx+1), slice(cy-1, cy+1))

print(E0_samples_cpu.shape[0])
for i in range(E0_samples_cpu.shape[0]):
    # Expected ideal vortex field at z_i:
    E_ideal = E0_samples_cpu[i] * vortex_factor_cpu# (0.9*vortex_factor_cpu+0.1 * np.exp(1j * k * cp.asnumpy(simu.YY))) #assmes that vorte doesn't change a lot

    # Difference: unwanted perturbation
    delta = E_vortex_samples_cpu[i] - E_ideal
    
    #delta = (np.abs(E_vortex_samples_cpu[i]) - np.abs(E_ideal))*np.angle(E_vortex_samples_cpu[i]/E_ideal)
    
    # Mask out the vortex core by setting the central 2x2 pixels to zero.
    delta[core_slice] = 0.0

    #z_val = z_samples_cpu[i]
    #print(z_val) #-->the old problem was here, it just takes the final z, need callback function in a cleaner way
    #unwanted_perturbations[z_val] = delta

    unwanted_perturbations[i/100] = delta #[i/100] from cm to m
print(len(unwanted_perturbations))

# Save the unwanted perturbations dictionary to file.
save_filename = os.path.join(full_path, "unwanted_perturbations.pkl")
with open(save_filename, "wb") as f:
    pickle.dump(unwanted_perturbations, f)

print(f"Unwanted perturbations saved to {save_filename}")
print("Done.")