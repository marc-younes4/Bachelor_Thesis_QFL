import contrast as contrast
import velocity as velocity
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from NLSE import NLSE
from scipy import signal, optimize
from cycler import cycler
from scipy.constants import c, epsilon_0
from skimage import restoration
from matplotlib import colors
from PIL import Image
import scipy.ndimage as ndi
from scipy.integrate import solve_ivp
from PyQt5 import QtWidgets
from matplotlib import animation
from IPython.display import HTML
import matplotlib.ticker as ticker
from tqdm import tqdm
import numba
import os
import matplotlib.cm as cm
from scipy.ndimage import map_coordinates
import pickle 

ell = 1 #vortex charge
emm = 2 #mode charge - to add to E1 if wanted 
k = 111 #plane wave frequency - to add to E1 if wanted  
N = 2048 
window = 20e-3 #15e-3 #window in meters, ideally >> waist size, otherwise reflective effects due to fft arise, 
                      # use absorbing boundaries by adding to NLSE(..., absorption_width = 0.3*window) and renaming the nlse_absorbing_boundaries.py file nlse.py 
p = N/window #pixel size (to get real units, need to mulitply by pixel size)
L_cm = 21 #length of the cell in cm
L = L_cm * 1e-2 #length of the cell in meters
noise_amp = 0 #0.01 #0.001
zoom = 300 #pixel shown will be zoom*2 around the center
xi = 17e-6 #healing length in meters
T = 0.07
alpha = -np.log(T) / L #attenuation coefficient
# %% Inputs
n20 = -9.5e-11
Isat0 = 6e5
waist = 1.7e-3
d_real = 2 * 3.15e-6
puiss = 3.2
I0 = 2 * puiss / (np.pi * waist**2)
n2 = n20
Isat = Isat0  # saturation intensity in W/m^2
k0 = 2 * np.pi / 780e-9 #m-1
nl_lenght = 0

exponent = 2
mach = 0.0

pdist = 2 * xi
ddist = 8 * xi
pos = np.array([-250e-6, 0])
position = 4

pos_array = np.zeros((N, N))

#Where to save
base_path = "MarcY_Leon/DATA/Atoms/2025/Multicharge/Simulations/0226_multi_quench"

# Construct the folder name dynamically
folder_name = f"folder_l{ell}_L{L_cm}cm_noise{noise_amp}" 

# Combine the base path and folder name
full_path = os.path.join(base_path, folder_name)

# Create the folder
os.makedirs(full_path, exist_ok=True) 
 

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


def plot_radial_profile(funct, func):
    """
    Plots the radial profiles of sqrt(intensity) and phase phi(r) in a two-panel subplot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    pixel_size = window / N
    # --------------------------
    # Compute radial profile for sqrt(intensity)
    intensity_cart = np.abs(funct)**2
    intensity_polar, r_grid, _ = cartesian_to_polar(intensity_cart)
    # Choose a window of angular columns around the middle:
    mid = intensity_polar.shape[1] // 2
    window_size = 10
    start = mid - window_size // 2
    end = mid + window_size // 2 + (window_size % 2)
    
    I_r = np.mean(intensity_polar[:, start:end], axis=1) #azimuthal average over a window of 10 rows
    sqrt_I_r = np.sqrt(I_r)
    r_values = r_grid[:, 0]*pixel_size*1000
    
    axs[0].plot(r_values, sqrt_I_r, linewidth=3)
    axs[0].set_xlabel(r"$r$ (mm)", fontsize=22)
    axs[0].set_ylabel(r"$\sqrt{I}$", fontsize=22)
    axs[0].tick_params(labelsize=16)
    axs[0].grid(True)
    axs[0].set_title(r"Radial profile: $\sqrt{I}$")
    
    # --------------------------
    # Compute radial profile for phase
    phase_cart = np.angle(funct)
    phase_polar, r_grid_phase, _ = cartesian_to_polar(phase_cart)
    phi_r = np.mean(phase_polar, axis=1) #full azimuthal average
    
    axs[1].plot(r_values, phi_r, linewidth=3)
    axs[1].set_xlabel(r"$r$ (mm)", fontsize=22)
    axs[1].set_ylabel(r"$\phi(r)$", fontsize=22)
    axs[1].tick_params(labelsize=16)
    axs[1].grid(True)
    axs[1].set_title("Radial profile: Phase")
    
    fig.tight_layout()
    save_path = os.path.join(base_path, folder_name, f"{func}.png")
    plt.savefig(save_path, dpi=800)
    plt.close()
    return 

# %% Vortex function
def Vortex(XX, YY, pos=(0, 0), xi=40e-6, ell=ell):
    YY = cp.asarray(simu.YY) + pos[1]
    XX = cp.asarray(simu.XX) + pos[0]
    r = cp.hypot(YY, XX)
    theta = cp.arctan2(YY, XX)
    Psi = r / cp.sqrt(r**2 + (xi / 0.83)**2) * cp.exp(1j * ell * theta)
    #plot_radial_profile(Psi.get(), 'Psi')

    Amp = cp.abs(Psi)
    Phase = cp.angle(Psi)

    return Amp, Phase

def callback_sample_1(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    if i % save_every == 0:
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z


def callback_sample_2(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
) -> None:
    if i % save_every == 0:
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


# %% Initialize simulation
LL = L

simu = NLSE(
    alpha,
    power=puiss, 
    window=window,
    n2=n2,
    V=None,
    L=LL,
    NX=N,
    NY=N,
    nl_length=nl_lenght,
)
simu.delta_z = 1e-4
simu.n2 = n2
simu.I_sat = Isat

N_samples = L_cm
z_samples = cp.zeros(N_samples + 1)  # +1 to account for the 0th step 
qs_samples = cp.zeros((N_samples + 1, 2, 2))
tau_samples = cp.zeros(N_samples + 1)
vort_rads = cp.zeros(N_samples + 1)
E0_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
Evort_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
E_samples = cp.zeros((N_samples + 1, N, N), dtype=np.complex64)
N_steps = int(round(LL / simu.delta_z))
save_every = N_steps // N_samples

# %% Compute background field

E0 = (cp.exp(-(cp.asarray(simu.XX) ** exponent + cp.asarray(simu.YY) ** exponent)/ waist**exponent)+ 1j * 0) #Gaussian Background

E0 += cp.random.normal(0, noise_amp / 2, E0.shape) + 1j * cp.random.normal(0, noise_amp / 2, E0.shape) #add noise 

E_background = simu.out_field(
    E0,
    LL,
    callback=callback_sample_2,
    callback_args=(E0_samples, z_samples),
    plot=False,
    precision="single",
)

# %% Compute field
amp, phase = Vortex(simu.XX, simu.YY, ell=ell, xi=xi )

E1 = E0.copy()
E1 *= amp #0.9 * amp
E1 *= cp.exp(1j * phase)
#E1 += 0.1 * cp.exp(1j * k * cp.asarray(simu.YY))                                #we add a plane wave 
#E1 += 0.1 *cp.exp(1j*emm*cp.arctan2(cp.asarray(simu.YY), cp.asarray(simu.XX)))  #we add an e^im\theta mode

print(f"Simulating E with {N_samples} samples...")
E = simu.out_field(
    E1,
    LL, #simu.deltaz
    callback=callback_sample_1,
    callback_args=(E_samples, z_samples),
    plot=False,
    precision="single",
    normalize=True,
)

def fmt(x, pos) -> str:
    a, b = "{:.0e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


E_samples_11 = E_samples[
    :,
    E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom,
    E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom,
]
E_samples_00 = E0_samples[
    :,
    E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
    E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
] 

for zzz in range(0, 1, 1):

    E_samples_0 = E_samples_00.get()[zzz, :, :] #background 
    E_samples_1 = E_samples_11.get()[zzz, :, :] #background + data 

    rho1 = np.abs(E_samples_1) ** 2
    phi1 = np.angle(E_samples_1)
    rho0 = np.abs(E_samples_0) ** 2
    phi0 = np.angle(E_samples_0)
    phi = np.angle(E_samples_1/E_samples_0)  #to remove backgrnd
    rho = rho1 - rho0                        #to remove backgrnd


    fig, ax = plt.subplots(1, 2, figsize=(15, 5), layout="constrained", dpi=800)

    im0 = ax[0].imshow(rho, cmap="RdGy", interpolation="none")
    im1 = ax[1].imshow(phi, cmap="twilight_shifted", interpolation="none")
    fig.colorbar(
        im0, ax=ax[0], label="Intensity", format=ticker.FuncFormatter(fmt), shrink=0.6
    )
    fig.colorbar(im1, ax=ax[1], label="Phase", shrink=0.6)
    #fig.suptitle(rf"Field at the end of the cell for k={np.round(k,2)} $m^{{-1}}$, l = {ell}")
    fig.suptitle(rf"{folder_name}, l = {ell}")
    ax[0].set_title(f"Intensity at z={zzz} cm")
    ax[1].set_title(f"Phase at z={zzz} cm")
    plt.savefig(f"MarcY_Leon/DATA/Atoms/2025/Multicharge/Simulations/0226_multi_quench/{folder_name}/{zzz}.png", dpi=800) #k{k}_l{ell}_noise{noise_amp}/{zzz}.svg", dpi=800)
    plt.close()
    print(zzz)

############################################################################################################

#####                                  plot radial profile for different z
# plt.figure(figsize=(10, 6))

# z_indices = [1,10]
# for i_z, z_idx in enumerate(z_indices):

#     intensity_cart    = np.abs(E_samples_11.get()[z_idx, :, :]) ** 2
#     intensity_cart_bg = np.abs(E_samples_00.get()[z_idx, :, :]) ** 2

#     intensity_polar, r_grid, _ = cartesian_to_polar(intensity_cart)
#     intensity_polar_bg, r_grid_bg, _ = cartesian_to_polar(intensity_cart_bg)

#     # Instead of taking one column, take a window of 6 columns around the middle
#     mid = intensity_polar.shape[1] // 2
#     window_size = 4
#     start = mid - window_size // 2
#     end = mid + window_size // 2 + (window_size % 2)

#     I_r = np.mean(intensity_polar[:, start:end], axis=1)
#     I_r_bg = np.mean(intensity_polar_bg[:, start:end], axis=1)

#     # I_r = (I_r - np.min(I_r)) / (np.max(I_r) - np.min(I_r))
#     # I_r_bg = (I_r_bg - np.min(I_r_bg)) / (np.max(I_r_bg) - np.min(I_r_bg))

#     sqrt_I_r = np.sqrt(I_r)
#     sqrt_I_r_bg = np.sqrt(I_r_bg)

#     intensity_diff = sqrt_I_r - sqrt_I_r_bg

#     pixel_size = window / N  # Assumes that 'window' and 'N' are defined
#     # Multiply by 1000 to convert from meters to millimeters
#     r_values = r_grid[:, 0] * pixel_size * 1000

#     color = cm.Blues(0.3 + 0.4 * i_z / (len(z_indices) - 1))
#     plt.plot(r_values, intensity_diff, linewidth=3, color=color,
#              label=f"z = {z_idx} cm")

# plt.xlabel(r"$r$ (mm)", fontsize=22)
# #plt.ylabel(r"$\sqrt{I}-\sqrt{I_{\mathrm{background}}}$", fontsize=22)
# plt.ylabel(r"$\sqrt{I}$", fontsize=22)
# plt.tick_params(labelsize=16)
# plt.grid(True)
# plt.legend()
# plt.savefig(f"MarcY_Leon/DATA/Atoms/2025/Multicharge/Simulations/0226_multi_quench/{folder_name}/radial profile.png", dpi=800) #k{k}_l{ell}_noise{noise_amp}/{zzz}.svg", dpi=800)
# plt.close()

# print('Done')

