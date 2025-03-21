
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
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
from scipy.ndimage import map_coordinates

ell = 1 #vortex charge
emm = 2 # m for e^im\theta mode 
k = 111 #plane wave frequency
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

def Vortex(XX, YY, pos=(0, 0), xi=10e-6, ell=ell):
    YY = cp.asarray(simu.YY) + pos[1]
    XX = cp.asarray(simu.XX) + pos[0]
    r = cp.hypot(YY, XX)
    theta = cp.arctan2(YY, XX)
    Psi = r / cp.sqrt(r**2 + (xi / 0.83) ** 2) * cp.exp(1j * ell * theta)
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


def callback_sample_3(
    simu: NLSE,
    A: np.ndarray,
    z: float,
    i: int,
    E_samples: np.ndarray,
    z_samples: np.ndarray,
    vort_rads: np.ndarray,
) -> None:
    import scipy.optimize as opt

    def Vortex2D(mn, y_val, x_val, xi) -> np.ndarray:
        y, x = mn
        r = np.sqrt((x - x_val) ** 2 + (y - y_val) ** 2)
        theta = np.arctan2(y, x)
        Psi = r / np.sqrt(r**2 + (2 * xi) ** 2)
        Amp = np.abs(Psi)
        return Amp.ravel()

    if i % save_every == 0:
        rho_vort = cp.abs(A)
        phi_flat = cp.angle(A)
        vort = velocity.vortex_detection(phi_flat.get(), plot=False, r=1)
        vort[:, 0] -= phi_flat.shape[-1] // 2
        vort[:, 1] -= phi_flat.shape[-1] // 2
        vort_select = np.logical_not(vort[:, 0] ** 2 + vort[:, 1] ** 2 > (300) ** 2)
        vortices = vort[vort_select, :]
        center_x = phi_flat.shape[-1] // 2 + vortices[0][0]
        center_y = phi_flat.shape[-1] // 2 + vortices[0][1]
        vort = cp.array([center_x, center_y])
        window_m = 15
        rho_zoom = rho_vort[
            int(center_y - window_m) : int(center_y + window_m),
            int(center_x - window_m) : int(center_x + window_m),
        ]
        rho_zoom = rho_zoom.get()
        x = np.arange(rho_zoom.shape[1])
        y = np.arange(rho_zoom.shape[0])
        X, Y = np.meshgrid(x, y)
        inital_guess = (window_m, window_m, 5)
        popt, pcov = opt.curve_fit(Vortex2D, (Y, X), rho_zoom.ravel(), p0=inital_guess)
        vortex_reshape = Vortex2D((Y, X), *popt).reshape(window_m * 2, window_m * 2)
        vort_rad = popt[2] * d_real
        E_samples[i // save_every] = A
        z_samples[i // save_every] = z
        vort_rads[i // save_every] = vort_rad

def fmt(x, pos) -> str:
    a, b = "{:.0e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


#Project to polar adapted from Killian Guerrero and https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system


def reproject_image_into_polar(data, origin=None, Jacobian=False,
                               dr=1, dt=None):
    """
    Reprojects a 2D numpy array (**data**) into a polar coordinate system,
    with the pole placed at **origin** and the angle measured clockwise from
    the upward direction. The resulting array has rows corresponding to the
    radial grid, and columns corresponding to the angular grid.

    Parameters
    ----------
    data : 2D np.array
        the image array
    origin : tuple or None
        (row, column) coordinates of the image origin. If ``None``, the
        geometric center of the image is used.
    Jacobian : bool
        Include `r` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        radial coordinate spacing for the grid interpolation.
        Tests show that there is not much point in going below 0.5.
    dt : float or None
        angular coordinate spacing (in radians).
        If ``None``, the number of angular grid points will be set to the
        largest dimension (the height or the width) of the image.

    Returns
    -------
    output : 2D np.array
        the polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of angular coordinates

    Notes
    -----
    Adapted from:
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (ny // 2, nx // 2)
    else:
        origin = list(origin)
        # wrap negative coordinates
        if origin[0] < 0:
            origin[0] += ny
        if origin[1] < 0:
            origin[1] += nx

    # Determine what the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = int(np.ceil((r.max() - r.min()) / dr)) #we re going from a square to a circle where not all areas are defined 

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = int(np.ceil((theta.max() - theta.min()) / dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Convert the r and theta grids to Cartesian coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    # then to a 2×n array of row and column indices for np.map_coordinates()
    rowi = (origin[0] - Y).flatten()
    coli = (X + origin[1]).flatten()
    coords = np.vstack((rowi, coli))

    # Remap with interpolation
    # (making an array of floats even if the data has an integer type)
    #zi = map_coordinates(data, coords, output=complex)
    zi = map_coordinates(data, coords, output=np.float32)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output *= r_i[:, np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin=None):
    """
    Creates `x` and `y` coordinates for the indices in a numpy array, relative
    to the **origin**, with the `x` axis going to the right, and the `y` axis
    going `up`.

    Parameters
    ----------
    data : numpy array
        2D data. Only the array shape is used.
    origin : tuple or None
        (row, column). Defaults to the geometric center of the image.

    Returns
    -------
        x, y : 2D numpy arrays
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_y, origin_x = origin
        # wrap negative coordinates
        if origin_y < 0:
            origin_y += ny
        if origin_x < 0:
            origin_x += nx

    x, y = np.meshgrid(np.arange(float(nx)) - origin_x,
                       origin_y - np.arange(float(ny)))
    return x, y


def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar.

    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates

    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta


def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian.

    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)   # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y

def polar_to_rm(polar_image, theta_range=2*np.pi):
    """
    Perform a Fourier Transform on the polar image (r, theta) along the theta axis
    to obtain (r, m), where m is the conjugate variable of theta.

    Parameters
    ----------
    polar_image : 2D np.array
        The polar-transformed image with shape (nr, nt), where nr is the radial
        resolution and nt is the angular resolution.
    theta_range : float
        The total angular range in radians. Default is 2π (full circle).

    Returns
    -------
    rm_image : 2D np.array
        The transformed image in (r, m), where m is the conjugate variable of theta.
    m : np.array
        The angular frequencies corresponding to the Fourier transform.
    """
    nr, nt = polar_image.shape
    
    # Perform FFT along the theta (axis=1)
    rm_image = np.fft.fft(polar_image, axis=1) #/polar_image.shape[1] 
    
    # Compute the angular frequency array (m)
    m = np.fft.fftfreq(nt, d=theta_range/nt)  # Angular frequencies
    
    return rm_image, m 

def swap_blocks(matrix):
    # Get the total number of columns
    num_cols = matrix.shape[1]
    
    # Ensure the number of columns is even (N/2 must be an integer)
    if num_cols % 2 != 0:
        raise ValueError("Number of columns must be even to split into two equal blocks")
    
    # Calculate the midpoint (N/2)
    midpoint = num_cols // 2
    
    # Rearrange the matrix to swap block A and block B
    swapped_matrix = np.hstack((matrix[:, midpoint:], matrix[:, :midpoint]))
    
    return swapped_matrix

def sum_columns(matrix, m, l = ell):
    # Get the number of columns in the matrix
    num_columns = matrix.shape[1]
    
    # Check if m and -m exist in the column range
    if m >= num_columns or -m >= num_columns:
        return f"Invalid column index m: {m} for a matrix with {num_columns} columns."
    
    # Determine indices for column m and -m
    middle_index = num_columns // 2  # Column 0 is the middle
    col_m = middle_index + m + l #our "0" is now l
    col_neg_m = middle_index - m + l
    
    # Check for valid indices
    if col_m < 0 or col_m >= num_columns or col_neg_m < 0 or col_neg_m >= num_columns:
        return f"Invalid column index for m: {m} and -m: {-m}"
    
    # Sum the elements in column m and -m
    sum_m = matrix[:, col_m].sum()
    sum_neg_m = matrix[:, col_neg_m].sum()
    
    return sum_m, sum_neg_m

# Define r limit
r_limit = 500 # Only show results for r ≤ 700
def plot_differences_for_multiple_m(matrix, m_range=(-15, 15), filename="multiple_differences_plot.png", l = ell):
    # Get the number of columns in the matrix
    num_columns = matrix.shape[1]
    
    # Determine the middle index
    middle_index = num_columns // 2  # Column 0 is the middle
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    for m in range(m_range[0], m_range[1] + 1):
        # if m == ell:
        #     continue
        # Compute the indices for column m and -m
        col_m = middle_index + m + l #our "0" is now l
        col_neg_m = middle_index - m + l
        
        # Skip invalid indices
        if col_m < 0 or col_m >= num_columns or col_neg_m < 0 or col_neg_m >= num_columns:
            continue
        
        # Compute the difference at each row for column m and -m
        difference = matrix[:, col_m] - matrix[:, col_neg_m]
        
        # Plot the difference for each row (r)
        #plt.plot(range(matrix.shape[0]), difference, label=f"m-l = {m}")
        plt.plot(range(min(r_limit, matrix.shape[0])), difference[:r_limit], label=f"m-l = {m}") #lot until r = r_limit

    # Add title, labels, and legend
    plt.title(f"k = {np.round(k,2)}, l = {ell}, z = {zzz}cm")
    plt.xlabel("r")
    plt.ylabel(fr"$u_m'^2 - v_m'^2$")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    # Adjust layout to fit legend
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(filename, dpi=300)
    #print(f"Plot saved as {filename}")
    plt.close()

# %% Initialize simulation
LL = L

simu = NLSE(
    alpha,
    power=puiss, #was puiss = puiss but got TypeError: NLSE.__init__() got an unexpected keyword argument 'puiss'
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

E0 = (
    cp.exp(
        -(cp.asarray(simu.XX) ** exponent + cp.asarray(simu.YY) ** exponent)
        / waist**exponent
    )
    + 1j * 0
)

#add noise to bckgnd
noise_amp = 0
E0 += cp.random.normal(0, noise_amp / 2, E0.shape) + 1j * cp.random.normal(
     0, noise_amp / 2, E0.shape   )

E_background = simu.out_field(
    E0,
    LL,
    callback=callback_sample_2,
    callback_args=(E0_samples, z_samples),
    plot=False,
    precision="single",
)

rho_ref_samples = np.abs(E0_samples.get()) ** 2
rho_max = np.nanmax(rho_ref_samples[0])
rho_ref_samples = rho_ref_samples / rho_max * I0
phi_ref_samples = np.angle(E0_samples.get())

# %% Compute field
amp, phase = Vortex(simu.XX, simu.YY, ell=ell, xi=0.0000000001)

N_loops = 2
#print(f'Number of loops: {N_loops}')
for i in range(1, N_loops, 1): #loop over many frequencies
    #k = k0 / (500 * i)
    k = 8055.37
    print(k)
    E1 = E0.copy()
    E1 *= 0.9 * amp
    E1 *= cp.exp(1j * phase)
    #E1 += 0.1 * cp.exp(1j * k * cp.asarray(simu.YY)) #we take 10% as a plane wave 

    #print(f"Simulating E with {N_samples} samples..., loop {i}")
    E = simu.out_field(
        E1,
        LL, 
        callback=callback_sample_1,
        callback_args=(E_samples, z_samples),
        plot=False,
        precision="single",
        normalize=True,
    )
    E_samples_1 = E_samples[
         :,
         E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom,
         E_samples.shape[-1] // 2 - zoom : E_samples.shape[-1] // 2 + zoom ,
     ]
    E_samples_0 = E0_samples[
         :,
         E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
         E0_samples.shape[-1] // 2 - zoom : E0_samples.shape[-1] // 2 + zoom,
     ]

    final_z = 21
    for zzz in range(0,final_z):

        E_samples0 = E_samples_0.get()[zzz, :, :] #background at L = 20 cm only
        E_samples1 = E_samples_1.get()[zzz, :, :] #background + data at L = 20 cm only 

        E_phase = np.angle(E_samples1)  # Extract the phase from the electric field

        from scipy.ndimage import map_coordinates

        def cartesian_to_polar(image, origin=None, dr=1, dt=None):
            """
            Converts a Cartesian image to polar coordinates.
            """
            ny, nx = image.shape
            if origin is None:
                origin = (ny // 2, nx // 2)

            # Create Cartesian coordinate grids
            x, y = np.meshgrid(np.arange(nx) - origin[1], np.arange(ny) - origin[0])
            r, theta = np.hypot(x, y), np.arctan2(y, x)

            # Define the polar grid resolution
            nr = int(np.ceil(r.max() / dr))
            if dt is None:
                nt = max(nx, ny)
            else:
                nt = int(np.ceil((2 * np.pi) / dt))

            r_i = np.linspace(0, r.max(), nr)
            theta_i = np.linspace(-np.pi, np.pi, nt)
            theta_grid, r_grid = np.meshgrid(theta_i, r_i)

            # Convert polar grid to Cartesian indices
            X_cart, Y_cart = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
            row_idx = origin[0] + Y_cart
            col_idx = origin[1] + X_cart
            coords = np.vstack((row_idx.ravel(), col_idx.ravel()))

            # Interpolate intensity values onto the polar grid
            polar_image = map_coordinates(image, coords, order=1, mode='nearest').reshape(nr, nt)

            return polar_image, r_grid, theta_grid

        # Convert intensity to polar coordinates
        intensity_cartesian = np.abs(E_samples1) ** 2
        intensity_polar, r_grid, theta_grid = cartesian_to_polar(intensity_cartesian)

        # Compute azimuthally averaged intensity I(r)
        I_r = np.mean(intensity_polar, axis=1)

        # Convert velocity field to polar coordinates
        vx, vy = velocity.velocity(E_phase)

        # Convert vx, vy to polar coordinates
        vx_polar, _, _ = cartesian_to_polar(vx)
        vy_polar, _, _ = cartesian_to_polar(vy)

        # Compute radial and tangential velocities in polar coordinates
        vr_polar = vx_polar * np.cos(theta_grid) + vy_polar * np.sin(theta_grid)
        vtheta_polar = -vx_polar * np.sin(theta_grid) + vy_polar * np.cos(theta_grid)

        # Compute mean velocity profiles
        vr_mean = np.mean(vr_polar, axis=1)  #average over theta 
        vtheta_mean = np.mean(vtheta_polar, axis=1)

        # Compute sound speed profile c_s(r)
        c_s = c * np.sqrt(np.abs(n20) * I_r)

        # Define the radial limit
        r_limit = 500

        # Mask data to only include r <= 500
        mask = r_grid[:, 0] <= r_limit

        # Normalize all three quantities for consistent scaling
        vtheta_norm = vtheta_mean[mask] / np.max(np.abs(vtheta_mean))  # Normalize tangential velocity
        cs_norm = c_s[mask] / np.max(c_s)  # Normalize speed of sound

        # Corresponding radial values
        r_plot = r_grid[:, 0][mask]

        # Compute the difference function Δ(r)
        delta_r = vtheta_norm - cs_norm

        # Find zero crossings (where Δ(r) changes sign)
        zero_crossings = np.where(np.diff(np.sign(delta_r)))[0]

        # Interpolate to find more accurate crossing points
        intersections = []
        for idx in zero_crossings:
            r1, r2 = r_plot[idx], r_plot[idx + 1]
            d1, d2 = delta_r[idx], delta_r[idx + 1]
            r_intersect = np.interp(0, [d1, d2], [r1, r2])
            intersections.append(r_intersect)

        # Plot v_theta and c_s
        plt.figure(figsize=(8, 6))
        plt.plot(r_plot, vtheta_norm, label=r"$v_\theta(r)$ (Tangential Velocity)", linestyle="--")
        plt.plot(r_plot, cs_norm, label=r"$c_s(r)$ (Sound Speed)", linestyle=":")
        plt.xlabel("r (Radial Distance)")
        plt.ylabel("Normalized Values")
        plt.title("Intersection of $v_\\theta(r)$ and $c_s(r)$")
        plt.legend()
        plt.grid(True)

        # Mark intersections
        for r_intersect in intersections:
            plt.axvline(r_intersect, color='red', linestyle="--", alpha=0.7)
            plt.text(r_intersect, 0.1, f"{r_intersect:.1f}", color="red")

        # Save the plot
        plt.savefig(f'MarcY_Leon/DATA/Atoms/2025/Multicharge/Simulations/vr_vtheta_cs_{zzz}.png')

        # Print intersection points
        print(f"Intersections occur at r = {intersections}")


 