import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint


# Global constants
G = 6.674e-8                      # Gravitational constant - cgs units
c = 3.0e10                        # Speed of light - cm/s
R = 1.0e6                         # Magnetar radius - cm
Msol = 1.99e33                    # Solar mass - grams
M = 1.4 * Msol                    # Magnetar mass - grams
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of Inertia
GM = G * M
t = np.logspace(0.0, 6.0, num=10001, base=10.0)


# Calculate initial conditions to pass to odeint
def init_conds(MdiscI, P):
    """
Function to convert a disc mass from solar masses to grams and an initial spin
period in milliseconds into an angular frequency.

    :param MdiscI: disc mass - solar masses
    :param P: initial spin period - milliseconds
    :return: an array containing the disc mass in grams and the angular freq.
    """
    Mdisc0 = MdiscI * Msol                 # Disc mass
    omega0 = (2.0 * np.pi) / (1.0e-3 * P)  # Angular frequency

    return np.array([Mdisc0, omega0])


# Model to be passed to odeint to calculate Mdisc and omega
def odes(y, t, B, MdiscI, RdiscI, epsilon, delta, n, alpha=0.1, cs7=1.0, k=0.9):
    """
Function to be passed to ODEINT to calculate the disc mass and angular frequency
over time.

    :param y: output from init_conds
    :param t: time points to solve equations for
    :param B: magnetic field strength - 10^15 G
    :param MdiscI: initial disc mass - solar masses
    :param RdiscI: disc radius - km
    :param epsilon: timescale ratio
    :param delta: mass ratio
    :param n: propeller "switch-on"
    :param alpha: sound speed prescription
    :param cs7: sound speed in disc - 10^7 cm/s
    :param k: capping fraction
    :return: time derivatives of disc mass and angular frequency to be integrated
             by ODEINT
    """
    # Initial conditions
    Mdisc, omega = y

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Disc radius
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic Dipole Moment
    M0 = delta * MdiscI * Msol             # Global Mass Budget
    tfb = epsilon * tvisc                  # Fallback timescale

    # Radii -- Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc) **
          (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    # Cap Alfven radius
    if Rm >= (k * Rlc):
        Rm = k * Rlc

    w = (Rm / Rc) ** (3.0 / 2.0)  # Fastness Parameter

    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotation parameter

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)   # Accretion
    Mdotfb = (M0 / tfb) * ((t + tfb) / tfb) ** (-5.0 / 3.0)  # Fallback
    Mdotdisc = Mdotfb - Mdotprop - Mdotacc

    if rot_param > 0.27:
        Nacc = 0.0  # Prevents magnetar break-up
    else:
        # Accretion torque
        if Rm >= R:
            Nacc = ((GM * Rm) ** 0.5) * (Mdotacc - Mdotprop)
        else:
            Nacc = ((GM * R) ** 0.5) * (Mdotacc - Mdotprop)

    omegadot = (Nacc + Ndip) / I  # Angular frequency time derivative

    return np.array([Mdotdisc, omegadot])


# Function to calculate a model light curve
def model_lc(pars, dipeff=1.0, propeff=1.0, f_beam=1.0, n=10.0, alpha=0.1,
             cs7=1.0, k=0.9):
    """
Function to calculate the model light curve for a given set of parameters.

    :param pars: list of input parameters including:
                   * B: magnetic field strenght - 10^15 G
                   * P: initial spin period - milliseconds
                   * MdiscI: initial disc mass - solar masses
                   * RdiscI: disc radius - km
                   * epsilon: timescale ratio
                   * delta: mass ratio
    :param dipeff: dipole energy-to-luminosity conversion efficiency
    :param propeff: propeller energy-to-luminosity conversion efficiency
    :param f_beam: beaming factor
    :param n: propeller "switch-on"
    :param alpha: sound speed prescription
    :param cs7: sound speed in disc - 10^7 cm/s
    :param k: capping fraction
    :return: an array containing total, dipole and propeller luminosities in
             units of 10^50 erg/s
    """
    B, P, MdiscI, RdiscI, epsilon, delta = pars  # Separate out variables
    y0 = init_conds(MdiscI, P)                   # Calculate initial conditions

    # Solve equations
    soln, info = odeint(odes, y0, t, args=(B, MdiscI, RdiscI, epsilon, delta,
                        n), full_output=True)
    if info["message"] != "Integration successful.":
        return "flag"

    # Split solution
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    # Constants
    Rdisc = RdiscI * 1.0e5                 # Disc radius - cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale - s
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic dipole moment

    # Radii -- Alfven, Corotation and Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0/ 3.0)
    Rlc = c / omega
    Rm = np.where(Rm >= (k * Rlc), (k * Rlc), Rm)

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW               # Rotational parameter

    # Efficiencies and Mass Flow Rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)  # Propelled
    Mdotacc = eta1 * (Mdisc / tvisc)   # Accreted

    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.27:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = ((GM * Rm[i]) ** 0.5) * (Mdotacc[i] - Mdotprop[i])
            else:
                Nacc[i] = ((GM * R) ** 0.5) * (Mdotacc[i] - Mdotprop[i])

    # Dipole luminosity
    Ldip = (mu ** 2.0 * omega ** 4.0) / (6.0 * (c ** 3.0))
    Ldip = np.where(Ldip <= 0.0, 0.0, Ldip)
    Ldip = np.where(np.isfinite(Ldip), Ldip, 0.0)

    # Propeller luminosity
    Lprop = (-1.0 * Nacc * omega) - ((GM / Rm) * eta2 * (Mdisc / tvisc))
    Lprop = np.where(Lprop <= 0.0, 0.0, Lprop)
    Lprop = np.where(np.isfinite(Lprop), Lprop, 0.0)

    # Total luminosity
    Ltot = f_beam * ((dipeff * Ldip) + (propeff * Lprop))

    return np.array([Ltot, Lprop, Ldip]) / 1.0e50


def make_directories(dirname):

    basename = os.path.join("data", "synthetic_dataset")
    if not os.path.exists(basename):
        os.mkdir(basename)

    if not os.path.exists(os.path.join(basename, dirname)):
        os.mkdir(os.path.join(basename, dirname))

    return os.path.join(basename, dirname)


# GRB types and parameters
grb_names = ["Humped", "Classic", "Sloped", "Stuttering"]
grbs = {"Humped": [1.0, 5.0, 1.e-3, 100.0, 0.1, 1.0],
        "Classic": [1.0, 5.0, 1.e-3, 1000.0, 0.1, 1.0],
        "Sloped": [1.0, 1.0, 1.e-3, 100.0, 10.0, 10.0],
        "Stuttering": [1.0, 5.0, 1.e-5, 100.0, 0.1, 100.0]}

for grb in grb_names:

    # Make a sub-directory for each burst type
    filepath = make_directories(grb)

    # Grab the GRB parameters and generate the model light curve
    pars = grbs[grb]
    Ltot, _, _ = model_lc(pars)

    # Generate random indices along the length of the data arrays
    indx = np.sort(np.random.randint(0, len(t), size=100))
    x = t[indx]
    y = Ltot[indx]

    # Generate some Gaussian noise in y
    yerr = 0.25 * y
    y += np.random.normal(0.0, scale=yerr, size=100)

    # Create a data frame of dataset
    df = pd.DataFrame({"x": x, "y": y, "yerr": yerr})

    # Write data frame to a CSV file
    df.to_csv(os.path.join(filepath, "{}.csv".format(grb)), index=False)
