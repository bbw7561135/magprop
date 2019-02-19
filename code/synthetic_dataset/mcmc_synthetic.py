import os
import argparse
import numpy as np
import emcee as em
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator


# Global constants
G = 6.674e-8                      # Gravitational constant - cgs units
c = 3.0e10                        # Speed of light - cm/s
R = 1.0e6                         # Magnetar radius - cm
Msol = 1.99e33                    # Solar mass - grams
M = 1.4 * Msol                    # Magnetar mass - grams
I = (4.0 / 5.0) * M * (R ** 2.0)  # Moment of Inertia
GM = G * M
tarr = np.logspace(0.0, 6.0, num=10001, base=10.0)

names = ["$B$", "$P$", "$\log(M_{\\rm D,i})$", "$\log(R_{\\rm D})$",
         "$\log(\epsilon)$", "$\log(\delta)$"]

# True GRB parameters
truths = {"Humped": [1.0, 5.0, -3.0, 2.0, -1.0, 0.0],
          "Classic": [1.0, 5.0, -3.0, 3.0, -1.0, 0.0],
          "Sloped": [1.0, 1.0, -3.0, 2.0, 1.0, 1.0],
          "Stuttering": [1.0, 5.0, -5.0, 2.0, -1.0, 2.0]}

# Parameter limits: B, P, MdiscI, RdiscI, epsilon, delta
# MdiscI, RdiscI, epsilon and delta in log-space
lims = {
    "lower": np.array([1.0e-3, 0.69, -5.0, np.log10(50.0), -1.0, -5.0]),
    "upper": np.array([10.0, 10.0, -1.0, np.log10(2000.0), 3.0, np.log10(50.0)])
}


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
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
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


# Function that returns a model light curve
def model_lc(pars, xdata=None, dipeff=1.0, propeff=1.0, f_beam=1.0, n=10.0,
             alpha=0.1, cs7=1.0, k=0.9):
    """
Function to calculate a model gamma-ray burst, X-ray light curve based on input
parameters.

    :param pars: A list or array of input parameters in order: B, P, MdiscI,
                 RdiscI, epsilon, delta
    :param xdata: Optional array of time points for GRB data
    :param dipeff: Fractional dipole energy-to-luminosity conversion efficiency
    :param propeff: Fractional propeller energy-to-luminosity conversion
                    efficiency
    :param f_beam: Beaming fraction
    :param n: Propeller "switch-on" efficiency
    :param alpha: Sound speed prescription
    :param cs7: Sound speed in disc - 10^7 cm/s
    :param k: Capping fraction

    if xdata is not None:
        :return L: array of luminosity values at given time point in xdata
                   - 10^50 erg/s
    else:
        :return: an array containing tarr, Ltot, Lprop, Ldip in units of secs
                 and 10^50 erg/s
    """
    # Separate out parameters
    B, P, MdiscI, RdiscI, epsilon, delta = pars

    # Calculate initial conditions
    y0 = init_conds(MdiscI, P)

    # Solve the equations
    soln, info = odeint(odes, y0, tarr, args=(B, MdiscI, RdiscI, epsilon,
                        delta, n), full_output=True)
    # Return a flag if the integration was not successful
    if info["message"] != "Integration successful.":
        return "flag"

    # Separate out solution
    Mdisc = np.array(soln[:, 0])
    omega = np.array(soln[:, 1])

    # Convert constants
    Rdisc = RdiscI * 1.0e5                 # Disc radius - cm
    tvisc = Rdisc / (alpha * cs7 * 1.0e7)  # Viscous timescale - secs
    mu = 1.0e15 * B * (R ** 3.0)           # Magnetic dipole moment

    # Radii - Alfven, Corotation, Light Cylinder
    Rm = ((mu ** (4.0 / 7.0)) * (GM ** (-1.0 / 7.0)) * (((3.0 * Mdisc) / tvisc)
          ** (-2.0 / 7.0)))
    Rc = (GM / (omega ** 2.0)) ** (1.0 / 3.0)
    Rlc = c / omega
    # Cap Alfven radius to Light Cylinder radius
    Rm = np.where(Rm >= (k * Rlc), (k * Rlc), Rm)

    w = (Rm / Rc) ** (3.0 / 2.0)     # Fastness parameter
    bigT = 0.5 * I * (omega ** 2.0)  # Rotational energy
    modW = (0.6 * M * (c ** 2.0) * ((GM / (R * (c ** 2.0))) / (1.0 - 0.5 * (GM /
            (R * (c ** 2.0))))))     # Binding energy
    rot_param = bigT / modW          # Rotational parameter

    # Dipole torque
    Ndip = (-1.0 * (mu ** 2.0) * (omega ** 3.0)) / (6.0 * (c ** 3.0))

    # Efficiencies and mass flow rates
    eta2 = 0.5 * (1.0 + np.tanh(n * (w - 1.0)))
    eta1 = 1.0 - eta2
    Mdotprop = eta2 * (Mdisc / tvisc)
    Mdotacc = eta1 * (Mdisc / tvisc)

    # Accretion torque
    Nacc = np.zeros_like(Mdisc)
    for i in range(len(Nacc)):
        if rot_param[i] > 0.0:
            Nacc[i] = 0.0
        else:
            if Rm[i] >= R:
                Nacc[i] = ((GM * Rm[i]) ** 0.5 * (Mdotacc[i] - Mdotprop[i]))
            else:
                Nacc[i] = ((GM * R) ** 0.5 * (Mdotacc[i] - Mdotprop[i]))

    # Luminosities - Dipole, Propeller and Total
    Ldip = dipeff * (-1.0 * Ndip * omega)
    Ldip = np.where((Ldip <= 0.0), 0.0, Ldip)
    Ldip = np.where(np.isfinite(Ldip), Ldip, 0.0)

    Lprop = propeff * (-1.0 * Nacc * omega)
    Lprop = np.where((Lprop <= 0.0), 0.0, Lprop)
    Lprop = np.where(np.isfinite(Lprop), Lprop, 0.0)

    Ltot = f_beam * (Ldip + Lprop)

    # Return values based on xdata
    if xdata is not None:
        lum_func = interp1d(tarr, Ltot)
        L = lum_func(xdata)

        return L / 1.0e50

    else:
        return np.array([tarr, Ltot / 1.0e50, Lprop / 1.0e50, Ldip / 1.0e50])


# Function to calculate log-likelihood
def lnlike(pars, data):

    if len(pars) != 6:
        raise ValueError("pars must have length 6")

    pars = np.array(pars)          # Ensure pars is a numpy array
    pars[2:6] = 10.0 ** pars[2:6]  # Unlog required parameters

    # Separate data
    x = data["x"]
    y = data["y"]
    yerr = data["yerr"]

    # Calculate model
    mod = model_lc(pars, xdata=x)

    if mod is 'flag':  # Flag models that break odeint
        return -np.inf
    else:
        return -0.5 * np.sum(((y - mod) / yerr) ** 2.0)


# Function to calculate the log-prior
def lnprior(pars):

    if (pars >= lims["lower"]).all() & (pars <= lims["upper"]).all():
        return 0.0
    else:
        return -np.inf


# Function to calculate the log-posterior
def lnprob(pars, data, fbad):

    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf

    ll = lnlike(pars, data)
    if not np.isfinite(ll):

        with open(fbad, "a") as f:
            for i in range(len(pars)):
                if i == 5:
                    f.write("{:.8f}\n".format(pars[i]))
                else:
                    f.write("{:.8f},".format(pars[i]))

        return -np.inf

    return ll + lp


# If bad data file is empty, then remove it
def clean_up(fbad):
    try:
        pd.read_csv(fbad)
    except pd.errors.EmptyDataError:
        os.remove(fbad)
        print "File deleted: {}".format(fbad)


# Make a trace plot
def plot_trace(sampler, fplot, Npars, Nwalk, Nstep, Nburn):

    fig, axes = plt.subplots(Npars + 1, 1, sharex=True, figsize=(6, 8))

    for i in range(Nwalk):
        axes[0].plot(range(Nstep), sampler.lnprobability[i, :], c='gray',
                     alpha=0.4)
    axes[0].axvline(Nburn, c='r')
    axes[0].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].set_ylabel('$\ln (p)$', fontsize=12)

    for i in range(Npars):
        for j in range(Nwalk):
            axes[i + 1].plot(range(Nstep), sampler.chain[j, :, i], c='gray',
                             alpha=0.4)
        axes[i + 1].axvline(Nburn, c='r')
        axes[i + 1].yaxis.set_major_locator(MaxNLocator(4, prune='lower'))
        axes[i + 1].tick_params(axis='both', which='major', labelsize=10)
        axes[i + 1].set_ylabel(names[i], fontsize=12)

    axes[-1].set_xlabel('Model Number', fontsize=12)
    fig.tight_layout(h_pad=0.1)
    fig.savefig(fplot, dpi=720)
    plt.clf()


# Output the chains to various files
def output_chain(sampler, fchain, Npars, Nwalk, Nstep):
    # Get flat flat chains and probabilities
    chain = sampler.flatchain
    lnprobs = sampler.flatlnprobability

    # Create data frame
    df = pd.DataFrame({
        "B": chain[:, 0],
        "P": chain[:, 1],
        "log_MdiscI": chain[:, 2],
        "log_RdiscI": chain[:, 3],
        "log_epsilon": chain[:, 4],
        "log_delta": chain[:, 5],
        "log_prob": lnprobs
    })

    # Write to CSV file
    df.to_csv(fchain, index=False)

    # List of parameter names
    par_names = ["B", "P", "logMD", "logRD", "logeps", "logdelt"]

    # Output each parameter chain to it's own file
    for k in range(Npars):
        fpar = "".join([fchain.strip(".csv"), "_{}.csv".format(par_names[k])])
        dict = {}
        for i in range(Nwalk):
            dict[i] = sampler.chain[i,:,k]
        df = pd.DataFrame(dict, index=range(Nstep))
        df.to_csv(fpar)

    # Output the log-probability to it's own file
    dict = {}
    for i in range(Nwalk):
        dict[i] = sampler.lnprobability[i,:]
    df = pd.DataFrame(dict, index=range(Nstep))
    df.to_csv("".join([fchain.strip(".csv"), "_lnprob.csv"]))


# Create file names to save data to
def create_filenames(name, data_path, plot_path):
    # Create CSV file paths
    fdata = os.path.join(data_path, "{}.csv".format(name))
    fchain = os.path.join(data_path, "{}_chain.csv".format(name))
    fbad = os.path.join(data_path, "{}_bad.csv".format(name))

    # Initialise bad data file
    f = open(fbad, "w")
    f.close()

    # Create trace plot file path
    fplot = os.path.join(plot_path, "{}_trace.png".format(name))

    return fdata, fchain, fbad, fplot


# Make directories to save outputs to
def make_directories(dirname):
    # Find data directory
    path = os.path.join("data", "synthetic_datasets")
    data_path = os.path.join(path, dirname)

    # Make plots directories
    if not os.path.exists("plots"):
        os.mkdir("plots")

    plot_path = os.path.join("plots", "synthetic_datasets")

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    plot_path = os.path.join(plot_path, dirname)

    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    return data_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform MCMC optimisation on synthetic datasets"
    )

    parser.add_argument(
        "name",
        help="Name of the GRB type. Options are: Humped, Classic, Sloped, Stuttering."
    )

    return parser.parse_args()


def main():
    # Get command-line args
    args = parse_args()

    # Get file paths
    data_path, plot_path = make_directories(args.name)

    # Get file names
    fdata, fchain, fbad, fplot = create_filenames(
        args.name, data_path, plot_path
    )

    # Get the data
    data = pd.read_csv(fdata)

    # Initialise MCMC parameters
    Npars = 6     # Number of fitting parameters
    Nwalk = 20  #24    # Number of affine invariant walkers
    Nstep = 100  #2000  # Number of Monte Carlo steps to take
    Nburn = 5  #00   # Number of burn-in steps

    # Calculate initial position
    p0 = truths[args.name]
    pos = [p0 + 1.0e-4 * np.random.randn(Npars) for i in range(Nwalk)]

    # Initialise Ensemble Sampler
    sampler = em.EnsembleSampler(Nwalk, Npars, lnprob, args=(data, fbad),
                                 threads=3)

    # Run the sampler
    sampler.run_mcmc(pos, Nstep)
    print "Average acceptance fraction: {:.4f}".format(
        np.mean(sampler.acceptance_fraction))

    # Output the chain
    output_chain(sampler, fchain, Npars, Nwalk, Nstep)

    # Plot the trace
    plot_trace(sampler, fplot, Npars, Nwalk, Nstep, Nburn)

    # Clean up files
    clean_up(fbad)


if __name__ == "__main__":
    main()
