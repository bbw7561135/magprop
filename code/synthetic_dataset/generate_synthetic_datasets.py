import os
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from magnetar.funcs import model_lc

grb_names = ["Humped", "Classic", "Sloped", "Stuttering"]
grbs = {
    "Humped": [1.0, 5.0, 1.e-3, 100.0, 0.1, 1.0],
    "Classic": [1.0, 5.0, 1.e-3, 1000.0, 0.1, 1.0],
    "Sloped": [1.0, 1.0, 1.e-3, 100.0, 10.0, 10.0],
    "Stuttering": [1.0, 5.0, 1.e-5, 100.0, 0.1, 100.0]
}

def make_directories(dirname):

    basename = os.path.join("data", "synthetic_dataset")
    if not os.path.exists(basename):
        os.mkdir(basename)

    if not os.path.exists(os.path.join(basename, dirname)):
        os.mkdir(os.path.join(basename,dirname))

for grb in grb_names:

    # Make a directory
    make_directories(grb)

    pars = grbs[grb]
    t, Ltot, Lprop, Ldip = model_lc(pars, dipeff=1.0, propeff=1.0)

    plt.loglog(t, Ldip, ':k')
    plt.loglog(t, Lprop, '--k')
    plt.loglog(t, Ltot, '-k')
