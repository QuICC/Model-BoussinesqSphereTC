import sys
import numpy as np
import validation_tools as vt

ref_dir, data_dir = vt.processArgv(sys.argv[1:])

results = []

# Tolerance per max rows
rows = list(range(0, 101, 10))
tols = [32, 65, 112, 130, 163, 179, 195, 260, 260, 238, 260]

prefixes = ['temperature', 'kinetic']
spectra = ['l', 'm', 'n']

# check energy and spectra
for prefix in prefixes:
    # Energy
    for r, t in zip(rows,tols):
        results.append(vt.tableTest(prefix + '_energy.dat', ref_dir, data_dir, r, tol = t, max_rows = r+1, perrow = True, max_firstcol = 1))

    # Spectra
    for mode in spectra:
        for r, t in zip(rows,tols):
            if mode == 'n':
                threshold = 1e-37
            else:
                threshold = -1
            results.append(vt.tableTest(prefix +  '_' + mode + f'_spectrum{r:04}.dat', ref_dir, data_dir, r, tol = t, percol = True, perrow = True, max_firstcol = 1, threshold = threshold))

# Nusselt number
for r, t in zip(rows,tols):
    results.append(vt.tableTest("nusselt.dat", ref_dir, data_dir, r, tol = t, max_rows = r+1))

# CFL
for r, t in zip(rows,tols):
    results.append(vt.tableTest("cfl.dat", ref_dir, data_dir, r, usecols=(0,1,3,4,5,6,7), tol = t, max_rows = r+1))

# Output test summary
vt.printSummary(results, rows, reftol = tols)
