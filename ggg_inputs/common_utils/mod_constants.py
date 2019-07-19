import os

avogadro = 6.022141e23  # molec./mol
mass_dry_air = 28.9644e-3  # kg/mol
mass_h2o = 18.01534e-3  # kg/mol

ratio_molec_mass = 28.964/18.02	 # Ratio of Molecular Masses (Dry_Air/H2O)

trop_flag = 1
middleworld_flag = 2
overworld_flag = 3

_mydir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(_mydir, '..', 'data'))

days_per_year = 365.25

priors_version = '1.0.0'
