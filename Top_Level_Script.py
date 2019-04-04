import sys
sys.path.insert(0, '/home/amalatak/Documents/Programming/Python/Orbital_Mechanics/')
sys.path.insert(0, '/home/amalatak/Documents/University/Sp19/Orbital_Mechanics/Homework/')
import OmechUtils as utils 
import numpy as np 
import Orbital_Constants as cn 
import ADCS_utils as ADCS
import matplotlib.pyplot as plt
import spiceypy as spice


# orbit constants #

a = 7000e3    # m
i = 0         # rad
e = .01
RAAN = 0      # rad
argp = 0      # rad
nu = 0        # rad

JDstart = 0   # days
n_orbits = 1    # number of orbits
t_int = 10    # time interval

Body_xyz0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # starting orientation in inertial coordinates
enable_controller = 1    # turn controller on and off
m_Sat = 500 # kg
Isat = m_Sat*np.array([[.125, 0, 0], [0, .39583, 0], [0, 0, .39583]]) # satellite inertial matrix (kg^2/m^3)


oe0 = np.array([a, e, i, argp, RAAN, nu])


# nominal data #

nominal_data = ADCS.nominal_data(oe0, JDstart, n_orbits, t_int)
# [time_vec, rvmat, DCM_nom, w_nom]

n_times = len(nominal_data[:, 0])        # number of data points


thrust_arr = np.zeros([16, 1])           #
thrustCommand = np.zeros([4, 4])         #
momentsPlot = np.zeros([3, n_times])     #
gyroAngular = np.zeros([n_times, 3])     #
attitudePlot = np.zeros([n_times, 9])    #
RPYplot = np.zeros([3, n_times])         #
angularI = np.zeros([3, 1])              # intertial angular velocity
RNV = np.zeros([3, 3])                   #
rSatBu = np.zeros([3, n_times])          #
dt_thust = .01                           # thruster fire time

time = .8   # WTF is this??

thrustPlots = np.zeros([16, n_times])    #


Body_xyz = Body_xyz0


for i in range(n_times):

    # obtain orientation of sat at time, i
    RNV = np.array([nominal_data[1, 7:10], nominal_data[1, 10:13], nominal_data[1, 13:16]])

    # ignore perturbations for now
    perturbations = np.array([0, 0, 0])
    dt_thrust = .1

    dynamics = ADCS.sat_dynamics(perturbations, thrust_arr, angularI, Body_xyz, Isat, dt_thrust)

    Body_xyz = dynamics[0:3, 0:3]
    attitudePlot = Body_xyz
    angularI = dynamics[0:3, 3:6]
    momentsPlot = dynamics[0:3, -1]

    sensors = ADCS.sensors(angularI, Body_xyz, nominal_data[i, 1:4])

    





