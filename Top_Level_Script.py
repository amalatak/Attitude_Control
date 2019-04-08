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
e = .01       # -
RAAN = 0      # rad
argp = 0      # rad
nu = 0        # rad
JDstart = 0   # days
n_orbits = 1  # number of orbits
t_int = 10    # time interval

Body_xyz0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # starting orientation in inertial coordinates
ang1 = 2 # deg
ang2 = 3 # deg
Body_xyz0 = np.matmul(utils.R3(ang1*np.pi/180), np.matmul(utils.R2(ang2*np.pi/180), Body_xyz0))
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
dt_thrust = .1                           # thruster fire time



thrustPlots = np.zeros([16, n_times])    #
Body_xyz = Body_xyz0
time = .8          # WTF is this??

for i in range(n_times):

    # obtain 3x3 orientation (RNV) of sat at time, i
    RNV = np.array([nominal_data[i, 7:10], nominal_data[i, 10:13], nominal_data[i, 13:16]])

    # ignore perturbations for now
    perturbations = np.array([0, 0, 0])

    dynamics = ADCS.sat_dynamics(perturbations, thrust_arr, angularI, Body_xyz, Isat, dt_thrust)
    #print(dynamics)

    Body_xyz = dynamics[0:3, 0:3]
    attitudePlot = Body_xyz
    angularI = dynamics[0:3, 7]
    momentsPlot = dynamics[0:3, -1]

    # [e_Sun2Sat_b, e_Sat_b, gyro]
    sensors = ADCS.sensors(angularI, Body_xyz, nominal_data[i, 1:4])

    t_command_t_duration_RPY = ADCS.ADCS(RNV, sensors[:, 0], sensors[:, 1], sensors[:, 2],\
         nominal_data[i, 1:4], dt_thrust)
    RPYplot[:, i] = t_command_t_duration_RPY[-1]

    RPY = t_command_t_duration_RPY[-1]

    thrust_command = np.array([t_command_t_duration_RPY[0][:], t_command_t_duration_RPY[1][:], \
        t_command_t_duration_RPY[2][:], t_command_t_duration_RPY[3][:]])

    thrust_out = ADCS.rcs_thrust(thrust_command, time, dt_thrust)

    if enable_controller == 1 and i < n_times-12: # not sure yet why the latter statement has to be there
        thrustPlots[:, i:(i+len(thrust_out[1]))] = thrust_out[0].T

    elif enable_controller != 1:
        thrustPlots[:, i:(i+len(thrust_out[1]))] = 0



plt.figure()
plt.plot(nominal_data[:, 0], RPYplot[0, :])
plt.plot(nominal_data[:, 0], RPYplot[1, :])
plt.plot(nominal_data[:, 0], RPYplot[2, :])
# plt.show()
