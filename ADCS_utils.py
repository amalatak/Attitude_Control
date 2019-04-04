import sys
sys.path.insert(0, '/home/amalatak/Documents/Programming/Python/Orbital_Mechanics/')
sys.path.insert(0, '/home/amalatak/Documents/University/Sp19/Orbital_Mechanics/Homework/')
import OmechUtils as utils 
import numpy as np 
import Orbital_Constants as cn 
import matplotlib.pyplot as plt
import spiceypy as spice
import random


def desired_attitude(rv_mat):
    n_samples = len(rv_mat[:, 0])
    RNV = np.zeros([n_samples, 9])
    for i in range(n_samples):
        R = -1*(rv_mat[i, 0:3]/np.linalg.norm(rv_mat[i, 0:3])) # radial direction
        N = np.cross(rv_mat[i, 0:3], rv_mat[i, 3:6])/np.linalg.norm(np.cross(rv_mat[i, 0:3], rv_mat[i, 3:6])) # out of plane
        V = np.cross(R, N)/np.linalg.norm(np.cross(R, N)) # Complete RH system
        RNV[i, 0:9] = np.concatenate((R, N, V))

    return RNV

def desired_angular_vel(rv_mat):
    n_samples = len(rv_mat[:, 0])
    w_nom = np.zeros([n_samples, 3])
    for i in range(n_samples):
        w_nom[i, 0:3] = np.cross(rv_mat[i, 0:3], rv_mat[i, 4:6])/(np.linalg.norm(rv_mat[i, 0:3])**2)
    return w_nom


def nominal_data(oe, JDstart, orbits, interval):
    # takes input of starting Julian Date, orbital elements (a, e, i, argp, RAAN, nu),
    # number of orbits and time interval
    # 
    # outputs an Nx19 matrix with [time, position, velocity, nominal attitude, nominal angular velocity]


    period = utils.orbital_period(oe[0], cn.mu_e)

    rv0 = utils.oe2rv(oe, cn.mu_e)

    Start_Time = JDstart*86400 # convert JD to s
    End_Time = Start_Time + orbits*period

    n_times = int((End_Time - Start_Time)/interval)

    time_vec = np.linspace(Start_Time, End_Time, n_times)
    time_vec = np.reshape(time_vec, [len(time_vec), 1])

    rvmat = utils.orbit_prop(rv0, time_vec) # Nx6 position matrix
    DCM_nom = desired_attitude(rvmat)
    w_nom = desired_angular_vel(rvmat)


    data = np.concatenate((time_vec, rvmat, DCM_nom, w_nom), axis=1)
    return data


def sat_dynamics(perturbations, thrust_arr, angular_inertial, Bxyz, I_sat, dt_thrust):
    #print(Bxyz)

    # initialize variables

    DCM0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    moment_total = np.zeros([3, 1])
    new_attitude_body = np.zeros([3, 3])
    new_attitude_intertial = np.zeros([3, 3])
    new_angular_body = np.zeros([3, 1])
    new_angular_inertial = np.zeros([3, 1])
    d_angle = np.zeros([3, 1])

    angular_body = np.matmul(np.matrix.transpose(DCM0), angular_inertial)

    # sum thrust moments around body axes
    roll_moment = .5*(thrust_arr[1] + thrust_arr[9] - thrust_arr[4] - thrust_arr[11])     # body x
    pitch_moment = .5*(thrust_arr[6] + thrust_arr[12] - thrust_arr[4] - thrust_arr[14])   # body y
    yaw_moment = .5*(thrust_arr[0] + thrust_arr[10] - thrust_arr[2] - thrust_arr[9])      # body z

    # add perturbing moments
    moment_total[0, 0] = roll_moment + perturbations[0]
    moment_total[1, 0] = pitch_moment + perturbations[1]
    moment_total[2, 0] = yaw_moment + perturbations[2]

    # calculate new angular velocity

        
    new_angular_body = angular_body + np.matmul(np.linalg.inv(I_sat), moment_total)*dt_thrust
    d_angle = angular_body*dt_thrust + .5*np.matmul(np.linalg.inv(I_sat), moment_total)*(dt_thrust**2)

    R3 = utils.R3(-d_angle[2, 0])
    R2 = utils.R2(-d_angle[1, 0])
    R1 = utils.R1(-d_angle[0, 0])


    transform = np.matmul(np.matmul(R3, R2), R1)
    new_attitude_body = np.matmul(np.linalg.inv(transform), np.linalg.inv(DCM0))
    # Tbinew = np.matmul(transform, Bxyz)


    new_attitude_intertial = np.matmul(new_attitude_body, Bxyz)
    true_TiB = Bxyz
    new_angular_inertial = np.matmul(Bxyz, new_angular_body)


    data = np.concatenate([new_attitude_body, new_attitude_intertial, new_angular_body,\
         new_angular_inertial, true_TiB, d_angle, moment_total], axis=1)

    # 3x15 or 3x [new_attitude_body(3), new_attitude_inertial(3), new_angular_body(1),
    #  new_anguler_inertial(1), true_TiB(3), d_angle(3), moment(1)]


    return data

def sensors(angular_vel_inertial, Bxyz, Sat_pos):
    
    au = np.au
    bias = np.zeros([3, 1])
    ARW = np.zeros([3, 1])

    gyro = angular_vel_inertial + bias + ARW # gyro measurement


    # vector directions

    r_Sun_i = au*np.array([0, 1, 0])
    e_Sun_i = r_Sun_i/np.linalg.norm(r_Sun_i)

    r_Sat_i = np.matrix.transpose(Sat_pos)
    e_Sat_i = r_Sat_i/np.linalg.norm(r_Sat_i)

    rSun2Sat = r_Sun_i - r_Sat_i
    e_Sun2Sat = rSun2Sat/np.linalg.norm(rSun2Sat)

    # Calculate Sat to Sun vector direction and Sat to Earth direction to send to 
    # ADCS, plus a little noise

    r_Sun2Sat_b = np.matmul(Bxyz, rSun2Sat) + .0001*np.array([[random.randint(0, 100)], \
        [random.randint(0, 100)], [random.randint(0, 100)]])


    r_Sat_b = np.matmul(-Bxyz, r_Sat_i) + .0001*np.array([[random.randint(0, 100)], \
        [random.randint(0, 100)], [random.randint(0, 100)]])

    e_Sun2Sat_b = r_Sun2Sat_b/np.linalg.norm(r_Sun2Sat_b)

    e_Sat_b = r_Sat_b/np.linalg.norm(r_Sat_b)

    meaurements = np.concatenate([e_Sun2Sat_b, e_Sat_b, gyro], axis=1)

    return meaurements


def ADCS(RNV, e_Sun2Sat_b, e_Sat_b, angular_observed, r_Sat_i, dt_thrust):

    # This function reads vector information from sensors to determine the
    # attitude and angular velocities of the satellite. Data is also read from
    # the orbitData function results to determine the desired attitude at any
    # given time. Once the observed attitude is calculated, the thruster
    # commands are calculated and then sent to the thruster block

    # define maximum and minimum errors

    thrust_command = np.zeros([4, 4])

    negative_angle = -2   # deg
    positive_angle = 2    # deg
    negative_rate = -.015 # deg/s
    positive_rate = .015  # deg/s

    au = np.au
    r_Sun_i = au*np.array([0, 1, 0])

    r_Sat_i = -np.matrix.transpose(r_Sat_i)

    r_Sun2Sat_i = r_Sun_i - r_Sat_i

    # ++++++++++++++++++ TRIAD +++++++++++++++++++++ #



    return 0