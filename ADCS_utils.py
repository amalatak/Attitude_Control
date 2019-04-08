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
    # rv_mat must be a vertical Nx6 position-velocity matrix
    # output a radial-normal-vel frame for all rows
    n_samples = len(rv_mat[:, 0])
    RNV = np.zeros([n_samples, 9])
    for i in range(n_samples):
        R = -1*(rv_mat[i, 0:3]/np.linalg.norm(rv_mat[i, 0:3])) # radial direction
        N = np.cross(rv_mat[i, 0:3], rv_mat[i, 3:6])/np.linalg.norm(np.cross(rv_mat[i, 0:3], rv_mat[i, 3:6])) # out of plane
        V = np.cross(R, N)/np.linalg.norm(np.cross(R, N)) # Complete RH system
        RNV[i, 0:9] = np.concatenate((R, N, V))
    return RNV 


def desired_angular_vel(rv_mat):
    # rv_mat must be a vertical Nx6 position-velocity matrix
    # output a nominal inertial angular velocity 
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
    #
    #

    # constants and initialization
    period = utils.orbital_period(oe[0], cn.mu_e)         # orbital period
    rv0 = utils.oe2rv(oe, cn.mu_e)                        # initial position-velocity
    Start_Time = JDstart*86400                            # convert JDstart to s
    End_Time = Start_Time + orbits*period                 # get end time
    n_times = int((End_Time - Start_Time)/interval)       # N number of elements for arrays
    time_vec = np.linspace(Start_Time, End_Time, n_times) # get 1xN time vector
    rvmat = utils.orbit_prop(rv0, time_vec)               # Nx6 position-velocity matrix
    DCM_nom = desired_attitude(rvmat)                     # Nx9 RNV matrix
    w_nom = desired_angular_vel(rvmat)                    # Nx3 angular velocity matrix

    # transpose time vector to allow concatenation
    time_vec = np.array([time_vec])
    time_vec = time_vec.T

    # format output
    data = np.concatenate((time_vec, rvmat, DCM_nom, w_nom), axis=1)
    return data


def sat_dynamics(perturbations, thrust_arr, angular_inertial, Bxyz, I_sat, dt_thrust):
    # Needs to be checked for errors
    #
    # inputs:
    #
    #  -  1x3  perturbation array
    #  -  16x1 thrust array
    #  -  3x1 inertial angular velocity array
    #  -  3x3 body XYZ orientation array
    #  -  3x3 satellite inertial matrix
    #  -  scalar thrust duration time
    #
    # outputs 3x13 matrix of:
    #
    #  -  3x3 new attitude in body frame
    #  -  3x3 new attitude in inertial frame
    #  -  3x1 new angular velocity in body frame
    #  -  3x1 new angular velocity in inertial frame
    #  -  3x3 transformation matrix, inertial to body
    #  -  3x1 euler angle vector
    #  -  3x1 moment vectors


    # initialize variables
    DCM0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])     # body attitude  ------------------- ?
    moment_total = np.zeros([3, 1])                        # total moment around body axis
    new_attitude_body = np.zeros([3, 3])                   # new attitude, body
    new_attitude_intertial = np.zeros([3, 3])              # new attitude, inertial
    new_angular_body = np.zeros([3, 1])                    # new angular vel, body
    new_angular_inertial = np.zeros([3, 1])                # new angular vel, inertial
    d_angle = np.zeros([3, 1])                             # change in angle over dt, specifically, dt_thrust
    angular_body = np.matmul(np.matrix.transpose(DCM0),\
         angular_inertial)                                 # body angular velocity ------------ ?


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
    new_attitude_intertial = np.matmul(new_attitude_body, Bxyz)

    # Tbinew = np.matmul(transform, Bxyz)

    true_TiB = Bxyz
    new_angular_inertial = np.matmul(Bxyz, new_angular_body)


    data = np.concatenate([new_attitude_body, new_attitude_intertial, new_angular_body,\
         new_angular_inertial, true_TiB, d_angle, moment_total], axis=1)

    # 3x15 or 3x [new_attitude_body(3), new_attitude_inertial(3), new_angular_body(1),
    #  new_anguler_inertial(1), true_TiB(3), d_angle(1), moment(1)]

    return data


def sensors(angular_vel_inertial, Bxyz, Sat_pos):
    # inputs:
    #  -  3x1 angular velocity in ECI
    #  -  3x3 satellite attitude in ECI
    #  -  1x3 Satellite position in ECI
    #
    # outputs 3x3 array of:
    #  -  3x1 Sun to satellite direction vector in body coordinates
    #  -  3x1 Satellite direction vector in body coordinates
    #  -  3x1 Angular velocity as measured by the gyro
    
    au = cn.au                                # scalar distance to the sun, in m
    bias = np.zeros([3, 1])                   # 3x1 gyro bias vec
    ARW = np.zeros([3, 1])                    # 3x1 angular random walk vec
    sun_dir = np.array([[0], [1], [0]])       # 3x1 unit vector to sun from ECI
    gyro = angular_vel_inertial + bias + ARW  # 3x1 gyro measurement of angular velocity

    # vector directions
    # 3x1 sun vectors
    r_Sun_i = au*sun_dir
    # e_Sun_i = r_Sun_i/np.linalg.norm(r_Sun_i)     # 3x1 sun vector ECI

    r_Sat_i = np.matrix.transpose(Sat_pos)
    r_Sat_i = np.array([Sat_pos])
    r_Sat_i = r_Sat_i.T
    # e_Sat_i = r_Sat_i/np.linalg.norm(r_Sat_i)     # 3x1 sat vector ECI

    rSun2Sat = r_Sun_i - r_Sat_i
    # e_Sun2Sat_i = rSun2Sat/np.linalg.norm(rSun2Sat) # 3x1 vector from sun to sat

    # Calculate Sat to Sun vector direction and Sat to Earth direction vector to  
    # send to ADCS, and add simulated noise

    r_Sun2Sat_b = np.matmul(Bxyz, rSun2Sat) + .0001*np.array([[random.randint(0, 100)], \
        [random.randint(0, 100)], [random.randint(0, 100)]])

    r_Sat_b = np.matmul(-Bxyz, r_Sat_i) + .0001*np.array([[random.randint(0, 100)], \
        [random.randint(0, 100)], [random.randint(0, 100)]])

    e_Sun2Sat_b = r_Sun2Sat_b/np.linalg.norm(r_Sun2Sat_b)

    e_Sat_b = r_Sat_b/np.linalg.norm(r_Sat_b)

    meaurements = np.concatenate([e_Sun2Sat_b, e_Sat_b, gyro], axis=1)

    return meaurements


def ADCS(RNV, e_Sun2Sat_b, e_Sat_b, observedAngular, r_Sat_i, dt_thrust):

    # This function reads vector information from sensors to determine the
    # attitude and angular velocities of the satellite. Data is also read from
    # the orbitData function results to determine the desired attitude at any
    # given time. Once the observed attitude is calculated, the thruster
    # commands are calculated and then sent to the thruster block

    # inputs:
    #  -  3x3 Radial-Normal-Vel orientation matrix
    #  -  3x1 Sun to Satellite unit vector in the body frame
    #  -  3x1 Satellite position unit vector in the body frame
    #  -  3x1 Angular Velocity vector 
    #  -  3x1 Satellite position vector, inertial frame
    #  -  scalar value for thruster duration
    #
    # outputs 1x20 array of:
    #  -  4x4 binary thruster command
    #  -  scalar thrust duration
    #  -  1x3 Roll-Pitch-Yaw angles in degrees
    # 
    # note that it outputs a list with arrays within the list

    # define maximum and minimum errors

    thrustCommand = np.zeros([4, 4])
    b_orient = np.zeros([3, 3])
    i_orient = np.zeros([3, 3])


    negativeAngle = -2   # deg
    positiveAngle = 2    # deg
    negativeRate = -.015 # deg/s
    positiveRate = .015  # deg/s

    au = cn.au
    r_Sun_i = au*np.array([0, 1, 0])

    r_Sat_i = -np.matrix.transpose(r_Sat_i)

    r_Sun2Sat_i = r_Sun_i - r_Sat_i

    thrust_duration = dt_thrust


    # ++++++++++++++++++ TRIAD +++++++++++++++++++++ #

    r_Sun_r = np.matmul(RNV, r_Sun_i)
    e_Sun_r = r_Sun_r/np.linalg.norm(r_Sun_r)
    r_Sun2Sat_r = np.matmul(RNV, r_Sun2Sat_i)
    e_Sun2Sat_r = r_Sun2Sat_r/np.linalg.norm(r_Sun2Sat_r)
    r_Sat_r = np.matmul(RNV, r_Sat_i)
    e_Sat_r = r_Sat_r/np.linalg.norm(r_Sat_r)

    # algorithm

    x_b = e_Sat_b[:, 0]
    z_b = np.cross(x_b, e_Sun2Sat_b, axisa=0, axisb=0, axisc=0)/np.linalg.norm(np.cross(x_b, e_Sun2Sat_b, axisa=0, axisb=0,axisc=0))
    y_b = np.cross(z_b, x_b, axisa = 0, axisb =0)


    x_r = e_Sat_r
    z_r = np.cross(x_r, e_Sun2Sat_r)/np.linalg.norm(np.cross(x_r, e_Sun2Sat_r))
    y_r = np.cross(z_r, x_r)

    # print(x_b.shape)

    b_orient[:, 0] = np.array([x_b[0, 0], x_b[1, 0], x_b[2, 0]])
    b_orient[:, 1] = y_b
    b_orient[:, 2] = np.array([z_b[0, 0], z_b[1, 0], z_b[2, 0]])

    i_orient[:, 0] = x_r
    i_orient[:, 1] = y_r
    i_orient[:, 2] = z_r

    T1 = np.matmul(b_orient, np.matrix.transpose(i_orient))
    T2 = np.matrix.transpose(T1)
    T2_copy = T2.copy()     # make non-strided somehow

    RPY = spice.m2eul(T2_copy, 1, 2, 3)
    RPY = np.array(RPY)     # Roll, Pitch, Yaw tuple to numpy array
    RPY = RPY*180/np.pi




# =============== Error Correction ================= #


    if RPY[0] < negativeAngle:           # Roll error correction
        thrustCommand[0, 1] = 1
        thrustCommand[2, 1] = 1
    elif RPY[0] > positiveAngle:
        thrustCommand[0, 3] = 1
        thrustCommand[2, 3] = 1

    if RPY[1] < negativeAngle:           # Pitch error correction
        thrustCommand[1, 2] = 1
        thrustCommand[3, 0] = 1
    elif RPY[1] > positiveAngle:
        thrustCommand[1, 0] = 1
        thrustCommand[3, 2] = 1

    if RPY[2] < negativeAngle:           # Yaw error correction
        thrustCommand[0, 0] = 1
        thrustCommand[2, 2] = 1
    elif RPY[2] > positiveAngle:
        thrustCommand[0, 2] = 1
        thrustCommand[2, 0] = 1



# ================= RATE correction ================= #

    if observedAngular[0, 0] > positiveRate:          # Roll rate correction
        thrustCommand[0, 3] = 1
        thrustCommand[2, 3] = 1

    elif observedAngular[0, 0] < negativeRate:
        thrustCommand[0, 1] = 1
        thrustCommand[2, 1] = 1


    if observedAngular[1, 0] > positiveRate:          # Pitch rate correction
        thrustCommand[1, 0] = 1
        thrustCommand[3, 2] = 1

    elif observedAngular[1, 0] < negativeRate:
        thrustCommand[1, 2] = 1
        thrustCommand[3, 0] = 1

    if observedAngular[2, 0] > positiveRate:          # Yaw rate correction
        thrustCommand[0, 2] = 1
        thrustCommand[2, 0] = 1

    elif observedAngular[2, 0] < negativeRate:
        thrustCommand[0, 0] = 1
        thrustCommand[2, 2] = 1

    t_command_t_duration_RPY = [thrustCommand[0, :], thrustCommand[1, :], \
        thrustCommand[2, :], thrustCommand[3, :], thrust_duration, RPY]

    return t_command_t_duration_RPY




def rcs_thrust(thrust_command, time, dt_thrust):
    # this function calculates all of the outputs of the
    # simulated RCS thrusters
    #
    # Inputs:
    #  -  4x4 binary thrust array
    #  -  scalar of time
    #
    # Outputs:

    thrust_shape = thrust_command.shape
    duration = np.linspace(0, time + .5, (time + .5)/dt_thrust)
    thrust_out_all = np.zeros([len(duration), thrust_shape[0]*thrust_shape[1]])
    thrust_out_shape = thrust_out_all.shape

    for i in range(thrust_out_shape[1]):
        thrust_out = PID(time, duration, dt_thrust)
        for j in range(thrust_out_shape[0]):
            thrust_out_all[j, i] = thrust_out[0, j]

    for k in range(thrust_shape[0]):
        for p in range(thrust_shape[1]):
            if thrust_command[k, p] == 0:
                thrust_out_all[(4*(k-1) + p), p] = 0

    return [thrust_out_all, duration]
    
def PID(time, thrust_time_arr, dt_thrust):

    # initializations
    max_thrust = 5
    command_thrust = 0
    thrust_out = np.zeros([1, len(thrust_time_arr)])
    pLast = 0
    dt = 0
    summ = 0
    Kp = 1.1
    Ki = .01
    Kd = .005

    for i in range(len(thrust_time_arr)-1):
        if thrust_time_arr[i] <= time:
            command_thrust = max_thrust
        elif thrust_time_arr[i] > time:
            command_thrust = 0
        
        p = command_thrust - thrust_out[0, i]
        dt = (p - pLast)/dt_thrust
        summ += p
        pid_term = Kp*p - Kd*dt + Ki*summ
        thrust_out[0, i+1] = thrust_out[0, i] + pid_term
        pLast = p

    thrust_out[0, len(thrust_time_arr)-1] = 0
    thrust_out[0, 0] = 0
    return thrust_out

    

