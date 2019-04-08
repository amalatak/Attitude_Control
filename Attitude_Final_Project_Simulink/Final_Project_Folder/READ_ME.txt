#======================================#


Final Project ReadME


The project submitted was written using Simulink 


There were no special libraries used in the making of this program with the notable exception of one custom MATLAB program, qmult, described at the bottom of the readme in case the file cannot be found. Everything else should run properly, assuming one has the ability to run MATLAB 2018a and the corresponding Simulink version. 


There are two switches in the Simulink model, one determines if measurement errors will be used and the other switches on/off the controller. 


All of the satellite constants and inertial matrices are found on multiple occasions throughout the program in constant blocks. All initial conditions are specified in each integrator or feeding into the integrator blocks




For any questions, email andrewmalatak@gmail.com












function q3 = qmult(q2, q1)
% performs quaternion multiplication on two length
% four vectors, with the fourth element being
% the scalar element
% outputs q3 as a vertical vector
% inputs should be vertical vectors


q3s = q1(4)*q2(4) - dot(q2(1:3), q1(1:3));
q3v = q1(4)*q2(1:3) + q2(4)*q1(1:3) - cross(q2(1:3), q1(1:3));
q3 = [q3v(1); q3v(2); q3v(3); q3s];