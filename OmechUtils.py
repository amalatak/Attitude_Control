import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as mtick
from scipy.integrate import odeint
import Orbital_Constants as cn
import spiceypy as spice

# J200 is ECI #



# all angles in radians #

# Rotation Matrices #

def R1(theta):
	# 3x3 R1 (Around X, Roll) rotation matrix, input is in radians
	rotmat = np.matrix([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])
	return rotmat

def R2(theta):
	# 3x3 R2 (Around Y, Pitch) rotation matrix, input is in radians
	rotmat = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
	return rotmat

def R3(theta):
	# 3x3 R3 (Around Z, Yaw) rotation matrix, input is in radians
	rotmat = np.matrix([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
	return rotmat

# Time Systems #

def JD2GMST(JD):
	# Outputs GMST in hours
	D = JD - cn.JD0
	GMST = (18.697374558 + 24.06570982441908*D)%24
	return GMST

def JD2LST(JD, lon):
	# Input lon as degrees
	# LST output in hours
	GMST = JD2GMST(JD)
	LST = GMST + lon/15
	return LST

def JD2ET(JD):
	D = JD - cn.JD0
	ET = D*86400
	return ET

def theta_g_prop(theta_g0, we, deltat_vec):
	theta_g_vec = theta_g0 + we*deltat_vec
	return theta_g_vec


# Frame Conversions #

def cart2sph(cart_arr):
	if len(cart_arr) != 3:
		cart_arr = cart_arr[0:3]
	r = np.linalg.norm(cart_arr)                   # radius
	theta = np.arccos(cart_arr[2]/r)               # angle from X
	phi = np.arctan2(cart_arr[1], cart_arr[0])     # angle from Z

	sp_coord = np.array([theta, phi, r])
	return sp_coord

def ECI2ECEF(rv, theta_g):
	r_ecef = np.matmul(R3(-theta_g), np.transpose(rv[0:3]))
	return r_ecef

def ECEF2ECI(rv, theta_g):
	r_eci = np.matmul(R3(theta_g), rv[0:3])
	return r_eci

def pECEF2pSEZ(pECEF, lat, lon):
	pSEZ = np.matmul(R2(np.pi/2 - lat), np.matmul(R3(lon), pECEF))
	return pSEZ

def rec_pos(lat, lon):
	angle_arr = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
	return cn.ae*angle_arr

def get_sun_pos_rel2Earth(time):
	return spice.spkezr('SUN', time, 'J2000', "NONE", 'EARTH')

def get_sun_pos_rel2Earth_ECEF(time):
	return spice.spkezr('SUN', time, 'IAU_EARTH', "NONE", 'EARTH')


#sxformc

# basic calculations
def orbital_period(a, mu):
	T = (2*np.pi*np.sqrt((a**3)/mu))
	return T



# Orbital Element Conversions

def rv2oe(rv, mu):
	# [a, emag, i, w, Omega, nu]

	pos = rv[0:3] # ijk position
	vel = rv[3:6] # ijk velocity
	rnorm = np.sqrt(pos.dot(pos))
	vnorm = np.sqrt(vel.dot(vel))

	h = np.cross(pos, vel)
	z = np.array([0, 0, 1])

	n_not_unit = np.cross(z,h)
	n = n_not_unit/np.sqrt(n_not_unit.dot(n_not_unit))

	e = (np.cross(vel, h)/mu - pos/rnorm)
	emag = np.sqrt(e.dot(e))

	#Energy = .5*np.linalg.norm(vel)**2 - mu/np.linalg.norm(pos)

	if (emag != 1):
		a = -mu/(2*(vnorm**2/2 - mu/rnorm))
		#P = a*(1 - emag**2)
	
	else:
		a = 'Undefined'
		#P = h**2/mu

	Omega = np.arccos(n[0])

	if n[1] < 0:
		Omega = 2*np.pi - Omega

	i = np.arccos(z.dot(h)/np.sqrt(h.dot(h)))

	w = np.arccos(n.dot(e)/emag)
	if e[2] < 0:
		w = 2*np.pi - w

	nu = np.arccos(pos.dot(e)/(emag*rnorm))
	if pos.dot(vel) < 0:
		nu = 2*np.pi - nu


	oe = np.array([a, emag, i, w, Omega, nu])

	return oe


def rv2oe_M0(rv, mu):
	oe = rv2oe(rv, mu)

	E0 = vtoAnomaly(oe[1], oe[-1])
	M0 = E0 - oe[1]*np.sin(E0)

	oe[5] = M0
	return oe

def oe2oe_M0(oe):
	E0 = vtoAnomaly(oe[1], oe[-1])
	M0 = E0 - oe[1]*np.sin(E0)

	oe[5] = M0
	return oe



def oe2rv_M0(oe, mu):
	M = oe[5]
	e = oe[1]
	E = Esolver(M, e)
	v = Anomalytov(E, e)

	oe[5] = v

	rv = oe2rv(oe, mu)
	return rv


def vtoAnomaly(e, v):
	# e is eccentricity and v is true anomaly
	# must account for if e >= 1
	# v input as degrees

	if e < 1:
		sinE = np.sin(v)*np.sqrt(1 - e**2)/(1 + e*np.cos(v))
		cosE = (e + np.cos(v))/(1 + e*np.cos(v))
		E = np.arctan2(sinE, cosE)

	return E # radians

def Anomalytov(E, e):
	# for elliptical paths
	# E is in radians

	sinv = np.sin(E)*np.sqrt(1 - e**2)/(1 - e*np.cos(E))
	cosv = (np.cos(E) - e)/(1 - e*np.cos(E))
	v = np.arctan2(sinv, cosv)

	return v 

def Esolver(M, e):
	# mean anomaly must be within +- 2pi

	if (M > -np.pi and M < 0) or (M > np.pi):
		E = M - e
	else:
		E = M + e

	Eold = E
	flag = 1

	while abs(E - Eold) > 1e-12 or flag:

		Eold = E
		flag = 0
		E = Eold + (M - Eold + e*np.sin(Eold))/(1 - e*np.cos(Eold))

	return E



def oe2rv(oe, mu):

	# extract orbital elements in km and degrees #

	a = oe[0]
	e = oe[1]
	i = oe[2]
	w = oe[3]
	Omega = oe[4]
	nu = oe[5]

	# 2D elements in pq plane #

	rmag = a*(1 - e**2)/(1 + e*np.cos(nu))
	rpqw = np.array([rmag*np.cos(nu), rmag*np.sin(nu), 0])
	p = a*(1 - e**2)
	vpqw = np.array([-np.sin(nu)*np.sqrt(mu/p), np.sqrt(mu/p)*(e + np.cos(nu)), 0])

	# convert to IJK frame #

	Rqpw2ijk = np.matmul(R3(-Omega), np.matmul(R1(-i), R3(-w)))

	rijk = np.matmul(Rqpw2ijk, rpqw)
	vijk = np.matmul(Rqpw2ijk, vpqw)

	# output [rvector, vvector] #
	rv = np.array([0., 0., 0., 0., 0., 0.])
	rv[0:3] = rijk
	rv[3:6] = vijk

	return rv

# Orbit Propogators #

def orbitdiff(t, pv):
	# pv is the cartesian position and velocity vector
	# pv = [X, Y, Z, Xdot, Ydot, Zdot]
	mu = 3.986004415e14
	
	r = pv[0:3]
	v = pv[3:6]

	dr = v
	dv = -mu*r/(np.linalg.norm(r)**3)

	diff = np.zeros(6)
	diff[0:3] = dr
	diff[3:6] = dv

	return diff


def orbit_prop(rv, timevec):

	rvmat = np.zeros((len(timevec), len(rv)))
	rvmat[0,:] = rv

	r = integrate.ode(orbitdiff).set_integrator("dopri5")
	r.set_initial_value(rv, timevec[0])

	# sol = odeint(orbitdiff, rv, timevec)


	for i in range(1, timevec.size):
		rvmat[i, :] = r.integrate(timevec[i])


		if not r.successful:
			raise RuntimeError("Can't Integrate")

	return rvmat[:, :]


def orbit_prop_analytic(rv0, mu, t0, t):
	dt = 60
	time = 0
	oe = rv2oe(rv0, mu)
	a = oe[0]
	e = oe[1]

	while time < (t - dt):

		time += dt
		v = oe[5]
		E0 = vtoAnomaly(e, v)
		n = np.sqrt(mu/(a**3))
		M0 = E0 - e*np.sin(E0)

		if abs(time - t) < dt:
			M = M0 + n*abs(time - t)
		else:
			M = M0 + n*dt

		E = Esolver(M, e)
		vnew = Anomalytov(E, e)
		oe[5] = vnew
		pv = oe2rv(oe, mu)

		# pos = pv[0:3]
		# vel = pv[3:6]
		# h = np.cross(pos, vel)
		# Energy = .5*np.linalg.norm(vel)**2 - mu/np.linalg.norm(pos)

	return pv


# Perturbation Analysis #

def SPE_Approx(oe, ae, J2, mu):
	# Approximates the SPE (Secularly Precessing Ellipse) rates for a satellite
	# outputs rate of change of:
	# Cap Omega, Omega, M

	a = oe[0]
	e = oe[1]
	I = oe[2]

	nbar = np.sqrt(mu/a**3)
	
	CapOmegaDot = -(3/2)*nbar*((ae/a)**2)*J2*np.cos(I)/np.sqrt(1 - e**2)
	wdot = -(3/4)*nbar*((ae/a)**2)*J2*(1 - 5*np.cos(I)**2)/((1 - e**2)**2)
	Mdot = nbar*(1 - (3/4)*((ae/a)**2)*J2*(1 - 3*np.cos(I)**2)/(1 - e**2)**(3/2))

	return np.array([wdot, CapOmegaDot, Mdot])

def SPE_Prop(oe_epoch, SPE_rates, time_vec):
	# oe_epoch must be in the following form [a, e, i, w, W, M]
	n = len(time_vec)
	oe_vec = np.empty([n, 6])
	for i in range(n):
		oe_vec[i, 0:3] = oe_epoch[0:3]
		oe_vec[i, 3:5] = oe_epoch[3:5] + SPE_rates[0:2]*time_vec[i]
		oe_vec[i, 5] = (oe_epoch[5] + SPE_rates[-1]*time_vec[i])%(2*np.pi)
	return oe_vec


# density function #
def density(alt):
	# input is in km, density is output in kg/m^3
	if alt < 25:
		h0 = 0; p0 = 1.225; H = 7.249
	elif alt < 30:
		h0 = 25; p0 = 3.899e-2; H = 6.349
	elif alt < 40:
		h0 = 30; p0 = 1.774e-2; H = 6.682
	elif alt < 50:
		h0 = 40; p0 = 3.972e-3; H = 7.554
	elif alt < 60:
		h0 = 50; p0 = 1.057e-3; H = 8.382
	elif alt < 70:
		h0 = 60; p0 = 3.206e-4; H = 7.714
	elif alt < 80:
		h0 = 70; p0 = 8.770e-5; H = 6.549
	elif alt < 90:
		h0 = 80; p0 = 1.905e-5; H = 5.799
	elif alt < 100:
		h0 = 90; p0 = 3.396e-6; H = 5.382
	elif alt < 110:
		h0 = 100; p0 = 5.297e-7; H = 5.877
	elif alt < 120:
		h0 = 110; p0 = 9.661e-8; H = 7.263
	elif alt < 130:
		h0 = 120; p0 = 2.438e-8; H = 9.473
	elif alt < 140:
		h0 = 130; p0 = 8.484e-9; H = 12.636
	elif alt < 150:
		h0 = 140; p0 = 3.845e-9; H = 16.149
	elif alt < 180:
		h0 = 150; p0 = 2.070e-9; H = 22.523
	elif alt < 200:
		h0 = 180; p0 = 5.464e-10; H = 29.740
	elif alt < 250:
		h0 = 200; p0 = 2.789e-10; H = 37.105
	elif alt < 300:
		h0 = 250; p0 = 7.248e-11; H = 45.546
	elif alt < 350:
		h0 = 300; p0 = 2.418e-11; H = 53.628
	elif alt < 400:
		h0 = 350; p0 = 9.518e-12; H = 53.298
	elif alt < 450:
		h0 = 400; p0 = 3.725e-12; H = 58.515
	elif alt < 500:
		h0 = 450; p0 = 1.585e-12; H = 60.828
	elif alt < 600:
		h0 = 500; p0 = 6.967e-13; H = 63.822
	elif alt < 700:
		h0 = 600; p0 = 1.454e-13; H = 71.835
	elif alt < 800:
		h0 = 700; p0 = 3.614e-14; H = 88.667
	elif alt < 900:
		h0 = 800; p0 = 1.170e-14; H = 124.64
	elif alt < 1000:
		h0 = 900; p0 = 5.245e-15; H = 181.05
	elif alt > 1000:
		h0 = 1000; p0 = 3.019e-15; H = 268.00
	else:
		print('Wut')

	density = p0*np.exp(-(alt - h0)/H)
	return density

def VelRel2Wind(rv, wevec, wind):
	vr = rv[3:6] - np.cross(wevec, rv[0:3]) - wind
	return vr

def drag(Cd, A, m, vr, alt):

	fd = Cd*A*.5*density(alt)*np.sqrt(vr.dot(vr))**2/m
	return fd

def dadt(vr, fd, a, mu):
	da_dt = -2*vr*a**2*fd/mu
	return da_dt


def cross_equator(rv, rv_last):
	# crosses the equator when Z changes signs
	if np.sign(rv[2]) != np.sign(rv_last[2]):
		return 1
	else:
		return 0

# Orbit Plot

def plot_orbit3d(rvmat):

	""" Takes an input of a matrix of cartesian orbital coordinates where the 
	first three columns are the XYZ position data
	"""

	orbit = rvmat[:, 0:3]
	plt.figure(10)
	ax = plt.axes(projection = '3d')
	plt.axis('equal')
	ax.plot3D(orbit[:, 0], orbit[:, 1], orbit[:, 2])

	"""
	max_range = np.array(
	[sol[:, 1].max() - sol[:, 1].min(), sol[:, 2].max() - sol[:, 2].min(), sol[:, 3].max() - sol[:, 3].min()]).max()
	Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (sol[:, 1].max() + sol[:, 1].min())
	Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (sol[:, 2].max() + sol[:, 2].min())
	Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (sol[:, 3].max() + sol[:, 3].min())
	# Comment or uncomment following both lines to test the fake bounding box:
	for xb, yb, zb in zip(Xb, Yb, Zb):
	ax.plot([xb], [yb], [zb], 'w')
	"""

	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_zlabel('Z (m)')
	plt.title('Orbit Propogation')
	ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
	ax.view_init(30, 20)
	plt.show()
	