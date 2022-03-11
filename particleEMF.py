# /home/nekton/miniconda3/Python_course/Particle_EMF
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class ElectromagneticField:
	def __init__(self, type, ison, k, tau, omega):
		self.k = k
		self.ison = ison
		self.tau = tau
		self.omega = omega

		if type == 'crossed':
			self.E = self.E_crossed
			self.H = self.H_crossed
		if type == 'linear':
			self.E = self.E_linear
			self.H = self.H_linear
		if type == 'circular':
			self.E = self.E_circular
			self.H = self.H_circular

	def damping(self, t):
		if (self.ison == True):
			return np.exp(-0.5 * t**2 / self.tau**2)
		else:
			return 1

	def E_crossed(self, t, y):
		E = 0.05
		return np.array([1, 0, 0]) * E * self.damping(t)

	def H_crossed(self, t, y):
		H = 1
		return np.array([0, 1, 0]) * H * self.damping(t)

	def E_linear(self, t, y):
		E = 1
		Ex = np.cos(self.omega*t - np.dot(self.k, y))
		return np.array([Ex, 0, 0]) * E * self.damping(t)

	def H_linear(self, t, y):
		H = 0.05
		Hy = np.cos(self.omega*t - np.dot(self.k, y))
		return np.array([0, Hy, 0]) * H * self.damping(t)

	def E_circular(self, t, y):
		E = 0.1
		Ex = np.cos(self.omega*t - np.dot(self.k, y))
		Ey = np.sin(self.omega*t - np.dot(self.k, y))
		return np.array([Ex, Ey, 0]) * E * self.damping(t)

	def H_circular(self, t, y):
		H = 1
		Hx = - np.sin(self.omega*t - np.dot(self.k, y))
		Hy = np.cos(self.omega*t - np.dot(self.k, y))
		return np.array([Hx, Hy, 0]) * H * self.damping(t)

# norm equal to vector longitude
	def E_value(self, t, x, y, z):
		return np.linalg.norm(self.E(t, (x, y, z)))

	def H_value(self, t, x, y, z):
		return np.linalg.norm(self.H(t, (x, y, z)))

	def I_value(self, t, x, y, z):
		return (self.E_value(t, x, y, z)**2 + self.H_value(t, x, y, z)**2) / (4 * np.pi)

	def Field_plot(self, field, axis1, axis2):
		x = np.linspace(-5, 5, 100)
		y = np.linspace(-5, 5, 100)
		if field == 'E':
			F = self.E
		if field == 'H':
			F = self.H
		Fx = np.empty((len(x), len(y)))
		Fy = np.empty((len(x), len(y)))
		Fz = np.empty((len(x), len(y)))

		if axis1 == 't':
			if axis2 == 'x':
				for i in range(len(x)):
					for j in range(len(y)):
						Fx[i][j] = F(x[i], (y[j], 0, 0))[0]
						Fy[i][j] = F(x[i], (y[j], 0, 0))[1]		# F(t, x)
						Fz[i][j] = F(x[i], (y[j], 0, 0))[2]
			elif axis2 == 'y':
				for i in range(len(x)):
					for j in range(len(y)):
						Fx[i][j] = F(x[i], (0, y[j], 0))[0]
						Fy[i][j] = F(x[i], (0, y[j], 0))[1]		# F(t, y)
						Fz[i][j] = F(x[i], (0, y[j], 0))[2]
			else:
				for i in range(len(x)):
					for j in range(len(y)):
						Fx[i][j] = F(x[i], (0, 0, y[j]))[0]
						Fy[i][j] = F(x[i], (0, 0, y[j]))[1]		# F(t, z)
						Fz[i][j] = F(x[i], (0, 0, y[j]))[2]
		elif axis1 == 'x':
			if axis2 == 'y':
				for i in range(len(x)):
					for j in range(len(y)):
						Fx[i][j] = F(0, (x[i], y[j], 0))[0]
						Fy[i][j] = F(0, (x[i], y[j], 0))[1]		# F(x, y)
						Fz[i][j] = F(0, (x[i], y[j], 0))[2]
			elif axis2 == 'z':
				for i in range(len(x)):
					for j in range(len(y)):
						Fx[i][j] = F(0, (x[i], 0, y[j]))[0]
						Fy[i][j] = F(0, (x[i], 0, y[j]))[1]		# F(x, z)
						Fz[i][j] = F(0, (x[i], 0, y[j]))[2]
		elif axis1 == 'y':
			for i in range(len(x)):
				for j in range(len(y)):
					Fx[i][j] = F(0, (0, x[i], y[j]))[0]
					Fy[i][j] = F(0, (0, x[i], y[j]))[1]			# F(y, z)
					Fz[i][j] = F(0, (0, x[i], y[j]))[2]

		#131 = 1 string 3 collumn 1st 'cell' for figure
		# $ for math-mode in f-string
		# Fx
		plt.subplot(131)
		plt.pcolormesh(x, y, Fx, cmap = "plasma", shading = 'auto')
		plt.xlabel(f'$2\\pi {axis2}/\\lambda_0$')
		plt.ylabel(f'$2\\pi {axis1}/\\lambda_0$')
		plt.title(f'${field}_x({axis1},{axis2}$) plane')
		# Fy
		plt.subplot(132)
		plt.pcolormesh(x, y, Fy, cmap = "plasma", shading = 'auto')
		plt.xlabel(f'$2\\pi {axis2}/\\lambda_0$')
		plt.ylabel(f'$2\\pi {axis1}/\\lambda_0$')
		plt.title(f'${field}_y({axis1},{axis2}$) plane')
		# Fz
		plt.subplot(133)
		plt.pcolormesh(x, y, Fz, cmap = "plasma", shading = 'auto')
		plt.xlabel(f'$2\\pi {axis2}/\\lambda_0$')
		plt.ylabel(f'$2\\pi {axis1}/\\lambda_0$')
		plt.title(f'${field}_z({axis1},{axis2}$) plane')

		plt.colorbar()
		plt.tight_layout()
		plt.savefig(f'{field}({axis1},{axis2}).jpg')

	def I_plot(self, axis1, axis2):
		x = np.linspace(-5, 5, 100)
		y = np.linspace(-5, 5, 100)
		I = np.empty((len(x), len(y)))
		H = np.empty((len(x), len(y)))
		if axis1 == 't':
			if axis2 == 'x':
				for i in range(len(x)):
					for j in range(len(y)):
						I[i][j] = self.I_value(x[i], y[j], 0, 0)			# I(t, x)
			elif axis2 == 'y':
				for i in range(len(x)):
					for j in range(len(y)):
						I[i][j] = self.I_value(x[i], 0, y[j], 0)			# I(t, y)
			else:
				for i in range(len(x)):
					for j in range(len(y)):
						I[i][j] = self.I_value(x[i], 0, 0, y[j])			# I(t, z)
		elif axis1 == 'x':
			if axis2 == 'y':
				for i in range(len(x)):
					for j in range(len(y)):
						I[i][j] = self.I_value(0, x[i], y[j], 0)			# I(x, y)
			elif axis2 == 'z':
				for i in range(len(x)):
					for j in range(len(y)):
						I[i][j] = self.I_value(0, x[i], 0, y[j])			# I(x, z)
		elif axis1 == 'y':
			for i in range(len(x)):
				for j in range(len(y)):
					I[i][j] = self.I_value(0, 0, x[i], y[j])				# I(y, z)

		# Intensity
		plt.pcolormesh(x, y, I, cmap = "plasma", shading = 'auto')
		plt.xlabel(f'$2\\pi {axis2}/\\lambda_0$')
		plt.ylabel(f'$2\\pi {axis1}/\\lambda_0$')
		plt.title(f'$Intensity({axis1},{axis2}$) plane')

		plt.colorbar()
		plt.savefig(f'I({axis1},{axis2})')

class Particle:
	def __init__(self, p, x, q, m, tf, tmin):
		self.p = p
		self.x = x
		self.q = q
		self.m = m
		self.tf = tf
		self.tmin = tmin
		self.initial_conditions = np.concatenate((self.p, self.x))

	def gamma(self, p):
		return 1 / (self.m * np.sqrt(1 + np.dot(p, p)))

	def Solve(self, field):
		k = field.k
		omega = field.omega
		ison = field.ison
		tau = field.tau

		def func(t, y):
			p = y[:3]
			r = y[3:]
			return np.concatenate((
				self.q * (field.E(t, r) + self.gamma(p) * np.cross(p, field.H(t, r))),
				self.gamma(p) * p))

		self.solution = solve_ivp(
			func,
			[self.tmin, self.tf],
			y0 = self.initial_conditions,
			t_eval = np.arange(self.tmin, self.tf, 0.1),
			)

		self.t = self.solution.t
		self.trajectory = self.solution.y[3:]
		self.momentum = self.solution.y[:3]
		self.energy = np.sqrt(1 + np.sum(self.momentum**2, 0))


	def Plot3D(self):
		x = self.trajectory[0]
		y = self.trajectory[1]
		z = self.trajectory[2]
		fig = plt.figure().add_subplot(111, projection = '3d')
		fig.plot(x, y, z, label = 'parametric curve')
		fig.set_xlabel('$x$')
		fig.set_ylabel('$y$')
		fig.set_zlabel('$z$')
		plt.savefig('3D_trajectory.jpg')

	def Plot2D(self):
		x = self.trajectory[0]
		y = self.trajectory[1]
		z = self.trajectory[2]
		t = self.t
		fig, cell = plt.subplots(2, 2)

		# xy
		cell[0, 0].plot(y, x, linewidth = 1)
		cell[0, 0].set_xlabel('$x$')
		cell[0, 0].set_ylabel('$y$')
		# xz
		cell[0, 1].plot(z, x, linewidth = 1)
		cell[0, 1].set_xlabel('$x$')
		cell[0, 1].set_ylabel('$z$')
		# yz
		cell[1, 0].plot(z, y, linewidth = 1)
		cell[1, 0].set_xlabel('$y$')
		cell[1, 0].set_ylabel('$z$')
		# xyz t
		cell[1, 1].plot(x, t, linewidth = 1, label = 'x')
		cell[1, 1].plot(y, t, linewidth = 1, label = 'y')
		cell[1, 1].plot(z, t, linewidth = 1, label = 'z')
		cell[1, 1].set_xlabel('$t$')
		cell[1, 1].set_ylabel('$displacement$')
		cell[1, 1].legend()
		fig.tight_layout()

		plt.savefig('2D_trajectory')

	def Plot_pp_g(self):
		fig, cell = plt.subplots(2, 2)
		# px
		cell[0, 0].plot( self.t, self.momentum[0], linewidth = 1)
		cell[0, 0].set_xlabel('$t$')
		cell[0, 0].set_ylabel('$p_x$')
		# py
		cell[0, 1].plot(self.t, self.momentum[1], linewidth = 1)
		cell[0, 1].set_xlabel('$t$')
		cell[0, 1].set_ylabel('$p_y$')
		# pz
		cell[1, 0].plot(self.t, self.momentum[2], linewidth = 1)
		cell[1, 0].set_xlabel('$t$')
		cell[1, 0].set_ylabel('$p_z$')
		# gamma
		cell[1, 1].plot(self.t, self.energy, linewidth = 1)
		cell[1, 1].set_xlabel('$t$')
		cell[1, 1].set_ylabel('$\\gamma$')

		fig.tight_layout()
		#plt.show()
		plt.savefig('p(t), W(t)')


omega = 1.0
k = np.array([0.0, 0.0, 1.0])
tau = 70.0
ison = True
tmin = 0.
tf = 150.

Field = ElectromagneticField('circular', ison, k, tau, omega)
Field.I_plot('t', 'x')				#
Field.Field_plot('E', 't', 'x')		# order: t, x, y, z
Field.Field_plot('H', 't', 'x')		#

p = np.array([0, 0, 1])
x = np.array([1, 1, 1])
q = -1
m = 1
Electron = Particle(p, x, q, m, tf, tmin)
Electron.Solve(Field)
Electron.Plot3D()
Electron.Plot_pp_g()
Electron.Plot2D()
