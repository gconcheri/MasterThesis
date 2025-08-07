import numpy as np
from scipy import sparse
from scipy.linalg import expm
from pfapack import pfaffian as pf
from numba import njit

#for now, Omega, Corr don't work as they should!


class FermionicGaussianRepresentation:

	def __init__(self, model, diagonalcov = True):

		self.model = model

		n = model.Nsites // 2
		self.n = n

		Omega = np.zeros((2 * n, 2 * n), complex)
		for ii in range(n):
			Omega[ii, 2 * ii] = 1
			Omega[ii, 2 * ii + 1] = -1j
			Omega[ii + n, 2 * ii] = 1
			Omega[ii + n, 2 * ii + 1] = 1j
		Omega /= 2
		self._Omega = Omega
		self._Omega_inv = 2 * Omega.T.conj()

		# Omega_inv corresponds to the Omega in the paper, however it is not exactly the same.
		#i.e. Omega_inv here creates complex fermions pairing adjacent Majorana operators gamma_i gamma_i+1!
		self.Cov = build_covariance_matrix(model, diagonalcov = diagonalcov)
		self.Cov_0 = self.Cov.copy()
		self.Cov_e = self.Cov.copy()
		# self.Corr = self.cov_to_corr()
		self.h0_x = generate_h_Majorana(model, Jxx=1.0, Jyy=0.0, Jzz=0.0)
		self.h0_y = generate_h_Majorana(model, Jxx=0.0, Jyy=1.0, Jzz=0.0)
		self.h0_z = generate_h_Majorana(model, Jxx=0.0, Jyy=0.0, Jzz=1.0)

		self.he_x = generate_h_Majorana(model, Jxx=1.0, Jyy=0.0, Jzz=0.0, type='Anyon')
		self.he_y = generate_h_Majorana(model, Jxx=0.0, Jyy=1.0, Jzz=0.0, type='Anyon')
		self.he_z = generate_h_Majorana(model, Jxx=0.0, Jyy=0.0, Jzz=1.0, type='Anyon')


	""" These commented lines for now don't work and we don't need them!"""
	#%%
	# def Hamiltonian_dirac(self, Hmaj):
	# 	'''
	# 	Calculates the Dirac Hamiltonian for the Majorana Hamiltonian Hmaj.
	# 	Note: H = i (gamma1 ... gamma2n) Hmaj (gamma1 ... gamma2n)^T 

	# 	Parameters:
	# 	----------
	# 	Hmaj: array_like
	# 		2n x 2n real antisymmetric matrix.

	# 	Returns:
	# 	--------
	# 	Hdirac: array_like
	# 		Hamiltonian in terms of Dirac operators.
	# 		H = (c1 ... cn c1^dagger ... cn^dagger) Hdirac (c1^dagger ... cn^dagger c1 ... cn)^T
	# 	'''
	# 	Hdirac = 1j * self._Omega_inv.T.conj() @ Hmaj @ self._Omega_inv
	# 	return Hdirac
	

	# def corr_to_cov(self):
	# 	# return np.real_if_close(1j * (np.eye(2 * self.n) - self._Omega_inv @ self.Corr @ self._Omega_inv.T.conj()))
	# 	return np.real_if_close(1j * (0.5*np.eye(2 * self.n) - self._Omega_inv @ self.Corr @ self._Omega_inv.T.conj()))
	

	# def cov_to_corr(self):
	# 	# return self._Omega @ (1j * self.Cov + np.eye(2 * self.n)) @ self._Omega.T.conj()
	# 	return - self._Omega @ (1j * self.Cov + 0.5* np.eye(2 * self.n)) @ self._Omega.T.conj()
	
	# def update_corr_matrix(self, H, t):
	# 	self.Corr = expm(1j * 2 * H * t) @ self.Corr @ expm(- 1j * 2 * H * t)

	#%%	

	def update_cov_e_matrix(self, R):
		self.Cov_e = R @self.Cov_e @ R.T

	def update_cov_0_matrix(self, R):
		self.Cov_0 = R @self.Cov_0 @ R.T
	
	def reset_cov_0_matrix(self):
		self.Cov_0 = self.Cov.copy()
	
	def reset_cov_e_matrix(self):
		self.Cov_e = self.Cov.copy()

#%%
	def expectation_val_Majorana_string_old(self, model = None, small_loop = False, majoranas = None, factor = None):
		"""
		Using Wick's theorem, we calculate <gamma_i ... gamma_j> = (-1j)^(len(gamma_i...gamma_j)/2) Pf(cov_{i,...,j})
		where cov_{i,...,j} is the covariance matrix
		Important: i<...<j

		majoranas is a list of 0 and 1 of length 2*n, where 1 indicates that that majorana is included.

		"""
		if majoranas is not None:
			majoranas_bool = majoranas.astype(bool)	
			num_ones = np.sum(majoranas_bool)
			prefactor_maj = (-1j)**(num_ones/2) 
			if factor is not None:
				prefactor_maj *= factor
			Cov_0_reduced = self.Cov_0[majoranas_bool][:, majoranas_bool]		
			Cov_e_reduced = self.Cov_e[majoranas_bool][:, majoranas_bool]
			return prefactor_maj * pf.pfaffian(Cov_0_reduced), prefactor_maj * pf.pfaffian(Cov_e_reduced)
		
		if small_loop:
			# for small loop, we use the loop operator defined in the model
			prefactor, indices, links = model.get_small_loop()
		else:
			prefactor, indices, links, _ = model.get_loop()

		majoranas = np.zeros(model.Nsites)
		majoranas[indices] = 1  # sets all specified indices to 1
		majoranas_bool = majoranas.astype(bool)
		prefactor_maj = (-1j)**(len(indices)/2)

		prefactor *= prefactor_maj

		u = u_config(model, type="Anyon")

		exp = 0
		for i,j in links:
			if u[i,j] == -1:
				exp +=1
		
		prefactor_e = prefactor*(-1)**exp
		
		Cov_0_reduced = self.Cov_0[majoranas_bool][:, majoranas_bool]		
		Cov_e_reduced = self.Cov_e[majoranas_bool][:, majoranas_bool]
	
		return prefactor*pf.pfaffian(Cov_0_reduced), prefactor_e*pf.pfaffian(Cov_e_reduced)  # is the pfaffian directly the expectation value of the string?

#%%

	def expectation_val_Majorana_string(self, Cov = None, indices = None, factor = None):
		"""
		Using Wick's theorem, we calculate: factor * <gamma_i ... gamma_j> = factor * (-1j)^(len(gamma_i...gamma_j)/2) Pf(cov_{i,...,j})
		where cov_{i,...,j} is the covariance matrix
		Important: i<...<j

		indices is list of indices of the majorana fermions to include in the string!
		from this list we create the list majoranas, which is a list of 0 and 1 of length 2*n, where 1 indicates that that majorana is included.

		"""
		if Cov is None:
			Cov = self.Cov

		# majoranas = [1 if i in indices else 0 for i in range(model.Nsites)]

		majoranas = np.zeros(self.model.Nsites)
		majoranas[indices] = 1  # sets all specified indices to 1
		majoranas_bool = majoranas.astype(bool)
		prefactor_maj = (-1j)**(len(indices)/2)

		if factor is not None:
			prefactor_maj *= factor

		Cov_reduced = Cov[majoranas_bool][:, majoranas_bool]		
	
		return prefactor_maj*pf.pfaffian(Cov_reduced)
	

	def expectation_value_loop(self, small_loop = False):

		if small_loop:
			# for small loop, we use the loop operator defined in the model
			prefactor, indices, links = self.model.get_small_loop()
		else:
			prefactor, indices, links, _ = self.model.get_loop()

		u = u_config(self.model, type="Anyon")

		exp = 0
		for i,j in links:
			if u[i,j] == -1:
				exp +=1
		
		prefactor_e = (-1)**exp

		exp_value_0 = self.expectation_val_Majorana_string(self.Cov_0, indices, prefactor)
		exp_value_e = self.expectation_val_Majorana_string(self.Cov_e, indices, prefactor*prefactor_e)

		return exp_value_0, exp_value_e
	
	
	def order_parameter(self, small_loop = False):
		"""
		given loop op. = O
		We calculate Order parameter = <psi_e|O|psi_e>/<psi_0|O|psi_0> 
		where the two expectation values are calculated with Wick's theorem with previously defined function
		loop operator written as string of majoranas gamma_i ... gamma_j
		"""

		exp_value_0, exp_value_e = self.expectation_value_loop(small_loop=small_loop)

		return exp_value_e/exp_value_0, exp_value_0, exp_value_e
	
	#for not it is clear that the current density does not work as it should. This could be related to the amibiguity in the 
	#definition of the occupation number of the fermion dependent on a certain bond orientation! 
# %%

#%%
	
	def majorana_density_0(self, indices): # I should add link terms here, but for now I assume all phi_jk = 1
		"""
		returns n_jk = (1 - i phi_jk c_j c_k)/2

		with indices = [j,k]

		assumption: all phi_jk = 1

		"""
		j, k = indices

		return (1 - self.Cov_0[j,k])/2.
	
	def total_majorana_density(self, index, standard = False):
		"""
		given index j,
		returns n_j =  Σ_{i!=j} n_ij
		"""
		j = index
		n_j = 0

		if standard == False:
			for i in range(self.model.Nsites):
				if i != j:
					x_i, _ = self.model.id_to_idxidy(i)
					x_j, _ = self.model.id_to_idxidy(j)

					if x_i < x_j:
						n_j += self.majorana_density_0([i,j])
					else:
						n_j += self.majorana_density_0([j,i])
		else:
			for i in range(self.model.Nsites):
				if i !=j:
					n_j += self.majorana_density_0([j,i])

		# n_j = np.sum(self.Cov_0[j, :]) - self.Cov_0[j, j] #this is equivalent to the above loop
		# n_j = (1 - n_j)/2. #this is the same as above, but it is not necessary to divide by 2 here, since we are summing over all i!=j
		return n_j
	
	def current_density(self, standard = False):
		current_sites = self.model.get_current_sites()
		current_density = []
		for i in current_sites:
			current_density.append(self.total_majorana_density(i, standard = standard))
		
		return current_density
#%%
	
	def floquet_operator_ham(self, T = 1, alpha = -1, anyon = False):
		t = T*np.pi/4.
		if anyon:
			hx = self.he_x; hy = self.he_y; hz = self.he_z
		else:
			hx = self.h0_x; hy = self.h0_y; hz = self.h0_z
		Rx = expm(alpha*hx*t*4)
		Ry = expm(alpha*hy*t*4)
		Rz = expm(alpha*hz*t*4)
		R = Rz @ Ry @ Rx 
		return Rx, Ry, Rz, R
	
	
	
def floquet_operator(Op, t, alpha = -1):
	"""
	Calculates R = e^alpha 4ht, given h matrix of majorana operator
	"""
	return expm(alpha*Op*t*4)

# def floquet_operator(hx, hy, hz, T = 1, alpha = -1):
# 	t = T*np.pi/4.

# 	Rx = expm(alpha*hx*t*4)
# 	Ry = expm(alpha*hy*t*4)
# 	Rz = expm(alpha*hz*t*4)
# 	R = Rz @ Ry @ Rx 
# 	return Rx, Ry, Rz, R



def u_config(model, type=None):
	u = np.ones((model.Nsites, model.Nsites), dtype=np.complex128)

	if type == 'Anyon':
		anyon_bondlist = model.get_anyonbonds()[0]
		for i, j in anyon_bondlist:
			u[i, j] = -1
			u[j, i] = -1
			#they are both equal to -1 because the operators h, h_dis are already defined as ci cj - cj ci
			#recall that: u_ij ci cj = u_ji c_j c_i
	return u


def generate_h_Majorana(model, Jxx=1.0, Jyy=1.0, Jzz=1.0, type=None):

	"""
	Hamiltonian is given by:

	H = - i Σ_{i,j} J_{ij, alpha}/2 u_ij (γ_i γ_j - γ_j γ_i) = - i Σ_{i,j} (h_ij γ_i γ_j + h_ji γ_j γ_i)
	with the combination (i,j) counted only once in the sum

	(convention H = +J XX +J YY + J ZZ)

	where γ_i are Majorana operators and J_{ij} are the coupling constants, where alpha = xx, yy, zz.

	If Type= None -> all uijs are 1, i.e. all plaquette terms = +1
	If type= anyon: they are chosen in such a way that we create an anyon at the central plaquette of our system.

	Convention: i,j combination occurs only once, in the order of xx, yy, zz bonds
	i.e. (i ∈ A sublattice, j ∈ B sublattice).
	
	Parameters:
	model: Model object containing the lattice structure and bond information.
	
	Returns:
	h: Matrix representing the Hamiltonian in Majorana representation.

	"""
	#Initialize the Hamiltonian as a sparse matrix or as a numpy array
	h = np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)
	#H = sparse.csr_array((np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)))  
	xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()

	# Add xx bonds
	if Jxx != 0:
		for i, j in xx_bondlist:
			h[i,j] += Jxx/2.
			# h[j,i] += - Jxx/2.

	# Add yy bonds
	if Jyy != 0:
		for i, j in yy_bondlist:
			h[i,j] += Jyy/2.
			# h[j,i] += - Jyy/2.

	# Add zz bonds
	if Jzz != 0:
		for i, j in zz_bondlist:
			h[i,j] += Jzz/2.
			# h[j,i] += - Jzz/2.

	h = h - h.T #should be equivalent to doing commented operations: should save time but at the moment doesn't seem like it

	if type == "Anyon":
		u = u_config(model, type = 'Anyon')
		h = h * u
			
	return h

#for now this term works only for OBC class!
def generate_disorder_term(model, cov, delta, type = None, disc = False):
	"""
	Disorder Potential is given by:

	V = i Σ_phi delta_phi phi_ij (γ_i γ_j - γ_j γ_i) = i Σ_{i,j} (h_dis_ij γ_i γ_j + h_dis_ji γ_j γ_i)
	with the combination (i,j) counted only once in the sum
	here i,j are the diagonal sites and phi indicates the bond we are considering

	"""

	diagonal_bonds = model.get_diagonalbonds()


	links_list = model.links_list
	values_list = []

	h_dis = np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)


	if type == "Anyon":
		u = u_config(model, type = type)
		for phi in links_list:
			value = 1
			for i,j in phi:
				value *=u[i,j]
			values_list.append(value) 

		for idx, (i,j) in enumerate(diagonal_bonds):
			rand_num = np.random.uniform(-delta, delta)
			h_dis[i,j] = rand_num * values_list[idx] * cov[i,j]

	for i,j in diagonal_bonds:
		rand_num = np.random.uniform(-delta, delta)
		h_dis[i,j] = rand_num * cov[i,j]

	h_dis = h_dis - h_dis.T

	return h_dis

def build_covariance_matrix(model, diagonalcov = True):
	"""
	Builds the covariance matrix of the initial states |psi_0> and |psi_e> of the 
	Honeycomb model
	
	cov_ij = i <(γ_i γ_j - γ_j γ_i)> / 2
	
	I.e. in every plaquette, the majorana fermions are coupled diagonally, thus 
	giving rise to diagonal dirac/complex unoccupied fermion (parity +1)
	"""

	diag_bonds = model.get_diagonalbonds()
	_, _, zz_bonds = model.get_bonds()
	Cov = np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)

	if diagonalcov:
		if hasattr(model, 'cov_value') and model.cov_value is not None:
			for idx, (i, j) in enumerate(diag_bonds):
				Cov[i, j] = model.cov_value[idx]
		else:
			for i, j in diag_bonds:
				Cov[i, j] = 1
				# Cov[j, i] = - 1
	else:
		for i,j in zz_bonds:
			Cov[i, j] = 1
			# Cov[j, i] = - 1

	Cov = Cov - Cov.T

	return Cov


# @njit #only works with purely numpy arrays!
# def cov_update(Cov, R):
#     return R @ Cov @ R.T



# if __name__ == '__main__':
# 	n = 16
# 	g = 1

# 	Hmaj = generate_h_Majorana(g, n)

# 	FGH = FermionicGaussianRepresentation(Hmaj)

# 	Cov = FGH.covariance_matrix_ground_state()

# %%
