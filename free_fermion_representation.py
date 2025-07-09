import numpy as np
from scipy import sparse
from scipy.linalg import expm
from pfapack import pfaffian as pf


class FermionicGaussianRepresentation:

	def __init__(self, model, diagonalcov = True):

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

		self.Cov = build_covariance_matrix(model, diagonalcov)
		self.Corr = self.cov_to_corr()




	def Hamiltonian_dirac(self, Hmaj):
		'''
		Calculates the Dirac Hamiltonian for the Majorana Hamiltonian Hmaj.
		Note: H = i (gamma1 ... gamma2n) Hmaj (gamma1 ... gamma2n)^T 

		Parameters:
		----------
		Hmaj: array_like
			2n x 2n real antisymmetric matrix.

		Returns:
		--------
		Hdirac: array_like
			Hamiltonian in terms of Dirac operators.
			H = (c1 ... cn c1^dagger ... cn^dagger) Hdirac (c1^dagger ... cn^dagger c1 ... cn)^T
		'''
		Hdirac = 1j * self._Omega_inv.T.conj() @ Hmaj @ self._Omega_inv
		return Hdirac

	def corr_to_cov(self):
		# return np.real_if_close(1j * (np.eye(2 * self.n) - self._Omega_inv @ self.Corr @ self._Omega_inv.T.conj()))
		return np.real_if_close(1j * (0.5*np.eye(2 * self.n) - self._Omega_inv @ self.Corr @ self._Omega_inv.T.conj()))
	

	def cov_to_corr(self):
		# return self._Omega @ (1j * self.Cov + np.eye(2 * self.n)) @ self._Omega.T.conj()
		return - self._Omega @ (1j * self.Cov + 0.5* np.eye(2 * self.n)) @ self._Omega.T.conj()

	
	def update_corr_matrix(self, H, t):
		self.Corr = expm(1j * 2 * H * t) @ self.Corr @ expm(- 1j * 2 * H * t)
	
	# def expectation_value(self, model, site_list):
	# 	"""
	# 	Using Wick's theorem, we calculate <gamma_i ... gamma_j> = Pf(cov_{i,...,j})
	# 	where cov_{i,...,j} is the covariance matrix bla bla bla

	# 	Important: i<...<j
	# 	"""

	def update_cov_matrix(self, R):
		self.Cov = R @self.Cov@ R.T

	def expectation_val_Majorana_string(self, majoranas):
		"""
		majoranas is a list of 0 and 1 of length 2*n, where 1 indicates that that majorana is included.
		"""
		
		majoranas_bool = majoranas.astype(bool)
		
		Cov_reduced = self.Cov[majoranas_bool][:, majoranas_bool]
	
		return pf.pfaffian(Cov_reduced) # is the pfaffian directly the expectation value of the string?



def u_config(model, type=None):
    xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()
    u = np.ones((model.Nsites, model.Nsites), dtype=np.complex128)

    if type == 'Anyon':
        anyon_bondlist, _, _ = model.get_anyonbonds()
        for i, j in anyon_bondlist:
            u[i, j] = -1
            u[j, i] = -1
    return u


def generate_Hamiltonian_Majorana(model, Jxx=1.0, Jyy=1.0, Jzz=1.0, type=None):

	"""
	H = i Σ_{i,j} J_{ij, alpha}/2 u_ij (γ_i γ_j - γ_j γ_i)

	where γ_i are Majorana operators and J_{ij} are the coupling constants, where alpha = xx, yy, zz.

	If Type= None -> all uijs are 1, i.e. all plaquette terms = +1
	If type= anyon: they are chosen in such a way that we create an anyon at the central plaquette of our system.

	Convention: i,j combination occurs only once, in the order of xx, yy, zz bonds
	i.e. (i ∈ A sublattice, j ∈ B sublattice).
	
	Parameters:
	model: Model object containing the lattice structure and bond information.
	
	Returns:
	H: Matrix representing the Hamiltonian in Majorana representation.

	"""
	#Initialize the Hamiltonian as a sparse matrix or as a numpy array
	H = np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)
	#H = sparse.csr_array((np.zeros((model.Nsites, model.Nsites), dtype=np.complex128)))  
	xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()

	# Add xx bonds
	if Jxx != 0:
		print(f"Adding xx bonds with Jxx = {Jxx}")
		for i, j in xx_bondlist:
			H[i,j] += 1j * Jxx/2.
			# H[j,i] += - 1j * Jxx/2.

	# Add yy bonds
	if Jyy != 0:
		print(f"Adding yy bonds with Jyy = {Jyy}")
		for i, j in yy_bondlist:
			H[i,j] += 1j * Jyy/2.
			# H[j,i] += - 1j * Jyy/2.

	# Add zz bonds
	if Jzz != 0:
		print(f"Adding zz bonds with Jzz = {Jzz}")
		for i, j in zz_bondlist:
			H[i,j] += 1j * Jzz/2.
			# H[j,i] += - 1j * Jzz/2.

	H = H - H.T #should be equivalent to doing commented operations: should save time but at the moment doesn't seem like it

	if type == "Anyon":
		print('type = Anyon!')
		u = u_config(model, type = 'Anyon')
		H = H * u
			
	return H
	

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
		for i, j in diag_bonds:
			Cov[i, j] = 1
			Cov[j, i] = - 1
	else:
		for i,j in zz_bonds:
			Cov[i, j] = 1
			Cov[j, i] = - 1

	return Cov




if __name__ == '__main__':
	n = 16
	g = 1

	Hmaj = generate_Hamiltonian_Majorana(g, n)

	FGH = FermionicGaussianRepresentation(Hmaj)

	Cov = FGH.covariance_matrix_ground_state()
