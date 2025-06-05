import numpy as np


class FermionicGaussianHamiltonian(object):

	def __init__(self, Hmaj):

		self.Hmaj = Hmaj

		n = len(Hmaj) // 2
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

		self._Corr0 = np.zeros((2 * n, 2 * n), complex)
		self._Corr0[n:, n:] = np.eye(n)

		c0 = np.array([[0., 1.], [-1., 0.]])
		self._Cov0 = np.zeros((2 * n, 2 * n))
		for k in range(n):
			self._Cov0[2 * k: 2 * k + 2, 2 * k: 2 * k + 2] = c0

	def Hamiltonian_dirac(self):
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
		Hdirac = 1j * self._Omega_inv.T.conj() @ self.Hmaj @ self._Omega_inv
		return Hdirac

	def correlation_matrix_ground_state(self):
		Hdirac = self.Hamiltonian_dirac()
		E, V = np.linalg.eigh(Hdirac)
		if False: print(E[self.n])
		Corr_ground = V @ self._Corr0 @ V.T.conj()
		return Corr_ground

	def correlation_matrix_Gibbs(self, beta):
		Hdirac = self.Hamiltonian_dirac()
		E, V = np.linalg.eigh(Hdirac)
		E = np.real_if_close(E)
		Corr_beta = np.diag(1 / (1 + np.exp(-2 * beta * E)))
		Corr_Gibbs = V @ Corr_beta @ V.T.conj()
		return Corr_Gibbs

	def corr_to_cov(self, corr):
		return np.real_if_close(1j * (np.eye(2 * self.n) - self._Omega_inv @ corr @ self._Omega_inv.T.conj()))

	def cov_to_corr(self, cov):
		return self._Omega @ (1j * cov + np.eye(2 * self.n)) @ self._Omega.T.conj()

	def covariance_matrix_ground_state(self):
		Corr_ground = self.correlation_matrix_ground_state()
		return self.corr_to_cov(Corr_ground)

	def covariance_matrix_Gibbs(self, beta):
		Corr_Gibbs = self.correlation_matrix_Gibbs(beta)
		return self.corr_to_cov(Corr_Gibbs)


def generate_Hamiltonian_Majorana(g, n):
	'''
	Ising model Hamiltonian H = -\sum XX -g \sum Z
	'''
	H = np.zeros((2 * n, 2 * n))
	for ii in range(n):
		H[2 * ii, 2 * ii + 1] = g
		H[2 * ii + 1, 2 * ii] = -g
		if ii < n - 1:
			H[2 * ii + 1, 2 * ii + 2] = 1
			H[2 * ii + 2, 2 * ii + 1] = -1

	return H


def give_a_ground_state(n, g):
	Hmaj = generate_Hamiltonian_Majorana(g, n)
	FGH = FermionicGaussianHamiltonian(Hmaj)
	Cov = FGH.covariance_matrix_ground_state()
	return Cov


if __name__ == '__main__':
	n = 16
	g = 1

	Hmaj = generate_Hamiltonian_Majorana(g, n)

	FGH = FermionicGaussianHamiltonian(Hmaj)

	Cov = FGH.covariance_matrix_ground_state()
