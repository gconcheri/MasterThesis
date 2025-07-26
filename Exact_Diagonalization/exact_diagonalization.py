import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, expm

X = sparse.csr_array([[0.,1.],[1.,0.]])
Y = sparse.csr_array([[0.,-1.j],[1.j,0.]])
Z = sparse.csr_array([[1.,0.],[0.,-1.]])
I = sparse.csr_array(np.eye(2))

Xindexlist = [[1,2],[4,7],[6,9],[10,13]]
Yindexlist = [[1,6],[3,4],[9,12],[7,10]]

def nonzero_values(array):
    non_zero_indices = np.nonzero(array)[0]  # Get the indices of non-zero values
    non_zero_values = array[non_zero_indices]  # Get the non-zero values
    print("Positions of non-zero values:", non_zero_indices)
    print("Non-zero values of array:", non_zero_values)


#We create operator that applies hadamard only on these two representative qubits
#where representative qubits are indexed as qubits in the toric code space:
#toric qubit 0 -> honeycomb qubits (0,1)
def Hadamard(qubit1, qubit2, n):
    assert qubit1 < qubit2
    Id = sparse.csr_array(np.eye(4))
    H_small = sparse.csr_array([[1./np.sqrt(2), 0., 0., 1./np.sqrt(2)], [0., 1., 0., 0.], [0., 0., 1., 0.], [1./np.sqrt(2), 0., 0., -1./np.sqrt(2)]])
    op_list = [Id]*n
    op_list[qubit1] = H_small
    op_list[qubit2] = H_small
    
    H = op_list[0]

    for op in op_list[1:]:
        H = sparse.kron(H,op, format='csr')
    return H
    
# Create op. that applies CNOT on each plaquette with representative qubits as control qubits and all 
# other qubits of plaquette as target ones

def CNOT(qubit1, qubit2, n, l): #l = effective spins per plaquette, n = tot EFFECTIVE spins
    assert qubit1 < 3
    assert 3 < qubit2
    # Id = sparse.csr_array(np.eye(2))
    # X = sparse.csr_array([[0.,1.],[1.,0.]])
    # Proj_0 = sparse.csr_array([[1.,0.],[0.,0.]]) #|0><0|
    # Proj_1 = sparse.csr_array([[0.,0.],[0.,1.]]) #|1><1|
    
    newId = sparse.csr_array(np.eye(2**(2*n-2*l)))
    CNOT_p1 = sparse.kron(plaquetteCNOT(qubit1,l),newId,'csr')
    CNOT_p2 = sparse.kron(newId,plaquetteCNOT(qubit2-l+1,l), 'csr') #we map qubit 3 -> 0, 4->1, 5->2, 6->3
    # even if geometry is not conserved, it still works because CNOT applied on this 
    #sequence of hilbert spaces in this order does not depend on geometry!
    # print(CNOT_p2 @ CNOT_p1)
    # print(CNOT_p1 @ CNOT_p2)
    assert np.array_equal((CNOT_p2 @ CNOT_p1).toarray(), (CNOT_p1 @ CNOT_p2).toarray()), "Sparse arrays are not equal"
    return CNOT_p2 @ CNOT_p1

def plaquetteCNOT(qubit, l):
    Id = sparse.csr_array(np.eye(4))
    X = np.zeros((4,4))
    X[0,3] = 1.
    X[3,0] = 1.
    X = sparse.csr_array(X)
    #print(X)
    
    Proj_0 = np.zeros((4,4)) #|00><00|
    Proj_0[0,0] = 1
    Proj_0 = sparse.csr_array(Proj_0)
    Proj_1 = np.zeros((4,4)) #|11><11|
    Proj_1[3,3] = 1
    Proj_1 = sparse.csr_array(Proj_1)

    op_list_id = [Id]*l
    op_list_id[qubit] = Proj_0

    op_list_x = [X]*l
    op_list_x[qubit] = Proj_1

    CNOT1 = op_list_id[0]
    CNOT2 = op_list_x[0]

    for op1, op2 in zip(op_list_id[1:],op_list_x[1:]):
        CNOT1 = sparse.kron(CNOT1, op1, 'csr')
        CNOT2 = sparse.kron(CNOT2, op2, 'csr')
    return CNOT1 + CNOT2


def U_a_full(n_eff_spins):
    U_a = sparse.csr_array([[(1.+1.j)/2., 0., 0., (1.+1.j)/2.],[0., 1., 0., 0.], [0., 0., 1., 0.], [(-1.+1.j)/2., 0., 0., (1.-1.j)/2.]])
    Id = sparse.csr_array(np.eye(4))
    op_list = [U_a]*n_eff_spins
    op_list[1] = Id
    op_list[3] = Id
    op_list[6] = Id

    U = op_list[0]

    for op in op_list[1:]:
        U = sparse.kron(U,op, format='csr')
    return U

def U_b_full(n_eff_spins):
    U_b = sparse.csr_array([[1./np.sqrt(2)*(1.-1.j),0., 0., 0.],[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1./np.sqrt(2)*(1.+1.j)]])
    Id = sparse.csr_array(np.eye(4))
    op_list = [Id]*n_eff_spins
    op_list[1] = U_b
    op_list[3] = U_b
    op_list[6] = U_b

    U = op_list[0]

    for op in op_list[1:]:
        U = sparse.kron(U,op, format='csr')
    return U

def flux_freestate(n_spins, qubit1, qubit2):
    #n_spins must be 14
    e1 = np.array([1,0])
    psi = e1
    for _ in range(n_spins-1):
        psi = np.kron(psi,e1)

    n_eff_spins = int(n_spins/2)
    H = Hadamard(qubit1,qubit2,n_eff_spins)
    print(H.shape)
    psi1 = H @ psi.copy()

    CNOT_op = CNOT(qubit1, qubit2, n_eff_spins, 4)
    psi2 =  CNOT_op @ psi1.copy()

    psi_fluxfree = U_a_full(n_eff_spins) @ U_b_full(n_eff_spins) @ psi2.copy()

    return psi_fluxfree

def diagonal_state(n_spins, qubit1, qubit2):

    psi_fluxfree = flux_freestate(n_spins, qubit1, qubit2).copy()

    psi_diagonal1 = floquet_unitary(psi_fluxfree, n_spins, applyX = False, applyY = True, applyZ = False, Yindexlist = [[3,4],[7,10]])
    psi_diagonal2 = psi_diagonal1.copy()
    psi_diagonal2 = floquet_unitary(psi_diagonal2, n_spins, applyX = True, applyY = False, applyZ = False, Xindexlist = [[4,7],[10,13]])
    psi_diagonal = psi_diagonal2.copy()

    return psi_diagonal


    



def Op_full(O, Id, pos, n):
    op_list = [Id]*n

    if isinstance(pos, list):
        for p in pos:
            op_list[p] = O
    else:
        op_list[pos] = O

    fullOp = op_list[0]
    for op in op_list[1:]:
        fullOp = sparse.kron(fullOp,op, format='csr')
    return fullOp   

def Op_string(Op_list, index_list, Id, n_spins):
    assert len(Op_list) == len(index_list)
    assert len(Op_list) > 0
    assert len(index_list) > 0
    full_op = []
    for op, idx in zip(Op_list, index_list):
        full_op.append(Op_full(op, Id, idx, n_spins))
    W = full_op[0]
    for op in full_op[1:]:
        W = W @ op
    return W

def floquet_unitary(psi, n_spins, T = 1, X = X, Z = Z, Y = Y, I = I, 
                    applyX = True, applyY = True, applyZ = True,
                    Xindexlist = Xindexlist, Yindexlist = Yindexlist):
    #apply XX unitary

    psi_t = psi

    i = 0
    j = 0
    z = 0

    #maybe do first sum of sparse matrices XX and then call one unique expm_mulitply

    if applyX:
        psi_t = expm_multiply(-1.j * T *np.pi/4. * Op_full(X, I, Xindexlist[0], n_spins), psi_t)

        if len(Xindexlist) > 1:
            for indeces in Xindexlist[1:]:
                psi_t = expm_multiply(-1.j * T * np.pi/4. * Op_full(X, I, indeces, n_spins), psi_t)
        i = 1

    #apply YY unitary

    if applyY:
        psi_t = expm_multiply(-1.j * T *np.pi/4. * Op_full(Y, I, Yindexlist[0], n_spins), psi_t)

        if len(Yindexlist) > 1:
            for indeces in Yindexlist[1:]:
                psi_t = expm_multiply(-1.j * T * np.pi/4. * Op_full(Y, I, indeces, n_spins), psi_t)
        j = 1

    #apply ZZ unitary: check
    if applyZ:
        for n in range(0, n_spins, 2):
            psi_t = Op_full(expm(-1.j * T * np.pi/4. * sparse.kron(Z,Z, 'csr')), I, n, n_spins-1) @ psi_t
        z = 1
    
    print("i,j,z", i,j,z)

    return psi_t


def floquet_evolution(psi, n_cycles, n_spins, ZZ, T = 1, X = X, Z = Z, Y = Y, I = I, 
                      applyX = True, applyY = True, applyZ = True,
                      Xindexlist = Xindexlist, Yindexlist = Yindexlist):
    # Apply the Floquet unitary operator to the initial state
    list_zz = []
    assert psi.ndim == 1 #check if array is 1D: (len,)
    psi_dagger = psi.conj()
    list_zz.append(psi_dagger @ ZZ @ psi)
    for _ in range(n_cycles):
        psi = floquet_unitary(psi, n_spins, T, X, Z, Y, I)
        psi_dagger = psi.conj()
        list_zz.append(psi_dagger @ ZZ @ psi)

    return psi, list_zz