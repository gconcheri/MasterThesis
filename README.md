# MasterThesis

# Exact Diagonalization (folder):
In the 3 jupyter files I create a 14 spin system by working on the full Hilbert space (i.e. 2^14 linearly independent states). I then create the flux-free state in 2 different ways and then drive it with the floquet operator. I then measure the plaquette operator and other pauli string ops to check whether the driving works correctly, going from an e anyon to an m anyon
Model.ipynb - first work on toric code space (dim: 2^7) and then map it to honeycomb space
Model_new.ipynb - I work directly on the honeycomb space, I use mainly numpy arrays
Model_new_sparsearray... - I work directly on the honeycomb space, I use mainly sparse arrays

# Class_site.py:
file where I create the classes to build the honeycomb lattice, depending on the boundary conditions I want to impose (PBCx, OBC, PBCxy). Here, I decide how to number the sites, and thus how to build the xx_bonds, yy_bonds, zz_bonds and so on. This class is needed to then create the Hamiltonian and the Floquet operator based on the bonds calculated in these classes! I also define the coordinates depending on the boundary conditions, which are used to plot the lattice.

In the PBC classes I also create the F.T. operators (and the respective ordering operators needed to define them) to then block diagonalize the Floquet operator and consequently plot the quasi-energy dispersion relations!

# honeycomb_plot.py:
file where I define the functions to plot the honeycomb lattice depending on the boundary conditions, according to the xx_bonds, yy_bonds, zz_bonds and the coordinates defined in Class_site.py. 

# free_fermion_representation.py:
file where I create the class that builds the Floquet operator and the Hamiltonian in the free fermion representation, i.e. in terms of Majorana operators. I also create the correlation matrix and the covariance matrix, which are needed to measure the expectation values of the plaquette operator and other Pauli string operators.

# fermionic_model_OBC.ipynb:
here I implement the class for OBC defined in Class_site.py to see if everything works, I plot the honeycomb lattice using honeycomb_plot.py and then I check the expectation value of the loop operator using the fermionic gaussian representation defined in free_fermion_representation.py. 

# fermionic_model_PBCx.ipynb:
here I implement the class for PBCx defined in Class_site.py to see if everything works, I plot the honeycomb lattice and also the cylindric honeycomb lattice using honeycomb_plot.py and then I check the expectation value of the loop operator using the fermionic gaussian representation defined in free_fermion_representation.py. I also plot the quasi-energy dispersion relations by block-diagonalizing the Floquet operator using the F.T. operators defined in Class_site.py.

# fermionic_model_PBCxy.ipynb:
here I implement the class for PBC along both x and y directions, defined in Class_site.py to see if everything works, I plot the toric honeycomb lattice using honeycomb_plot.py and then I check the expectation value of the loop operator using the fermionic gaussian representation defined in free_fermion_representation.py. I also plot the quasi-energy dispersion relations by block-diagonalizing the Floquet operator.

# test_floquet.ipynb:
Here I test whether the Floquet operator defined as R=e^4h works correctly!


TO DO:

maybe even actually add all the files in respective folders and then add paths to all files such that I can use files even if they are in different folder: maybe there is a easier way! i.e. check project_activate

aggiungi funzioni di plot nella simulazione cluster, in modo che vengano direttamente salvati alcuni plot nella directory

sistema cluster folder su workstation! allinealo con quello che hai nel pc