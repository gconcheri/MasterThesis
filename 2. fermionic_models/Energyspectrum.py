import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Class_site as site
import importlib
importlib.reload(site)




def plot_dispersion_relation(model, T, h0_x, h0_y, h0_z):
    t = T*np.pi/4.
    R0x = expm(h0_x*t*4)
    R0y = expm(h0_y*t*4)
    R0z = expm(h0_z*t*4)

    R = R0z @ R0y @ R0x 

    S = model.reordering_operator()
    print(S.shape)
    print(np.linalg.norm(S @ S.T - np.eye(S.shape[0])))
    FT, ks1, ks2 = model.FTOperator()

    # block-diagonalize
    Rk = FT.T.conj() @ R @ FT

    len_block = 2

    # diagonalize blocks
    eps = []
    for i in range(model.Npx*model.Npy):
        e = np.linalg.eigvals(Rk[len_block*i:len_block*i+len_block, len_block*i:len_block*i+len_block])
        eps.append(e)
    eps = np.real_if_close(1j * np.log(eps))

    #better way of plotting this!

    # Create a meshgrid for qx and qy values
    qx_vals, qy_vals = np.meshgrid(ks1, ks2)

    eps_array = np.reshape(eps, -1)
    energies_all = np.zeros((len(ks1)*len(ks2), len_block))
    # eps_array = np.concatenate([eps_array, eps_array[0:len_block]])
    i = 0
    for start in range(0, len(eps_array), len_block):
        end = start + len_block
        energies_all[i, :] = np.sort(eps_array[start:end])
        i += 1


    # Reshape the energies to match the meshgrid dimensions
    energies_all = energies_all.reshape(len(ks2), len(ks1), len_block)

    # Start 3D plotting
    fig = plt.figure(#figsize=(10, 10)
                    )
    ax = fig.add_subplot(111, projection='3d')

    # Set aspect ratio: (x, y, z). Increase z to stretch vertically
    ax.set_box_aspect((1, 1, 1))  # Try (1, 1, 2), (1, 1, 3), etc.

    # Parameters for the surface plot
    surface_opts = {
        'linewidth': 0,
        'antialiased': True,  # meglio “True” per superfici lisce
        'rstride': 1,
        'cstride': 1,
        'alpha': 0.9,
        'cmap': 'viridis'
    }

    for band in range(len_block):
        energy_band = energies_all[:, :, band]
        energy_band = energy_band.round(6)  # Rounding to avoid visual glitches

        #Plot the surface for each energy band
        ax.plot_surface(qx_vals, qy_vals, energy_band,
                    **surface_opts)
        # print(energy_band)

    ax.set_xlabel('$q_x$')
    ax.set_ylabel('$q_y$')
    ax.set_zlabel('Energy $\omega(q)$')
    ax.view_init(elev=10, azim=10)
    ax.set_title(f'Disp relation, JT = {T}' )
    plt.tight_layout()
    plt.show()