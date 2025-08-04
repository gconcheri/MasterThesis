import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_honeycomb(model, 
                    fig_size=(20,20), 
                    highlight_idxidy=None, sites=None, highlight_color='orange', #inputs to highlight sites
                    plaquette_site= None, #input to shade a certain plaquette with lowest site= plaquette_site
                    loop = False,
                    plot_anyon_bonds=False, #input to draw anyon bonds
                    plot_diagonal_bonds = False, #input to draw diagonal bonds (explained above)
                    otherbonds_list = None, #input to draw any list of bonds
                    nonzeropairs = None, Cov = None #nonzeropairs is the list of bonds for which the covariance matrix is non-zero
                    ):
    coords = model.get_coordinates()
    xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()
    
    plt.figure(figsize=fig_size)

    # Plot sites:
    
    #Plot lattice sites
    for i in range(model.Nsites):
        if model.partition[i] == 'A':
            plt.scatter(coords[i, 0], coords[i, 1], color='k', s=100, marker='o', zorder=3)  # full dot
        else:
            plt.scatter(coords[i, 0], coords[i, 1], facecolors='white', edgecolors='k', s=100, marker='o', zorder=3)  # empty dot
    # plt.scatter(coords[:, 0], coords[:, 1], color='k', zorder=3)

    #Shade a specific plaquette
    if plaquette_site is not None:
        plaquette_indices = model.get_plaquettecoordinates(plaquette_site)
        # get their coordinates
        plaquette_coords = coords[plaquette_indices, :]  # only x and y for 2D
        polygon = Polygon(plaquette_coords, closed=True, facecolor='grey', alpha=0.3, edgecolor='none')
        plt.gca().add_patch(polygon)

    if loop:
        plaquette_indices = model.get_loop()[3] #list of sublists, where each sublists contains indices of one of 4 plaquettes at the vertices of the loop
        loop_coordinates = [] #here is where we will save the 4 central plaquette coordinates, in order to draw the loop

        for p in plaquette_indices:
            plaquette_coords = coords[np.array(p)]
            mean_x = np.mean(plaquette_coords[:, 0])
            mean_y = np.mean(plaquette_coords[:, 1])
            loop_coordinates.append((mean_x, mean_y))

        loop_coordinates = np.array(loop_coordinates)
        len_loop = len(plaquette_indices)
        for i in range(len_loop):
            plt.plot([loop_coordinates[i,0], loop_coordinates[(i+1)%len_loop,0]], [loop_coordinates[i,1], loop_coordinates[(i+1)%len_loop,1]], '--', color = 'green', lw=2)


        # loop_coords = [model.get_central_plaquette_coord(idx = i[0], idy = i[1]) for i in loop_p_coords]
        # for i,j in loop_coords:
            # plt.plot([loop_coords[i, 0], loop_coords[j, 0]], [coords[i, 1], coords[j, 1]], '--', lw=2)
  


    # Highlight a specific site if requested
    if highlight_idxidy is not None:
        idx, idy = highlight_idxidy
        site_id = model.idxidy_to_id(idx, idy)
        plt.scatter(coords[site_id, 0], coords[site_id, 1], color=highlight_color, s=300, zorder=5, label='highlighted site')

    #Highlight a list of sites if requested
    if sites is not None:
        for site_id in sites:
            plt.scatter(coords[site_id, 0], coords[site_id, 1], color=highlight_color, s=300, zorder=5, label='highlighted sites')

    # Plot bonds:

    #Plot xx,yy,zz bonds
    for bond in xx_bondlist:
        i, j = bond
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'r-', label='xx' if bond == xx_bondlist[0] else "", lw=2)
    for bond in yy_bondlist:
        i, j = bond
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'b-', label='yy' if bond == yy_bondlist[0] else "", lw=2)
    for bond in zz_bondlist:
        i, j = bond
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'g-', label='zz' if bond == zz_bondlist[0] else "", lw=2)

    # Plot anyon bonds if requested
    if plot_anyon_bonds:
        anyon_bonds = model.get_anyonbonds()[0]
        for i, j in anyon_bonds:
            plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color='magenta', lw=5, label='anyon bond' if (i, j) == anyon_bonds[0] else "", zorder=2)

    #plot diagonal bonds if requested
    if plot_diagonal_bonds:
        diag_bonds = model.get_diagonalbonds()
        for i, j in diag_bonds:
            plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color='purple', lw=2, label='diagonal bond' if (i, j) == diag_bonds[0] else "", zorder=2)    

    #plot other bonds if requested
    if otherbonds_list is not None:
        for i, j in otherbonds_list:
            plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color='pink', lw=3, label='other links' if (i, j) == otherbonds_list[0] else "", zorder=2)
    
    if nonzeropairs is not None and Cov is not None:
        for i, j in nonzeropairs:
            value = abs(Cov[i, j])
            # Scale the linewidth for better visibility (adjust the multiplier as needed)
            lw = 1 + 5 * value
            plt.plot([coords[i, 0], coords[j, 0]],[coords[i, 1], coords[j, 1]],color='orange',lw=lw,zorder=2)

    plt.axis('equal')
    plt.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def plot_honeycomb_cylinder(
    model,
    plot_static=True,
    make_gif=False,
    gif_filename="cylinder_rotation.gif",
    elev=20,
    azim=20,
    dotsize = 100,
    fig_size=(10, 30),
    frames=90, #fluidity of gif
    interval=50, # time interval between frames
    highlight_idxidy=None,
    sites=None,
    plaquette_site=None, 
    highlight_color='orange',
    plot_anyon_bonds=False,
    plot_diagonal_bonds=False
):
    coords = model.get_coordinates_cylindric()
    xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    def draw(ax, elev, azim):
        ax.cla()
        #Plot lattice sites:
        for i in range(model.Nsites):
            if model.partition[i] == 'A':
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2], color='k', s=dotsize, marker='o', zorder=3)  # full dot
            else:
                ax.scatter(coords[i, 0], coords[i, 1], coords[i,2], facecolors='white', edgecolors='k', s=dotsize, marker='o', zorder=3)  # empty dot

        # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='k', s=10, zorder=3)

        #Shade a specific plaquette
        if plaquette_site is not None:
            plaquette_indices = model.get_plaquettecoordinates(id = plaquette_site)
            # get their coordinates
            verts = [coords[plaquette_indices]]  # shape (1, 6, 3)
            poly = Poly3DCollection(verts, facecolor='grey', alpha=0.3, edgecolor='none')
            ax.add_collection3d(poly)

        # Highlight a specific site if requested
        if highlight_idxidy is not None:
            idx, idy = highlight_idxidy
            site_id = model.idxidy_to_id(idx, idy)
            ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                       color=highlight_color, s=dotsize, zorder=5)

        # Highlight a list of sites if requested
        if sites is not None:
            for site_id in sites:
                ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                           color=highlight_color, s=dotsize, zorder=5)

        #Plot bonds:
        for bond, color in zip([xx_bondlist, yy_bondlist, zz_bondlist], ['r', 'b', 'g']):
            for i, j in bond:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color=color, lw=2)
        # Plot anyon bonds if requested
        if plot_anyon_bonds:
            anyon_bonds, px, py = model.get_anyonbonds()
            for i, j in anyon_bonds:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color='magenta', lw=5, zorder=1)
        # Plot diagonal bonds if requested
        if plot_diagonal_bonds:
            diag_bonds = model.get_diagonalbonds()
            for i, j in diag_bonds:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color='green', lw=2, zorder=1)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # Imposta limiti assi per evitare "schiacciamento"
        ax.set_box_aspect([1,1,1])  # assi proporzionati
        ax.set_xlim(np.min(coords[:,0]), np.max(coords[:,0]))
        ax.set_ylim(np.min(coords[:,1]), np.max(coords[:,1]))
        ax.set_zlim(np.min(coords[:,2]), np.max(coords[:,2]))

    if plot_static:
        draw(ax, elev, azim)
        plt.show()

    if make_gif:
        def update(angle):
            draw(ax, angle, angle)
        ani = animation.FuncAnimation(
            fig, update, frames=np.linspace(0, 360, frames), interval=interval
        )
        ani.save(gif_filename, writer='pillow') # or ani.save(filename, writer='ffmpeg') with filename = "ciao.mp4"

def plot_honeycomb_torus(
    model,
    plot_static=True,
    make_gif=False,
    gif_filename="cylinder_rotation.gif",
    elev=20,
    azim=20,
    dotsize = 100,
    fig_size=(10, 30),
    frames=90, #fluidity of gif
    interval=50, # time interval between frames
    highlight_idxidy=None,
    sites=None,
    plaquette_site=None, 
    highlight_color='orange',
    plot_anyon_bonds=False,
    plot_diagonal_bonds=False,
    r_tilde = 0.5,
    r_0 = 2
):
    coords = model.get_coordinates_torus(r_tilde = r_tilde, r_0 = r_0)
    xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    def draw(ax, elev, azim):
        ax.cla()
        #Plot lattice sites:
        for i in range(model.Nsites):
            if model.partition[i] == 'A':
                ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2], color='k', s=dotsize, marker='o', zorder=3)  # full dot
            else:
                ax.scatter(coords[i, 0], coords[i, 1], coords[i,2], facecolors='white', edgecolors='k', s=dotsize, marker='o', zorder=3)  # empty dot

        # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='k', s=10, zorder=3)

        #Shade a specific plaquette
        if plaquette_site is not None:
            plaquette_indices = model.get_plaquettecoordinates(plaquette_site)
            # get their coordinates
            verts = [coords[plaquette_indices]]  # shape (1, 6, 3)
            poly = Poly3DCollection(verts, facecolor='grey', alpha=0.3, edgecolor='none')
            ax.add_collection3d(poly)

        # Highlight a specific site if requested
        if highlight_idxidy is not None:
            idx, idy = highlight_idxidy
            site_id = model.idxidy_to_id(idx, idy)
            ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                       color=highlight_color, s=dotsize, zorder=5)

        # Highlight a list of sites if requested
        if sites is not None:
            for site_id in sites:
                ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                           color=highlight_color, s=dotsize, zorder=5)

        #Plot bonds:
        for bond, color in zip([xx_bondlist, yy_bondlist, zz_bondlist], ['r', 'b', 'g']):
            for i, j in bond:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color=color, lw=2)
        # Plot anyon bonds if requested
        if plot_anyon_bonds:
            anyon_bonds = model.get_anyonbonds()[0]
            for i, j in anyon_bonds:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color='magenta', lw=5, zorder=4)
        # Plot diagonal bonds if requested
        if plot_diagonal_bonds:
            diag_bonds = model.get_diagonalbonds()
            for i, j in diag_bonds:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]],
                        [coords[i, 2], coords[j, 2]],
                        color='green', lw=2, zorder=4)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        # Imposta limiti assi per evitare "schiacciamento"
        ax.set_box_aspect([1,1,1])  # assi proporzionati
        ax.set_xlim(-(r_0+r_tilde), r_0+r_tilde)
        ax.set_ylim(-(r_0+r_tilde), r_0+r_tilde)
        ax.set_zlim(-(r_0+r_tilde), r_0+r_tilde)

    if plot_static:
        draw(ax, elev, azim)
        plt.show()

    if make_gif:
        def update(angle):
            draw(ax, elev, angle)
        ani = animation.FuncAnimation(
            fig, update, frames=np.linspace(0, 360, frames), interval=interval
        )
        ani.save(gif_filename, writer='pillow') # or ani.save(filename, writer='ffmpeg') with filename = "ciao.mp4"


# import Class_site as site

# model = site.SitesPBCx(Npx=20, Npy=20)  # Example model, adjust as needed
# plot_honeycomb_cylinder(model)