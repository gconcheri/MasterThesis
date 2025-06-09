import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



class BaseSites:
    """
    Class to define the sites of a honeycomb lattice.
    This class is meant to be inherited by other classes that define specific boundary conditions.

    We consider a 2D honeycomb lattice with Npx plaquettes along the x direction and Npy plaquettes along the y direction.

    The sites are indexed from 0 to Nsites-1, where Nsites is the total number of sites in the lattice.

    The sites are divided into two sublattices A and B, which are defined by the partition attribute.

    The id_to_idxidy method converts a site id to its corresponding (idx, idy) coordinates,
    while the idxidy_to_id method converts (idx, idy) coordinates back to a site id.

    The honeycomb lattice Hamiltonian is given by: 
    H = - J_xx * Σ_{xlinks} σ_x^j σ_x^k - J_yy * Σ_{ylinks} σ_y^j σ_y^k - J_zz * Σ_{zlinks} σ_z^j σ_z^k
    where j and k are the indices of the sites connected by the links in the x, y, and z directions.

    The get_bonds method returns the list of bonds xx, yy, zz in the lattice, which can be used to construct the Hamiltonian.
    """

    def __init__(self, Npx, Npy):
        #Npx number of plaquettes along x
        #Npy num of plaquettes along y
        self.Npx = Npx
        self.Npy = Npy
        # shared initialization

class SitesOBC(BaseSites):
    """ The lattice has open boundary conditions (OBC) along both x and y directions. """

    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)
        # OBC-specific initialization
        self.Nyrows = self.Npy + 1
        self.Nxsites_1 = 2*self.Npx +1 #number of sites in the first and last row
        self.Nxsites_2 = 2*(self.Npx + 1) #number of sites in bulk rows
        self.Nsites = self.get_Nsites()
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        self.ids_A = [id for id in self.ids if self.partition[id] == 'A']
        self.ids_B = [id for id in self.ids if self.partition[id] == 'B']
        
    def id_to_idxidy(self, id):
        idy = (id + 1) // self.Nxsites_2
        if idy == 0:
            idx = id
        else:
            idx = (id +1) % self.Nxsites_2

            #probably other way to do this
            # #convert id to idx, idy
            # if id < self.Nxsites_1:
            #     #first row
            #     idx = id
            #     idy = 0
            # elif id < self.Nxsites_1 + self.Nxsites_2 * (self.Nsitesy - 2):
            #     #bulk rows
            #     idx = (id - self.Nxsites_1) % self.Nxsites_2
            #     idy = (id - self.Nxsites_1) // self.Nxsites_2 + 1
            # else:
            #     #last row
            #     idx = (id - self.Nxsites_1 - self.Nxsites_2 * (self.Nsitesy - 2)) % self.Nxsites_2
            #     idy = self.Nsitesy - 1

        return idx, idy
    
    def idxidy_to_id(self, idx, idy):
        #convert idx, idy to id
        if idy == 0 or idy == 1:
            return idx + self.Nxsites_1*idy
        else:
            return idx + self.Nxsites_1 + (idy-1) * self.Nxsites_2
    
    def get_partition(self):
        partition = []
        #tells you whether one site is part of sublattice A or B
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            if idy % 2 == 0 and idy != self.Nyrows - 1:  # even rows
                if idx % 2 == 0:
                    partition.append('A')
                else:
                    partition.append('B')
            elif idy % 2 == 1 and idy != self.Nyrows - 1:  # odd rows
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
            elif idy == self.Nyrows - 1:  # last row
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
        partition = np.array(partition)
        return partition
    
    def get_Nsites(self):
        Nsites = 2*self.Nxsites_1
        if self.Npy > 2:
            Nsites = (self.Nyrows - 2)*self.Nxsites_2 + Nsites
        return Nsites
    
    def get_Nxsite(self, id):
        idx, idy = self.id_to_idxidy(id)
        if idy == 0 or idy == self.Nyrows - 1:
            return self.Nxsites_1
        else:
            return self.Nxsites_2
        
    
    #define bond list for xx bonds, then for yy bonds and zz bonds, in this way we can then construct the Hamiltonian with fermionic operators!!
    def get_bonds(self):
        """
        we are defining these bonds according to the plaquette term convention:
        xx bonds: always from left to right
        zz bonds: always from top to bottom
        yy bonds: always from right to left

        Or in other words, the bonds always start from a site of sublattice A and end on a site of sublattice B.

        Another idea to define this function: take every site of sublattice A and add 
        bond to its right neighbour to yybond, to left neighbour to xxbond and to the site below it to zzbond.
        All of this simply using idx,idy coordinates.
        """

        #returns a list of bonds
        xx_bondlist = []
        yy_bondlist = []
        zz_bondlist = []
        
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            if idy == 0: # first row
                print("first row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_1-1): # last site in the first row
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        zz_bondlist.append([id, id+self.Nxsites_1])
                    else:
                        yy_bondlist.append([id, id+1])
                else:
                    zz_bondlist.append([id, id+self.Nxsites_1])
            elif idy % 2 == 1 and idy != (self.Nyrows-1): # odd rows
                print("odd row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_2-1):
                    if idx % 2 == 1: 
                        #print("id", id, "idx", idx, "idy", idy)
                        xx_bondlist.append([id, id+1])
                        if idy != (self.Nyrows-2):
                            zz_bondlist.append([id, id+self.Nxsites_2])
                    else:
                        yy_bondlist.append([id, id+1])
                else:
                    if idy != (self.Nyrows-2):
                        zz_bondlist.append([id, id+self.Nxsites_2])
            elif idy % 2 == 0 and idy != (self.Nyrows-1) and idy != 0: # even rows
                print("even row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_2-1):
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        if idy != (self.Nyrows-2):
                            zz_bondlist.append([id, id+self.Nxsites_2])
                    else:
                        yy_bondlist.append([id, id+1])
            elif idy == (self.Nyrows - 1): # last row
                if idx != (self.Nxsites_1-1):
                    print("last row")
                    print("id", id, "idx", idx, "idy", idy)
                    if idx % 2 == 1: 
                        xx_bondlist.append([id, id+1])
                    else:
                        yy_bondlist.append([id, id+1])
                        if self.Npy % 2 == 0: # even number of plaquettes along y
                            zz_bondlist.append([id-self.Nxsites_1, id])
                        else: # odd number of plaquettes along y
                            zz_bondlist.append([id-self.Nxsites_2, id])
                else:
                        if self.Npy % 2 == 0: # even number of plaquettes along y
                            zz_bondlist.append([id-self.Nxsites_1, id])
                        else: # odd number of plaquettes along y
                            zz_bondlist.append([id-self.Nxsites_2, id])
    
        
        yy_bondlist = [[b,a] for [a, b] in yy_bondlist]  # reverse the order of yy bonds to match the convention
        
        return xx_bondlist, yy_bondlist, zz_bondlist
    
    def get_diagonalbonds(self):
        """
        Returns a list of diagonal bonds in the honeycomb lattice.
        Diagonal bonds connect sites in different sublattices (A and B) that are diagonally adjacent.
        i.e. site from the top left corner of a plaquette to the bottom right corner of the next plaquette.
        """
        diag_bondlist = []

        for i, id in enumerate(self.ids_A):
            idx, idy = self.id_to_idxidy(id)
            if i % (self.Npx+1) != self.Npx: #if id is not the last site in sublattice A for that row
                if idy < self.Nyrows - 2:  # if id not in the last two rows
                    next_id = self.idxidy_to_id(idx + 2, idy + 1)
                    diag_bondlist.append([id, next_id])
                elif idy == self.Nyrows - 2:    # we are in the second last row
                    if self.Npy % 2 == 0:
                        next_id = self.idxidy_to_id(idx + 1, idy + 1)
                        diag_bondlist.append([id, next_id])
                    else:
                        next_id = self.idxidy_to_id(idx + 2, idy + 1)
                        diag_bondlist.append([id, next_id])
        return diag_bondlist
    
    def get_anyonbonds(self): 
        """
        Returns a list of bonds on which to fix u_ij=-1 in order to create anyon in the plaquette
        located at the plaquette coordinates: px = Npx//2, py = Npx//2, written in numb of
        horizontal and vertical plaquettes from the origin (top-left corner in lattice).
        """
        anyon_bondlist = []
        # px = self.id_to_idxidy(self.ids_A[self.Npx//2])[0]  # horizontal plaquette coordinate
        # py = self.Nyrows // 2

        # print("px", px, "py", py)

        # id_start = self.idxidy_to_id(px,py)

        index = len(self.ids_A)//2
        x, y = self.id_to_idxidy(self.ids_A[index])
        if x >= self.Nxsites_1//2:
            x = 0
            y +=1
            id = self.idxidy_to_id(x,y)
            index = np.argmin(np.abs(np.array(self.ids_A) - id))
    
        while x < self.Nxsites_1//2:
            index += 1
            x, _ = self.id_to_idxidy(self.ids_A[index])


        id_start = self.ids_A[index]
        idx_start, idy_start = self.id_to_idxidy(id_start)
        
        id = id_start
        idx = idx_start

        while idx < self.Nxsites_2-1:
            anyon_bondlist.append([id, id+1])
            id += 1
            idx, _ = self.id_to_idxidy(id)

        return anyon_bondlist, idx_start, idy_start
            
    # more efficient way to get coordinates

    def get_coordinates(self):
        coords = []
        a = 1  # lattice spacing
        offset_y = 0.5
        offset_x = np.sqrt(3) / 2.
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            print("id", id, "idx", idx, "idy", idy)
            x = np.sqrt(3) * idx / 2.
            y = - 1.5 * idy
            # In un reticolo honeycomb, i siti sono su due sottoreticoli (A e B)
            if idy != self.Nyrows - 1:  # not the last row
                if self.partition[id] == 'B':  # Sublattice B
                    y += offset_y
            else:  # last row
                if self.Nyrows % 2 == 0:
                    if self.partition[id] == 'B':
                        y += offset_y
                else:
                    x += offset_x
                    if self.partition[id] == 'B':
                        y += offset_y
                        
            print("x", x, "y", y)
            coords.append((x, y))
        return np.array(coords)


class SitesPBC(BaseSites):

    """ The lattice has periodic boundary conditions (PBC) along the x direction and open boundary conditions (OBC) along the y direction. """
    # PBC only along x direction: cylindrical geometry

    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)
        # PBC-specific initialization
        self.Nyrows = Npy + 1
        self.Nxsites = 2*Npx
        self.Nsites = 2 * Npx * (Npy + 1) 
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        self.ids_A = [id for id in self.ids if self.partition[id] == 'A']
        self.ids_B = [id for id in self.ids if self.partition[id] == 'B']

    def id_to_idxidy(self, id):
        #convert id to idx, idy
        idy = id // self.Nxsites
        idx = id % self.Nxsites
        return idx, idy
    
    def idxidy_to_id(self, idx, idy):
        #convert idx, idy to id
        return idx + idy * self.Nxsites
    
    def get_partition(self): #it's the same for PBC and OBC
        #tells you whether one site is part of sublattice A or B
        partition = []
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            if idy % 2 == 0 and idy != self.Nyrows - 1:  # even rows
                if idx % 2 == 0:
                    partition.append('A')
                else:
                    partition.append('B')
            elif idy % 2 == 1 and idy != self.Nyrows - 1:  # odd rows
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
            elif idy == self.Nyrows - 1:  # last row
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
        partition = np.array(partition)
        return partition
    
    def get_bonds(self):

        """
        we are defining these bonds according to the plaquette term convention:
        xx bonds: always from left to right
        zz bonds: always from top to bottom
        yy bonds: always from right to left

        Or in other words, the bonds always start from a site of sublattice A and end on a site of sublattice B.
        """

        xx_bondlist = []
        yy_bondlist = []
        zz_bondlist = []

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            next_id = self.idxidy_to_id((idx+1) % self.Nxsites, idy)
            if idy % 2 == 0 and idy != (self.Nyrows-1):
                if idx % 2 == 0:
                    xx_bondlist.append([id, next_id])
                    if idy != (self.Nyrows-2):
                        zz_bondlist.append([id, id+self.Nxsites])
                else:
                    yy_bondlist.append([id, next_id])
            elif idy % 2 == 1 and idy != (self.Nyrows-1):
                if idx % 2 == 1:
                    xx_bondlist.append([id, next_id])
                    if idy != (self.Nyrows-2):
                        zz_bondlist.append([id, id+self.Nxsites])
                else:
                    yy_bondlist.append([id, next_id])
            elif idy == (self.Nyrows - 1):
                if idx % 2 == 1:
                    xx_bondlist.append([id, next_id])
                else:
                    yy_bondlist.append([id, next_id])
                    if self.Npy % 2 == 0: # even number of plaquettes along y
                        zz_bondlist.append([id-self.Nxsites+1, id])
                    else: # odd number of plaquettes along y
                        zz_bondlist.append([id-self.Nxsites, id])      
        
        yy_bondlist = [[b,a] for [a, b] in yy_bondlist]  # reverse the order of yy bonds to match the convention              

        return xx_bondlist, yy_bondlist, zz_bondlist
    
    def get_diagonalbonds(self):
        """
        Returns a list of diagonal bonds in the honeycomb lattice.
        Diagonal bonds connect sites in different sublattices (A and B) that are diagonally adjacent.
        i.e. site from the top left corner of a plaquette to the bottom right corner of the next plaquette.
        """
        diag_bondlist = []

        for id in self.ids_A:  # iterate over every second site (only sites in A sublattice)
            idx, idy = self.id_to_idxidy(id)
            if idy < self.Nyrows - 2:  # we are not in the last and second last row
                next_id = self.idxidy_to_id((idx + 2) % self.Nxsites, idy + 1)
                diag_bondlist.append([id, next_id])
            elif idy == self.Nyrows - 2:  # we are in the second last row
                if self.Npy % 2 == 0:  # even number of plaquettes along y
                    next_id = self.idxidy_to_id((idx + 1) % self.Nxsites, idy + 1)
                    diag_bondlist.append([id, next_id])
                else:  # odd number of plaquettes along y
                    next_id = self.idxidy_to_id((idx + 2) % self.Nxsites, idy + 1)
                    diag_bondlist.append([id, next_id])
        return diag_bondlist

    def get_anyonbonds(self): 
        """
        Returns a list of bonds on which to fix u_ij=-1 in order to create anyon in the plaquette
        located at the plaquette coordinates: px = Npx//2, py = Npx//2, written in numb of
        horizontal and vertical plaquettes from the origin (top-left corner in lattice).
        """
        anyon_bondlist = []
        # px = self.id_to_idxidy(self.ids_A[self.Npx//2])[0]  # horizontal plaquette coordinate
        # py = self.Nyrows // 2

        # print("px", px, "py", py)

        # id_start = self.idxidy_to_id(px,py)

        index = len(self.ids_A)//2
        x, _ = self.id_to_idxidy(self.ids_A[index])
        while x < self.Nxsites//2:
            index += 1
            x, _ = self.id_to_idxidy(self.ids_A[index])

        id_start = self.ids_A[index]
        idx_start, idy_start = self.id_to_idxidy(id_start)

        id = id_start
        idx = idx_start

        while idx < self.Nxsites-1:
            anyon_bondlist.append([id, id+1])
            id += 1
            idx, _ = self.id_to_idxidy(id)

        return anyon_bondlist, idx_start, idy_start

    
    def get_coordinates(self):
        coords = []
        a = 1  # lattice spacing
        offset_y = 0.5
        offset_x = np.sqrt(3) / 2.
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)

            #print("id", id, "idx", idx, "idy", idy)
            x = np.sqrt(3) * idx / 2.
            y = - 1.5 * idy
            # In un reticolo honeycomb, i siti sono su due sottoreticoli (A e B)
            if idy != self.Nyrows - 1:  # not the last row
                if self.partition[id] == 'B':  # Sublattice B
                    y += offset_y
            else:  # last row
                if self.Nyrows % 2 == 0:
                    if self.partition[id] == 'B':
                        y += offset_y
                else:
                    x += offset_x
                    if self.partition[id] == 'B':
                        y += offset_y
            coords.append((x, y))
        return np.array(coords)
    

    def get_coordinates_cylindric(self, a = 1):
        """
        Returns 3D coordinates (x, y, z) for plotting on a cylinder of radius R.
        x: wraps around the cylinder (angle)
        y: height along the cylinder
        """
        coords = [] 
        offset_z = 0.5
        offset_theta = 2 * np.pi / self.Nxsites
        R = np.sqrt(3)*a*self.Nxsites/(4*np.pi)
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            # Map idx to angle theta
            theta = 2 * np.pi * idx / self.Nxsites
            # Map idy to height (z)
            z = -1.5 * idy
            # Standard honeycomb offset for sublattices

            if idy != self.Nyrows - 1:  # not the last row
                if self.partition[id] == 'B':  # Sublattice B
                    z += offset_z            
            else:
                if self.Nyrows % 2 == 0:
                    if self.partition[id] == 'B':
                        z += offset_z
                else:
                    theta += offset_theta
                    if self.partition[id] == 'B':
                        z += offset_z
            # Convert to Cartesian coordinates
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            coords.append((x, y, z))
        return np.array(coords)

    # def get_coordinates(self):
    #     coords = []
    #     a = 1  # lattice spacing
    #     for id in self.ids:
    #         idx, idy = self.id_to_idxidy(id)
    #         # In un reticolo honeycomb, i siti sono su due sottoreticoli (A e B)
    #         if idy != self.Nyrows - 1:
    #             if self.partition[id] == 'A':
    #                 x = np.sqrt(3) * idx / 2.
    #                 y = - 1.5 * idy
    #                 if idx == 0:

def plot_honeycomb(model, 
                    fig_size=(20,20), 
                    highlight_idxidy=None, indices=None, highlight_color='orange', 
                    plot_anyon_bonds=False, plot_diagonal_bonds = False):
    coords = model.get_coordinates()
    xx_bondlist, yy_bondlist, zz_bondlist = model.get_bonds()
    
    plt.figure(figsize=fig_size)
    # Plot sites
    plt.scatter(coords[:, 0], coords[:, 1], color='k', zorder=3)

    # Plot bonds
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
        anyon_bonds, px, py = model.get_anyonbonds()
        for i, j in anyon_bonds:
            plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color='magenta', lw=5, label='anyon bond' if (i, j) == anyon_bonds[0] else "", zorder=4)

    #highlight_idxidy = (px, py)
    # Highlight a specific site if requested
    if highlight_idxidy is not None:
        idx, idy = highlight_idxidy
        site_id = model.idxidy_to_id(idx, idy)
        plt.scatter(coords[site_id, 0], coords[site_id, 1], color=highlight_color, s=300, zorder=5, label='highlighted site')

    if indices is not None:
        for site_id in indices:
            plt.scatter(coords[site_id, 0], coords[site_id, 1], color=highlight_color, s=300, zorder=5, label='highlighted sites')


    if plot_diagonal_bonds:
        diag_bonds = model.get_diagonalbonds()
        for i, j in diag_bonds:
            plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                     color='green', lw=2, label='diagonal bond' if (i, j) == diag_bonds[0] else "", zorder=4)        
    

    plt.axis('equal')
    plt.axis('off')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation


def plot_honeycomb_cylinder(
    model,
    plot_static=True,
    make_gif=False,
    gif_filename="cylinder_rotation.gif",
    elev=20,
    azim=20,
    fig_size=(10, 30),
    frames=90,
    highlight_idxidy=None,
    indices=None,
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
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='k', s=10, zorder=3)
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
                        color='magenta', lw=5, zorder=4)
        # Highlight a specific site if requested
        if highlight_idxidy is not None:
            idx, idy = highlight_idxidy
            site_id = model.idxidy_to_id(idx, idy)
            ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                       color=highlight_color, s=100, zorder=5)
        # Highlight indices if requested
        if indices is not None:
            for site_id in indices:
                ax.scatter(coords[site_id, 0], coords[site_id, 1], coords[site_id, 2],
                           color=highlight_color, s=100, zorder=5)
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
            fig, update, frames=np.linspace(0, 360, frames), interval=50
        )
        ani.save(gif_filename, writer='pillow') # or ani.save(filename, writer='ffmpeg') with filename = "ciao.mp4"

# Example usage:
#modell = site.SitesPBC(Npx=20, Npy=20)
# plot_honeycomb_cylinder(modell, plot_static=True, make_gif=True)

model = SitesOBC(Npx=4, Npy=4)
anyonbonds = model.get_diagonalbonds()
print(anyonbonds)