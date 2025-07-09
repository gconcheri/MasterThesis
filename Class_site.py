import numpy as np

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

    def get_coordinates(self): #same for OBC and PBCx!
        coords = []
        a = 1  # lattice spacing
        offset_y = 0.5
        offset_x = np.sqrt(3) / 2.
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
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

        return idx, idy
    
    def idxidy_to_id(self, idx, idy):
        #convert idx, idy to id
        if idy == 0 or idy == 1:
            return idx + self.Nxsites_1*idy
        else:
            return idx + self.Nxsites_1 + (idy-1) * self.Nxsites_2
    

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
                if idx != (self.Nxsites_1-1): # last site in the first row
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        zz_bondlist.append([id, id+self.Nxsites_1])
                    else:
                        yy_bondlist.append([id, id+1])
                else:
                    zz_bondlist.append([id, id+self.Nxsites_1])
            elif idy % 2 == 1 and idy != (self.Nyrows-1): # odd rows
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
                if idx != (self.Nxsites_2-1):
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        if idy != (self.Nyrows-2):
                            zz_bondlist.append([id, id+self.Nxsites_2])
                    else:
                        yy_bondlist.append([id, id+1])
            elif idy == (self.Nyrows - 1): # last row
                if idx != (self.Nxsites_1-1):
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
        i.e. bonds connecting site from the top left corner of a plaquette to the bottom right corner of the same plaquette.
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
        located at the plaquette coordinates: (idx_start, idy_start) defined below.
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
    
    def get_plaquettecoordinates(self, id = None, idx = None, idy = None):
        """
        function that takes as input the central, lower site of a plaquette written as id in [0,..,self.Nsites]
        and returns the list of id coordinates of the sites of that plaquette"""
        if id is not None:
            idx, idy = self.id_to_idxidy(id)
        id_right = self.idxidy_to_id(idx+1, idy)
        id_left = self.idxidy_to_id(idx-1, idy)
        if idy == self.Nyrows-1 and self.Npy % 2==0:
            id_up = self.idxidy_to_id(idx+1, idy-1)
            id_upleft = self.idxidy_to_id(idx, idy-1)
            id_upright = self.idxidy_to_id(idx+2, idy-1)
        else:
            id_up = self.idxidy_to_id(idx, idy-1)
            id_upleft = self.idxidy_to_id(idx-1, idy-1)
            id_upright = self.idxidy_to_id(idx+ 1, idy-1)
        
        return [id, id_right, id_upright, id_up, id_upleft, id_left]

# class SitesProtBonds(BaseSites):


class SitesPBCx(BaseSites):

    """ The lattice has periodic boundary conditions (PBC) along the x direction and open boundary conditions (OBC) along the y direction. """
    # PBC only along x direction: cylindrical geometry

    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)
        # PBC-specific initialization
        self.Nyrows = Npy + 1
        self.Nxsites = 2*Npx
        self.Nsites = self.Nxsites * self.Nyrows
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

        # for id in self.ids:
        #     idx, idy = self.id_to_idxidy(id)
        #     next_id = self.idxidy_to_id((idx+1) % self.Nxsites, idy)
        #     if idy % 2 == 0 and idy != (self.Nyrows-1):
        #         if idx % 2 == 0:
        #             xx_bondlist.append([id, next_id])
        #             if idy != (self.Nyrows-2):
        #                 zz_bondlist.append([id, id+self.Nxsites])
        #         else:
        #             yy_bondlist.append([id, next_id])
        #     elif idy % 2 == 1 and idy != (self.Nyrows-1):
        #         if idx % 2 == 1:
        #             xx_bondlist.append([id, next_id])
        #             if idy != (self.Nyrows-2):
        #                 zz_bondlist.append([id, id+self.Nxsites])
        #         else:
        #             yy_bondlist.append([id, next_id])
        #     elif idy == (self.Nyrows - 1):
        #         if idx % 2 == 1:
        #             xx_bondlist.append([id, next_id])
        #         else:
        #             yy_bondlist.append([id, next_id])
        #             if self.Npy % 2 == 0: # even number of plaquettes along y
        #                 zz_bondlist.append([id-self.Nxsites+1, id])
        #             else: # odd number of plaquettes along y
        #                 zz_bondlist.append([id-self.Nxsites, id])

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            next_id = self.idxidy_to_id((idx+1) % self.Nxsites, idy)
            if id in self.ids_A:

                if idy != (self.Nyrows - 1):
                    id_down = self.idxidy_to_id(idx, idy+1)
                    if idy == (self.Nyrows-2) and self.Npy % 2 == 0: # even number of plaquettes along y
                        id_down = self.idxidy_to_id(idx-1, idy+1)
                    zz_bondlist.append([id, id_down])

                xx_bondlist.append([id, next_id])
            else:
                yy_bondlist.append([id, next_id])    
                    
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
    
    def get_plaquettecoordinates(self, id = None, idx = None, idy = None):
        """
        function that takes as input the central, lower site of a plaquette written as id in [0,..,self.Nsites]
        and returns the list of id coordinates of the sites of that plaquette"""
        if id is not None:
            idx, idy = self.id_to_idxidy(id)
        id_right = self.idxidy_to_id((idx+1) % self.Nxsites, idy)
        id_left = self.idxidy_to_id((idx-1) % self.Nxsites, idy)
        if idy == self.Nyrows-1 and self.Npy % 2==0:
            id_up = self.idxidy_to_id((idx+1) % self.Nxsites, idy-1)
            id_upleft = self.idxidy_to_id(idx, idy-1)
            id_upright = self.idxidy_to_id((idx+2)%self.Nxsites, idy-1)
        else:
            id_up = self.idxidy_to_id(idx, idy-1)
            id_upleft = self.idxidy_to_id((idx-1)%self.Nxsites, idy-1)
            id_upright = self.idxidy_to_id((idx+ 1)%self.Nxsites, idy-1)
        
        return [id, id_right, id_upright, id_up, id_upleft, id_left]


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

    def FTOperator(self):
        # define Fourier transform
        L = self.Npx
        ks = 2*np.pi * np.arange(-L//2, L//2)/L
        V = np.exp(1j * ks[None, :] * np.arange(L)[:, None]) / np.sqrt(L)
        V = np.kron(V, np.eye(2*self.Nyrows))
        V = self.reordering_operator().T @ V # @ self.reordering_operator()
        return V, ks 

    #define also translation operator
    
    def reordering_operator(self):
        Op = np.zeros((self.Nsites, self.Nsites))

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            index = 2*idy + (idx//2)*2*self.Nyrows + (idx % 2)
            Op[index,id] = 1.
        return Op

class SitesPBCxy(SitesPBCx):
    """ The lattice has periodic boundary conditions (PBC) along both the x and y direction. 
    With respect to PBCx we keep same id_to_idxidy, idxidy_to_id, get_partition and get_coordinates

    we change: get_bonds ...

    CONDITION: Npy must be EVEN!
    """
    # PBC both along x and y direction: torus geometry

    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)

        if self.Npy % 2 != 0:
            raise ValueError("Number of vertical plaquettes Npy must be EVEN for SitesPBCxy!")

        self.Nyrows = self.Npy #only different initialization w.r.t. PBCx case
        self.Nxsites = 2*Npx
        self.Nsites = self.Nxsites * self.Nyrows
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        self.ids_A = [id for id in self.ids if self.partition[id] == 'A']
        self.ids_B = [id for id in self.ids if self.partition[id] == 'B']

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

        # This time we use partition (sites labelled by A, B) to easily construct the bonds
        #we don't have to impose additional conditions as in the PBCx case because here we always have even Npy plaquettes

        for id in self.ids: 
            idx, idy = self.id_to_idxidy(id)
            next_id = self.idxidy_to_id((idx+1) % self.Nxsites, idy)
            if id in self.ids_A:
                id_down = self.idxidy_to_id(idx, (idy+1)%self.Npy)
                xx_bondlist.append([id, next_id])
                zz_bondlist.append([id, id_down])
            else:
                yy_bondlist.append([id, next_id])

        yy_bondlist = [[b,a] for [a, b] in yy_bondlist]  # reverse the order of yy bonds to match the convention              

        return xx_bondlist, yy_bondlist, zz_bondlist


    def FTOperator(self):
        # define Fourier transform
        L1 = self.Npx
        L2 = self.Npy

        ks1 = 2*np.pi * np.arange(-L1//2, L1//2)/L1
        ks2 = 2*np.pi * np.arange(-L2//2, L2//2)/L2

        V1 = np.exp(1j * ks1[None, :] * np.arange(L1)[:, None]) / np.sqrt(L1)
        V2 = np.exp(1j * ks2[None, :] * np.arange(L2)[:, None]) / np.sqrt(L2)

        V = np.kron(V2, V1)
        V = np.kron(V, np.eye(2))

        V = self.reordering_operator().T @ V # @ self.reordering_operator()

        return V, ks1, ks2

    def reordering_operator(self):
        Op = np.zeros((self.Nsites, self.Nsites))

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            id_new = self.idxidy_to_id((idx+idy)%self.Nxsites, idy) 
            Op[id_new,id] = 1.
        return Op
    
    def get_coordinates_torus(self, r_0 = 1, r_tilde = 1, z_0 = 0):
        """
        Returns 3D coordinates (x, y, z) for plotting on a cylinder of radius R.
        x: wraps around the cylinder (angle)
        y: height along the cylinder
        """
        coords = [] 
        offset_t = 2 * np.pi / (3 * self.Nyrows)
        offset_theta = 2 * np.pi / self.Nxsites

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            # Map idx to angle theta
            theta = 2 * np.pi * idx / self.Nxsites
            # Map idy to height (z)
            t = - 2 * np.pi * idy / self.Nyrows
            # Standard honeycomb offset for sublattices

            if idy != self.Nyrows - 1:  # not the last row
                if self.partition[id] == 'B':  # Sublattice B
                    t += offset_t
            else:
                if self.Nyrows % 2 == 0:
                    if self.partition[id] == 'B':
                        t += offset_t
                else:
                    theta += offset_theta
                    if self.partition[id] == 'B':
                        t += offset_t
            # Convert to Cartesian coordinates
            x = (r_0 + r_tilde*np.cos(t)) * np.cos(theta)
            y = (r_0 + r_tilde*np.cos(t)) * np.sin(theta)
            z = z_0 + r_tilde*np.sin(t)
            coords.append((x, y, z))
        return np.array(coords)


# Example usage:
#modell = site.SitesPBC(Npx=20, Npy=20)
# plot_honeycomb_cylinder(modell, plot_static=True, make_gif=True)

model = SitesPBCx(Npx=2, Npy=2)
reop = model.reordering_operator()
FT = model.FTOperator()