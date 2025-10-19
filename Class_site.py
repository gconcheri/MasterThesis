import numpy as np
from bisect import bisect_right

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

    def __init__(self, Npx, Npy, index = None, edge = False):
        self.index = index
        self.edge = edge
        super().__init__(Npx, Npy)
        # OBC-specific initialization
        self.Nyrows = self.Npy + 1
        self.Nxsites_1 = 2*self.Npx + 1 #number of sites in the first and last row
        self.Nxsites_2 = 2*(self.Npx + 1) #number of sites in bulk rows
        self.Nsites = self.get_Nsites()
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        self.ids_A = [id for id in self.ids if self.partition[id] == 'A']
        self.ids_B = [id for id in self.ids if self.partition[id] == 'B']  

    #%% site functions
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
        if self.Npy > 1:
            Nsites = (self.Nyrows - 2)*self.Nxsites_2 + Nsites
        return Nsites
    
    def get_Nxsite(self, id = None, idy = None):
        if idy is None:
            _, idy = self.id_to_idxidy(id)
        if idy == 0 or idy == self.Nyrows - 1:
            return self.Nxsites_1
        else:
            return self.Nxsites_2
        
    #%% bond functions
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
                        if self.Npy != 1:
                            if self.Npy % 2 == 0: # even number of plaquettes along y
                                zz_bondlist.append([id-self.Nxsites_1, id])
                            else: # odd number of plaquettes along y
                                zz_bondlist.append([id-self.Nxsites_2, id])
                else:
                        if self.Npy != 1:
                            if self.Npy % 2 == 0: # even number of plaquettes along y
                                zz_bondlist.append([id-self.Nxsites_1, id])
                            else: # odd number of plaquettes along y
                                zz_bondlist.append([id-self.Nxsites_2, id])
        
        yy_bondlist = [[b,a] for [a, b] in yy_bondlist]  # reverse the order of yy bonds to match the convention
        
        return xx_bondlist, yy_bondlist, zz_bondlist
    
    def get_diagonalbonds(self, links = True, edgepar = None):
        """
        Returns a list of diagonal bonds in the honeycomb lattice.
        Diagonal bonds in the bulk connect sites in different sublattices (A and B) that are diagonally adjacent.
        i.e. bonds connecting site from the top left corner of a plaquette to the bottom right corner of the same plaquette.

        If links is True, it also returns a list of links that connect the sites in the diagonal bonds.
        """


        diag_bondlist = []
        links_list = []


        for i, id in enumerate(self.ids_A):
            idx, idy = self.id_to_idxidy(id)
            if i % (self.Npx+1) != self.Npx: #if id is not the last site in sublattice A for that row
                if idy < self.Nyrows - 2:  # if id not in the last two rows
                    diag_id = self.idxidy_to_id(idx + 2, idy + 1)
                    down_id = diag_id -2
                    centre_id = diag_id-1
                    diag_bondlist.append([id, diag_id])
                    links_list.append([[id,down_id], [down_id, centre_id], [diag_id, centre_id]])
                elif idy == self.Nyrows - 2:    # we are in the second last row
                    if self.Npy % 2 == 0:
                        diag_id = self.idxidy_to_id(idx + 1, idy + 1)
                        down_id = diag_id -2
                        centre_id = diag_id-1
                        diag_bondlist.append([id, diag_id])
                        links_list.append([[id,down_id], [down_id, centre_id], [diag_id, centre_id]])
                    else:
                        diag_id = self.idxidy_to_id(idx + 2, idy + 1)
                        down_id = diag_id -2
                        centre_id = diag_id-1
                        diag_bondlist.append([id, diag_id])
                        links_list.append([[id,down_id], [down_id, centre_id], [diag_id, centre_id]])

        if edgepar == None:
            edgepar = self.edge

        if edgepar:
            if self.Npy != self.Npx or self.Npy % 2 == 0 or self.Npy == 1:  
                raise AssertionError("Npx and Npy must both be equal, odd and greater than 1")
            # Add diagonal bonds for the last row
            for id in self.ids_B[:self.Npx-2:2]:
                diag_bondlist.append([id, id+2])
                links_list.append([[id+1,id], [id+1, id+2]])
            
            id = self.ids_B[self.Npx-1]
            diag_bondlist.append([id, id+1])
            links_list.append([[id, id+1]])


            for y in range(self.Npy)[1::2]:
                id = self.idxidy_to_id(0,y)
                id_down = self.idxidy_to_id(1,y+1)
                diag_bondlist.append([id, id_down])
                links_list.append([[id+1, id], [id+1,id_down]])

                x = self.Nxsites_2 - 1
                id = self.idxidy_to_id(x, y)
                id_down = self.idxidy_to_id(x-1, y+1)
                diag_bondlist.append([id, id_down])
                links_list.append([[id, id_down+1], [id_down,id_down+1]])
            
            for id in self.ids_A[-self.Npx+1::2]:
                diag_bondlist.append([id, id+2])
                links_list.append([[id,id+1], [id+2, id+1]])

            id = self.ids_B[-self.Npx-1]
            diag_bondlist.append([id, id+1])
            links_list.append([[id+1, id]])
        
        if links:
            self.links_list = links_list
        
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

        #I should improve this eventually 
        #Approach to find the index of the bottom site of the most central plaquette, 
        # given whatever initial model of Npx and Npy plaquettes
        index = self.index
        if index is None:
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

        if self.Npy > 1 and idy_start < self.Nyrows - 1:
            while idx < self.Nxsites_2-1:
                anyon_bondlist.append([id, id+1])
                id += 1
                idx, _ = self.id_to_idxidy(id)
        else:
            while idx < self.Nxsites_1-1:
                anyon_bondlist.append([id, id+1])
                id += 1
                idx, _ = self.id_to_idxidy(id)

        return anyon_bondlist, id_start, idx_start, idy_start
    
    def get_plaquettecoordinates(self, id = None, idx = None, idy = None):
        """
        function that takes as input the central, lower site of a plaquette written as id in [0,..,self.Nsites]
        and returns the list of id coordinates of the sites of that plaquette"""
        if id is not None:
            idx, idy = self.id_to_idxidy(id)
        else: 
            id = self.idxidy_to_id(idx,idy)
        id_right = self.idxidy_to_id(idx+1, idy)
        id_left = self.idxidy_to_id(idx-1, idy)
        if idy == self.Nyrows-1 and self.Npy % 2 == 0:
            id_up = self.idxidy_to_id(idx+1, idy-1)
            id_upleft = self.idxidy_to_id(idx, idy-1)
            id_upright = self.idxidy_to_id(idx+2, idy-1)
        else:
            id_up = self.idxidy_to_id(idx, idy-1)
            id_upleft = self.idxidy_to_id(idx-1, idy-1)
            id_upright = self.idxidy_to_id(idx+ 1, idy-1)
        
        return [id, id_right, id_upright, id_up, id_upleft, id_left]
    
    #%% loop functions
    def get_loop_parallelogram(self):
        """
        Returns:
        1. list of indeces corresponding to sites of c Majorana operators corresponding to certain parallelogram loop
        2. list of xx, yy, zz bonds of links of ujk present in loop
        3. the sign in front of majorana string when calculating the expectation value with FGS Wick Theorem 

        I have to find a way to write for now largest parallelogram loop, by considering geometry of honeycomb lattice and then 
        based on that I simply take all indeces in bulk rows except for idx = 0 and idx = self.Nxsites, or something like that

        """
        if self.Npy < 2 or self.Npx <2:
            raise AssertionError("Npx and Npy must both be greater than 1")
        
        M = self.Npx - (self.Npy+1)//2 #in this way if Npy even: I get Npy/2, if odd I get (Npy+1)/2

        # xx_bonds, yy_bonds, zz_bonds = self.get_bonds()
        indeces_list = []
        links_list = []

        prefactor = self.get_prefactor_parallelogram()
    
        for y in range(1,self.Nyrows):
            x_0 = y
            if y == self.Nyrows-1 and self.Npy % 2 == 0:
                x_0 = y-1

            for x in range(M):
                id = self.idxidy_to_id(x_0+2*x,y)
                id_n = id+1
                id_nn = id+2
                id_up = self.idxidy_to_id(x_0+2*x+1, y-1)

                if y == 1:
                    indeces_list.append(id)
                    links_list.append([id, id_n]) #xbonds
                    links_list.append([id_nn, id_n]) #ybonds

                elif y>1 and y<self.Nyrows-1: 
                    indeces_list.append(id)
                    indeces_list.append(id_n)
                    links_list.append([id, id_n]) #xbonds
                    links_list.append([id_nn, id_n]) #ybonds
                    links_list.append([id_up,id_n]) #zbonds
                
                else:
                    if self.Npy % 2 == 0:
                        id_up = self.idxidy_to_id(x_0+2*x+2, y-1)

                    indeces_list.append(id_n)
                    links_list.append([id_up,id_n]) #zbonds

        final_x = self.id_to_idxidy(self.ids[-2])[0]

        plaquette_coords = [[1,1], [2*M+1,1], [final_x, self.Npy], [final_x-2*M, self.Npy]]
        plaquette_indices = []

        for a in plaquette_coords:
            plaquette_indices.append(self.get_plaquettecoordinates(idx = a[0], idy = a[1]))
        
        
        return prefactor, indeces_list, links_list, plaquette_indices

    def get_prefactor_parallelogram(self):
        M = self.Npx - (self.Npy+1)//2 #in this way if Npy even: I get Npy/2, if odd I get (Npy+1)/2
        # print("M: ", M)
        A = 0
        for i in range(M):
            A += i 
        # print("A ", A)
        B = M**2
        # print("B ", B)
        C = (self.Nyrows-3)*B
        # print("C ", C)
        Np = M*(self.Npy-1) #number of total highlighted plaquettes, i.e. total elementary loops required to form big loop
        # print("Np ", Np)
        D = 0
        if Np%4 == 0:
            D = 1
        elif Np % 4 == 1:
            D = 1j
        elif Np % 4 == 2:
            D = -1
        else: 
            D = -1j
        # print("D ", D)
        
        return D*(-1)**(A+C)

    def get_small_loop(self):
        #smallest loop, only around upper left plaquette
        indeces_list = self.get_diagonalbonds()[0]
        id = indeces_list[0]
        id_down = indeces_list[1]
        id_n = id +1 
        id_nn = id + 2

        links_list = [[id, id_n], [id_nn, id_n], [id_nn, id_down]]

        prefactor = 1j 

        return prefactor, indeces_list, links_list
    
    def get_general_loop(self, plaquette_list = None):
        """
        Takes as input: list of odd length of numbers, each corresponding to # of plaquettes that form loop in that specific row. 
        i.e. [4,2,4] -> central number = 2 is number of plaquettes in the central row

        For now we focus on model where Npx, Npy are odd!

        OBS: The loop is always centered around the central plaquette

        Returns:
        1. list of indeces corresponding to sites of c Majorana operators corresponding to certain loop
        2. list of xx, yy, zz bonds of links of ujk present in loop
        3. the sign in front of majorana string when calculating the expectation value with FGS Wick Theorem 

        """

        length = [self.Npx-1, self.Npx-2]
        max_plaquette_list = [length[i%2] for i in range(self.Npy-1)]

        if plaquette_list is not None:
            for element1, element2 in zip(plaquette_list, max_plaquette_list):
                if element1 > element2:
                    raise AssertionError("Each element of plaquette_list must be less than or equal to corresponding element in max_plaquette_list: ", max_plaquette_list)

        if plaquette_list is None:
            plaquette_list = max_plaquette_list


        # if len(plaquette_list) % 2 == 0:
        #     raise AssertionError("plaquette_list must be of odd length")

        indeces_list = self.get_diagonalbonds(edgepar=None)

        indeces_list_new = []

        for j in range(self.Npy):
            llist_1 = []
            for i in range(self.Npx):
                llist_1.append(indeces_list[i+j*self.Npx])
            indeces_list_new.append(llist_1)

        indeces_loop_list = []

        mid_row_idx = self.Npy//2 #middle row index in indeces_list_new
        mid_plaq_idx = (self.Npx-1)//2 #middle plaquette index in each row of indeces_list_new
        loop_half_length = (len(plaquette_list)-1)//2
        loop_start_row = mid_row_idx - loop_half_length
    
        for i, element in enumerate(plaquette_list):
            if self.Npx % 2 == 0:
                indeces_loop_list.append(indeces_list_new[loop_start_row+i][mid_plaq_idx-(element-1)//2:mid_plaq_idx+(element+2)//2])
            else:
                indeces_loop_list.append(indeces_list_new[loop_start_row+i][mid_plaq_idx-(element//2):mid_plaq_idx+(element+1)//2])

        indeces_loop_list_flattened = [item for sublist in indeces_loop_list for item in sublist]

        links_list = []

        for id, id_diag in indeces_loop_list_flattened:
            id_next = id + 1
            id_nextnext = id + 2
            links_list.append([id, id_next]) #xbonds
            links_list.append([id_nextnext, id_next]) #ybonds
            links_list.append([id_diag, id_nextnext]) #zbonds
        
        indeces_loop_list_flattened = [site for bond in indeces_loop_list_flattened for site in bond]

        #total number of plaquettes in the loop
        Np = sum(plaquette_list)

        #base prefactor due to number of plaquettes
        base_prefactor = 1j**Np  
        #prefactor due to ordering of Majorana operators, following swapping anticommutation relations
        _, prefactor = self.compute_loop_prefactor(indeces_loop_list)

        prefactor = prefactor * base_prefactor

        return prefactor, indeces_loop_list_flattened, links_list
 
    def compute_loop_prefactor(self, indeces_loop_list):
        """Compute the integer and a complex phase prefactor based on ordering rules.

        Parameters
        ----------
        indeces_loop_list : list[list[list[int,int]]]
            Structure like: [ row0, row1, ... ] where each row is a list of bonds [id, diag_id] in respective row.


        Rule: for each id (first element of each [id, diag_id] pair)
        starting from the SECOND sublist (row index 1):
          1. Count how many diag_id in the PREVIOUS sublist/row have x-coordinate STRICTLY greater than x(id), x(diag_id_prev) > x(id_current) →  prev_count 
          2. Count how many diag_id in the CURRENT sublist have x-coordinate <= x(id), which is simply the index k of the bond in that row → same_row_count = k
        Add both counts to an accumulator: total += prev_count + same_row_count
        Repeat for every id in every sublist from index 1 onward.

        We return:
          - total_count: the accumulated integer
          - phase: 1j**total_count (example complex phase if you want to reuse same convention)

        """
        if not indeces_loop_list:
            return 0, 1+0j

        x_cache = {}
        def x_of(site_id):
            if site_id not in x_cache:
                x_cache[site_id] = self.id_to_idxidy(site_id)[0]
            return x_cache[site_id]

        # Pre-sort previous-row diag_id x-coordinates
        diag_x_rows = []
        for row in indeces_loop_list:
            diag_x = [x_of(bond[1]) for bond in row]
            diag_x.sort()
            diag_x_rows.append(diag_x)

        total = 0

        for j in range(len(indeces_loop_list[0])): #length of first row = M
            #I am calculating the A parameter here
            total += j  # j is index in the first row

        for r in range(1, len(indeces_loop_list)): #iteration to calculate B, C... parameter
            prev_diag_sorted = diag_x_rows[r-1]
            len_prev = len(prev_diag_sorted)
            for k, bond in enumerate(indeces_loop_list[r]):
                id_site = bond[0]
                x_id = x_of(id_site)
                # prev greater than x_id
                pos_prev = bisect_right(prev_diag_sorted, x_id)
                greater_prev = len_prev - pos_prev
                # same row contribution = k
                same_row_count = k
                total += greater_prev + same_row_count

        prefactor = (-1) ** total # (-1)^(A+B+C...)

        return total, prefactor
     
    def get_loop(self, plaquette_list = None, type = 'general'):
        if type == 'parallelogram':
            prefactor, indeces_list, links_list, plaquette_indices = self.get_loop_parallelogram()
            return prefactor, indeces_list, links_list, plaquette_indices
        elif type == 'small':
            prefactor, indeces_list, links_list = self.get_small_loop()
            return prefactor, indeces_list, links_list
        elif type == 'general':
            prefactor, indeces_list, links_list = self.get_general_loop(plaquette_list = plaquette_list)
            return prefactor, indeces_list, links_list

    #%% other functions

    #sistema questa funzione per far si che divida il reticolo in due metà esatte
    def get_sites_halfpartition(self):
        """
        Returns a list of the site indices in systems A x<x_c and B x>=x_c given by cutting lattice in half in x direction.
        """
    
        halflattice_list_A = []

        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            if idy == 0 or idy == self.Nyrows - 1:
                if idx < self.Nxsites_1/2:
                    halflattice_list_A.append(id)
            else:
                if idx < self.Nxsites_2/2:
                    halflattice_list_A.append(id)

        lattice_list = self.ids.tolist()

        halflattice_list_B = [x for x in lattice_list if x not in halflattice_list_A]

        return halflattice_list_A, halflattice_list_B
    
    #usa metodo di general loop
    def get_sites_bulkpartition(self, plaquette_list = None):
        list_A = self.get_general_loop(plaquette_list = plaquette_list)[1]
        lattice_list = self.ids.tolist()
        list_B = [x for x in lattice_list if x not in list_A]

        return list_A, list_B



    def get_current_sites(self):
        if self.Npy != self.Npx or self.Npy % 2 == 0 or self.Npy == 1:
            raise AssertionError("Npx and Npy must both be equal, odd and greater than 1")

        # Returns a list of the site indices to use in the calculation of the current

        current_list = []

        y = self.Nyrows/2
        if y % 2 == 1:
            x = self.Nxsites_2/2
        else:
            x = self.Nxsites_2/2 + 1

        while x < self.Nxsites_2:
            current_list.append(int(self.idxidy_to_id(x,y)))
            x += 2
        
        return current_list
    


class SitesProtBonds(BaseSites):
    """ The lattice has open boundary conditions (OBC) along both x and y directions, with protruding bonds. """

    def __init__(self, Npx, Npy, index = None, edge = True):

        self.index = index
        self.edge = edge  # whether to include edge diagonal bonds in the model
        super().__init__(Npx, Npy)
        
        #Take values from old class which I will need to define same values for new class:
        self.obc_model = SitesOBC(Npx,Npy, index = index, edge = False)

        # Protruding bonds specific initialization

        self.Nxsites_0 = self.Npx  # number of sites in the first and last row
        self.Nxsites_1 = 2*self.Npx + 1 #number of sites in the first and last row
        self.Nxsites_2 = 2*(self.Npx + 1) #number of sites in bulk rows
        self.Nyrows = self.Npy + 3  # number of rows in the lattice
        self.Nsites = self.get_Nsites()
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        self.ids_A = [id for id in self.ids if self.partition[id] == 'A']
        self.ids_B = [id for id in self.ids if self.partition[id] == 'B']

    def get_Nsites(self):
        Nsites_OBC = self.obc_model.get_Nsites()
        Nsites = Nsites_OBC
        # if self.Npx < 2:
        #     raise AssertionError("Npx must be greater than 1")
        return Nsites + 2 * self.Npx
    
    def get_partition(self):
        partition_OBC = self.obc_model.get_partition()
        partition = partition_OBC.tolist()

        for _ in range(self.Npx):
            partition.insert(0, 'A')
            partition.append('B')
        
        return np.array(partition)

    
    def id_to_idxidy(self, id):
        if id < self.Npx: #id is one of upper protruding sites
            idy = 0
            idx = 2*id+1
        elif id < self.Npx+self.Nxsites_1:
            idy = 1
            idx = id - self.Npx
        elif id < self.Nsites - self.Npx: 
            idy = (id - self.Npx + 1) // self.Nxsites_2 + 1
            idx = (id - self.Npx +1) % self.Nxsites_2
        else:
            idy = self.Nyrows-1
            idx = 2*(id - self.Npx - self.Nxsites_1 +1) % self.Nxsites_2 +1

        return idx, idy
    
    def idxidy_to_id(self, idx, idy):
        #convert idx, idy to id
        if idy == 0:
            return idx//2
        elif idy==1 or idy==2:
            return idx + self.Nxsites_1*(idy-1) + self.Npx
        elif idy>2 and idy < self.Nyrows-1:
            return self.Npx + self.Nxsites_1 + (idy-2) * self.Nxsites_2 + idx
        else:
            return self.Npx + self.Nxsites_1 + (idy-3) * self.Nxsites_2 + self.Nxsites_1 + idx//2
    
    def get_bonds(self):
        # Recupera i bond originali
        xx_bonds = []
        yy_bonds = []
        zz_bonds = []
        
        # Prendi i bond OBC standard
        xx_bonds_OBC, yy_bonds_OBC, zz_bonds_OBC = self.obc_model.get_bonds()
        # Aggiungi i bond originali con la traslazione
        xx_bonds = self.OBClist_translation(xx_bonds_OBC)
        yy_bonds = self.OBClist_translation(yy_bonds_OBC)
        
        #zz_bonds
        for id in range(self.Npx):
            x, y = self.id_to_idxidy(id)
            id_down = self.idxidy_to_id(x,y+1)
            zz_bonds.append([id, id_down])
    
        zz_bonds = zz_bonds + self.OBClist_translation(zz_bonds_OBC)
    
        for id in self.ids[-1:-1-self.Npx:-1]:
            x, y = self.id_to_idxidy(id)
            id_up = self.idxidy_to_id(x,y-1)
            zz_bonds.append([id_up, id])

        return xx_bonds, yy_bonds, zz_bonds
    
    def get_anyonbonds(self):
        anyonbonds_OBC = self.obc_model.get_anyonbonds()
        return self.OBClist_translation(anyonbonds_OBC[0]), anyonbonds_OBC[1], anyonbonds_OBC[2], anyonbonds_OBC[3]
    
    def get_diagonalbonds(self, cov = True, edgepar = None):
        diagonalbonds_OBC = self.obc_model.get_diagonalbonds()
        diagonalbonds = []
        #cov_value specifies values of covariance matrix cov[i,j] given diagbond [i,j] 
        # in order for it to coincide with E.D. psi_e state defined in ed_vs_fgs 
        cov_value = []
        
        diagonalbonds = diagonalbonds + self.OBClist_translation(diagonalbonds_OBC)
        cov_value = [1 for _ in diagonalbonds_OBC]

        if edgepar == None:
            edgepar = self.edge

        if edgepar:
            for id in range(self.Npx):
                x, y = self.id_to_idxidy(id)
                id_down = self.idxidy_to_id(x,y+1)
                diagonalbonds.append([id, id_down])
                cov_value.append(-1)
                
            id = self.ids[-1-self.Npx+1]
            idx, idy = self.id_to_idxidy(id)
            id_upleft = self.idxidy_to_id(idx-1, idy-1)
            diagonalbonds.append([id_upleft, id])
            cov_value.append(-1)

            id = self.ids[-1]
            idx, idy = self.id_to_idxidy(id)
            id_down = self.idxidy_to_id(idx, idy-1)
            if self.Npy % 2 == 0:
                id_upright = self.idxidy_to_id(idx+2, idy-2)
            else:
                id_upright = self.idxidy_to_id(idx+1, idy-2)
            diagonalbonds.append([id_upright, id_down])
            cov_value.append(1)

            for id in self.ids[-1-self.Npx+2:]:
                idx, idy = self.id_to_idxidy(id)
                id_upleft = self.idxidy_to_id(idx-2, idy-1)
                diagonalbonds.append([id_upleft, id])
                cov_value.append(1)
    
        
        if cov:
            self.cov_value = cov_value

        return diagonalbonds
    
    def get_plaquettecoordinates(self, id=None, idx=None, idy=None):

        return self.OBClist_translation(self.obc_model.get_plaquettecoordinates(id = id, idx = idx, idy = idy))
    
    def get_loop_parallelogram(self):
        prefactor, indeces_list, links_list, plaquette_indices = self.obc_model.get_loop_parallelogram()
        indeces_list = self.OBClist_translation(indeces_list)
        links_list = self.OBClist_translation(links_list)
        plaquette_indices_new = []
        for p in plaquette_indices:
            p = self.OBClist_translation(p)
            plaquette_indices_new.append(p)
        return prefactor, indeces_list, links_list, plaquette_indices_new
    
    def get_small_loop(self):
        prefactor, indeces_list, links_list = self.obc_model.get_small_loop()
        indeces_list = self.OBClist_translation(indeces_list)
        links_list = self.OBClist_translation(links_list)        
        return prefactor, indeces_list, links_list
    
    def get_general_loop(self, plaquette_list = None):
        prefactor, indeces_list, links_list = self.obc_model.get_general_loop(plaquette_list = plaquette_list)
        indeces_list = self.OBClist_translation(indeces_list)
        links_list = self.OBClist_translation(links_list)        
        return prefactor, indeces_list, links_list
    
    def get_loop(self, plaquette_list = None, type = 'general'):
        if type == 'parallelogram':
            prefactor, indeces_list, links_list, plaquette_indices = self.get_loop_parallelogram()
            return prefactor, indeces_list, links_list, plaquette_indices
        elif type == 'small':
            prefactor, indeces_list, links_list = self.get_small_loop()
            return prefactor, indeces_list, links_list
        elif type == 'general':
            prefactor, indeces_list, links_list = self.get_general_loop(plaquette_list = plaquette_list)
            return prefactor, indeces_list, links_list
        
    
    def OBClist_translation(self, lst):
        if isinstance(lst[0], list) or isinstance(lst[0], tuple):
            # List of pairs: [[i, j], ...]
            return [[i + self.Npx, j + self.Npx] for i, j in lst]
        else:
            # Flat list: [i, j, ...]
            return [i + self.Npx for i in lst]
            

    def get_coordinates(self):
        coords = []
        #Prendo valori coordinate nel caso OBC
        coords_OBC = self.obc_model.get_coordinates()
        coords_OBC = coords_OBC.tolist()

        offset_y = 0.5
        offset_x = np.sqrt(3) / 2.


        #aggiungi coordinate nuove!
        for id in range(self.Npx):
            idx, _ = self.id_to_idxidy(id)
            x = np.sqrt(3) * idx / 2.
            y = + 1 + offset_y
            coords.append((x,y))

        coords = coords + coords_OBC
        
        for id in self.ids[-1-self.Npx+1:]:
            idx, _ = self.id_to_idxidy(id)
            y = -1.5*(self.Nyrows-3)-2*offset_y
            x = np.sqrt(3) * idx / 2.
            if self.Npy % 2 == 0:
                x += offset_x
            coords.append((x,y))
        
        return np.array(coords)



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
        else: 
            id = self.idxidy_to_id(idx,idy)
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
# model = SitesProtBonds(Npx=2, Npy=3)
# model.get_coordinates()


# model = SitesPBCx(Npx=2, Npy=2)
# reop = model.reordering_operator()
# FT = model.FTOperator()