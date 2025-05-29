import numpy as np
import matplotlib.pyplot as plt


class BaseSites:
    def __init__(self, Npx, Npy):
        #Npx number of plaquettes along x
        #Npy num of plaquettes along y
        self.Npx = Npx
        self.Npy = Npy
        # shared initialization

class SitesOBC(BaseSites):
    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)
        # OBC-specific initialization
        self.Nysites = self.Npy + 1
        self.Nxsites_1 = 2*self.Npx +1 #number of sites in the first and last row
        self.Nxsites_2 = 2*(self.Npx + 1) #number of sites in bulk rows
        self.Nsites = self.get_Nsites()
        self.ids = np.arange(self.Nsites)
        self.partition = self.get_partition()
        
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
            if idy % 2 == 0 and idy != self.Nysites - 1:  # even rows
                if idx % 2 == 0:
                    partition.append('A')
                else:
                    partition.append('B')
            elif idy % 2 == 1 and idy != self.Nysites - 1:  # odd rows
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
            elif idy == self.Nysites - 1:  # last row
                if idx % 2 == 0:
                    partition.append('B')
                else:
                    partition.append('A')
        partition = np.array(partition)
        return partition
    
    def get_Nsites(self):
        Nsites = 2*self.Nxsites_1
        if self.Npy > 2:
            Nsites = (self.Nysites - 2)*self.Nxsites_2 + Nsites
        return Nsites
        
    
    #define bond list for xx bonds, then for yy bonds and zz bonds, in this way we can then construct the Hamiltonian with fermionic operators!!
    def get_bonds(self):
        #returns a list of bonds
        xx_bondlist = []
        yy_bondlist = []
        zz_bondlist = []
        
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            if idy == 0: # first row
                print("first row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_1-1):
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        zz_bondlist.append([id, id+self.Nxsites_1])
                    else:
                        yy_bondlist.append([id, id+1])
                else:
                    zz_bondlist.append([id, id+self.Nxsites_1])
            elif idy % 2 == 1 and idy != (self.Nysites-1): # odd rows
                print("odd row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_2-1):
                    if idx % 2 == 1: 
                        #print("id", id, "idx", idx, "idy", idy)
                        xx_bondlist.append([id, id+1])
                        if idy != (self.Nysites-2):
                            zz_bondlist.append([id, id+self.Nxsites_2])
                    else:
                        yy_bondlist.append([id, id+1])
                else:
                    if idy != (self.Nysites-2):
                        zz_bondlist.append([id, id+self.Nxsites_2])
            elif idy % 2 == 0 and idy != (self.Nysites-1) and idy != 0: # even rows
                print("even row")
                print("id", id, "idx", idx, "idy", idy)
                if idx != (self.Nxsites_2-1):
                    if idx % 2 == 0: 
                        xx_bondlist.append([id, id+1])
                        if idy != (self.Nysites-2):
                            zz_bondlist.append([id, id+self.Nxsites_2])
                    else:
                        yy_bondlist.append([id, id+1])
            elif idy == (self.Nysites - 1): # last row
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
    

        # for idy in range(self.Nysites):
        #     if idy == 0 or idy % 2 ==  or idy == self.Nysites - 1:
        #         for idx in range(self.Nxsites_1 - 1):
        #             id = self.idxidy_to_id(idx, idy)
        #             if id < self.Nsites - 1:
        #                 bondlist.append((id, self.idxidy_to_id(idx + 1, idy)))
        #     for idx in range(0, self.Nxsites_2 - 1,2):
        #         id = self.idxidy_to_id(idx, idy)
        #         if id < self.Nsites - 1:
        #             bondlist.append((id, self.idxidy_to_id(idx + 2, idy)))
        # for id in range(self.Nsites):
        #     idx, idy = self.id_to_idxidy(id)
        #     if idx < self.Nxsites_2 - 1:
        #         bondlist.append((id, self.idxidy_to_id(idx + 1, idy)))
        # else:
        #     if idx < self.Nxsites - 1:
        #         bondlist.append((id, self.idxidy_to_id(idx + 1, idy)))
        
        return xx_bondlist, yy_bondlist, zz_bondlist
            
    """ def get_coordinates(self):
        coords = []
        a = 1  # lattice spacing
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            # shift even rows
            x = idx * a + 0.5 * a * (idy % 2)
            y = idy * (np.sqrt(3)/2) * a
            coords.append((x, y))
        return np.array(coords) """

    def get_coordinates(self):
        coords = []
        a = 1  # lattice spacing
        for id in self.ids:
            idx, idy = self.id_to_idxidy(id)
            print("id", id, "idx", idx, "idy", idy)
            # In un reticolo honeycomb, i siti sono su due sottoreticoli (A e B)
            if idy != self.Nysites - 1:  # not the last row
                if self.partition[id] == 'A':  # Sublattice A
                    print("sublattice A")
                    x = np.sqrt(3) * idx / 2.
                    print("x", x)
                    y = - 1.5 * idy
                    print("y", y)
                else:  # Sublattice B
                    print("sublattice B")
                    x = np.sqrt(3) * idx / 2.
                    y = - 1.5 * idy + 0.5
                print("x", x, "y", y)
            else:  # last row
                if self.Nysites % 2 == 0:
                    if self.partition[id] == 'A':
                        print("last row, sublattice A")
                        x = np.sqrt(3) * idx / 2.
                        y = - 1.5 * idy
                    else:
                        print("last row, sublattice B")
                        x = np.sqrt(3) * idx / 2.
                        y = - 1.5 * idy + 0.5
                else:
                    if self.partition[id] == 'A':
                        print("last row, sublattice A")
                        x = np.sqrt(3) * idx / 2. + np.sqrt(3) / 2.
                        y = - 1.5 * idy
                    else:
                        print("last row, sublattice B")
                        x = np.sqrt(3) * idx / 2. + np.sqrt(3) / 2.
                        y = - 1.5 * idy + 0.5
            coords.append((x, y))
        return np.array(coords)


class SitesPBC(BaseSites):
    # PBC only along x direction: cylindrical geometry
    def __init__(self, Npx, Npy):
        super().__init__(Npx, Npy)
        # PBC-specific initialization
        self.Nysites = 2*Npy
        self.Nxsites = 2*Npx
        self.Nsites = 2 * Npx * (Npy + 1) 
        self.ids = np.arange(self.Nsites)
        self.partition = np.where(self.ids % 2 == 0, 1, 0)

    def id_to_idxidy(self, id):
        #convert id to idx, idy
        idy = id // self.Nxsites
        idx = id % self.Nxsites
        return idx, idy
