import numpy as np
#OPEN BOUNDARY CONDITIONS
class sites:
    def __init__(self, Npx, Npy, OBC = True):
        #Npx number of plaquettes along x
        #Npy num of plaquettes along y
        self.Npx = Npx
        self.Npy = Npy
        self.OBC = OBC
        if OBC:
            
            self.Nysites = 2*Npy + 1
            self.Nxsites_1 = 2*self.Npx +1 #number of sites in the first and last row
            self.Nxsites_2 = 2*(self.Npx + 1) #number of sites in bulk rows
            self.Nsites = 3 * Npx * Npy + 4 
            self.ids = np.arange(self.Nsites)
            self.partition = np.where(self.ids % 2 == 0, 1, 0)
        else:
            self.Nysites = 2*Npy
            self.Nxsites = 2*Npx
            self.Nsites = 2 * Npx * Npy
            self.ids = np.arange(self.Nsites)
            self.partition = np.where(self.ids % 2 == 0, 1, 0)            
    def id_to_idxidy(self, id):
        if self.OBC:
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
        else:
            #convert id to idx, idy
            idy = id // self.Nxsites
            idx = id % self.Nxsites
            "Check this, it might be wrong"

        return idx, idy
    
    def idxidy_to_id(self, idx, idy):
        #convert idx, idy to id
        if self.OBC:
            if idy == 0 or idy == 1:
                return idx + self.Nxsites_1*idy
            else:
                return idx + self.Nxsites_1 + (idy-1) * self.Nxsites_2
    
    def get_partition(self,id):
        #tells you whether one site is part of sublattice A or B
        return self.partition[id]
    
    #define bond list for xx bonds, then for yy bonds and zz bonds, in this way we can then construct the Hamiltonian with fermionic operators!!
    def xx_bondlist(self):
        #returns a list of xx bonds
        bondlist = []
        for id in range(self.Nsites):
            idx, idy = self.id_to_idxidy(id)
            if self.OBC:
                if idx < self.Nxsites_2 - 1:
                    bondlist.append((id, self.idxidy_to_id(idx + 1, idy)))
            else:
                if idx < self.Nxsites - 1:
                    bondlist.append((id, self.idxidy_to_id(idx + 1, idy)))
        return bondlist
            