# DESCRIPTION: core module for computing normal modes for WEBnma v3
# AUTHOR: dandan.xue@uib.no
# DATE: Feb, 2019
from __future__ import absolute_import
from os.path import join, basename

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh 
from scipy.spatial.distance import cdist

from utils.pdb import read_pdb
from utils.residue_mass import RES_MASS
from utils.modefiles import write_modefile
from config import MODE_NM


# Note for diagonalization:
# eigh: 
#     Solve an ordinary or generalized eigenvalue problem for a complex
#     Hermitian or real symmetric matrix.(use LAPACK?)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
#
# eigsh:
#     Find k eigenvalues and eigenvectors of the real symmetric square
#     matrix or complex hermitian matrix A.(use ARPACK)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html


# for konrad's model
CONST_A = 8.6e5
CONST_B = 2.39e5
CONST_C = 1.28e2
CONST_D = 0.4  # = 4 angstrom


def konrad_force_cons(diss):
    n = diss.shape[0]  
    ks = np.zeros(diss.shape)
    for i in range(n):
        for j in range(i):
            r = diss[i,j]
            if r < CONST_D:        
                ks[i,j] = CONST_A * r - CONST_B
            else:
                ks[i,j] = CONST_C * r**(-6)
            ks[j,i] = ks[i,j]
    return ks


def Hij(i,j,d,k):
    r_0 = j - i
    r_0 = r_0.reshape((1,3))
    hij = - k/(d**2) * (r_0 * r_0.transpose())
    return hij

            
def build_H(CAs, ks, diss, mass):
    n = CAs.shape[0]
    h = np.zeros((n,n,3,3))
    H = np.zeros((3*n, 3*n))

    for i in range(n):
        for j in range(i):
            h[i,j] = h[j,i] = Hij(CAs[i], CAs[j],diss[i,j], ks[i,j])

            H[i*3: i*3+3, j*3: j*3+3] = h[i,j] 
            H[j*3: j*3+3, i*3: i*3+3] = h[i,j]
                    
    for i in range(n):
        h[i,i] = - sum([h[i,j] for j in range(n) if j !=i ])
        H[i*3: i*3+3, i*3: i*3+3] = h[i,i]
    
    for i in range(n):
        for j in range(n):
            H[i*3: i*3+3, j*3 : j*3+3] = \
            H[i*3: i*3+3, j*3 : j*3+3] / mass[i] / mass[j]
    
    return H    


def calc_modes(CAs, mass, n=MODE_NM):
    diss = cdist(CAs, CAs)  # distance matrix of all C-Alpha atoms    
    ks = konrad_force_cons(diss)
    h = build_H(CAs, ks, diss, mass) # build Hessian matrix

    n = min(n, 3*len(CAs)-6)
    if n > 20:
        e, v = eigh(h)
        return e[:n+6],v[:,:n+6]
    else:
        return eigsh(h,k=n+6,which='SA')


def main(pdbfile, tar_dir='.', filename='modes.txt', mode_num=MODE_NM):
    PDB_ntuple = read_pdb(pdbfile, unit_nm=True)
    CAs = PDB_ntuple.ca_coords
    e,v = calc_modes(CAs, PDB_ntuple.weight, mode_num)

    modefile = join(tar_dir, filename)
    write_modefile(e,v,modefile, PDB_ntuple.residues_full)

    return modefile


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[1][:-4]+'_modes_v3.txt', sys.argv[3])
    
