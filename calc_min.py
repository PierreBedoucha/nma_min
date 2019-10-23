from __future__ import absolute_import
from utils.pdb import read_pdb, rewrite_pdb
from calc_nm import calc_modes, build_H, konrad_force_cons
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
import os


# def build_g(CAs, CAs_ref, H):
#     n = H.shape[0]
#     g = np.zeros((n, n))
#     r = CAs - CAs_ref
#     rT = r.transpose()
#     rT = np.ravel(rT)
#     # dissT = diss.transpose()
#     # x, y = dissT.shape
#     # m, n = x * 3, y * 3
#     # dissT = np.broadcast_to(
#     #     dissT.reshape(x, 1, y, 1), (x, 3, y, 3)
#     # ).reshape(m, n)
#     # for i in range(n):
#     #     for j in range(n):
#     #         g[i,j] = H[i,j] * dissT[i, j]
#     g = rT @ H
#     return g

def build_g(CAs, CAs_ref, ks):
    r = CAs - CAs_ref
    g = ks @ r
    g_ravel = np.ravel(g)
    return g_ravel

# def build_gProj(g, v, mass):
#     n = v.shape[0]
#     m = v.shape[1]
#     gProj = np.zeros((n, m - 6))
#     # gProj = (g @ v) / np.linalg.norm(v)
#     mass = np.repeat(mass, 3)
#     for j in range(6, m, 1):
#         # gProj[:, j - 6] = (g * v[:, j]) / np.linalg.norm(v[:, j])
#         gProj[:, j - 6] = (np.dot(g, v[:, j]) / np.linalg.norm(v[:, j])) * mass
#     return gProj

def build_gProj(g, v, mass):
    n = v.shape[0]
    m = v.shape[1]
    gProj = np.zeros((n, m - 6))
    mass = np.repeat(mass, 3)
    g = _mass_weight_normalize_vector(g, mass)
    for j in range(6, m, 1):
        mode = _mass_weight_normalize_vector(v[:, j], mass)
        gProj[:, j - 6] = massWeightedDotProduct(mode, g, mass)
    return gProj

def massWeightedDotProduct(array, other, mass):
    return np.add.reduce(np.ravel(array * other * mass))

def _mass_weight_normalize_vector(value, mass):
    return value / np.sqrt(massWeightedDotProduct(value, value, mass))

def build_hNR(gProj, v, e):
    # gProjVe = (gProj @ v) / e
    n = v.shape[0]
    m = v.shape[1]
    gProjVe = np.zeros((n, m - 6))
    for j in range(6, m, 1):
        gProjVe[:, j - 6] = (gProj[:, j - 6] * v[:, j]) / e[j]
    gProjVeSum = -1 * np.sum(gProjVe, axis=1)
    return gProjVeSum

def project_struct(ref_pdb, pdbfile, CAs_A, hNR):
    x_mask = [True, False, False] * CAs_A.shape[0]
    y_mask = [False, True, False] * CAs_A.shape[0]
    z_mask = [False, False, True] * CAs_A.shape[0]

    x_p = hNR[x_mask]
    y_p = hNR[y_mask]
    z_p = hNR[z_mask]

    hNR_3D = [list(x) for x in zip(x_p, y_p, z_p)]
    hNR_3D = np.asarray(hNR_3D)
    hNR_3D = hNR_3D * 10
    new_CAs = CAs_A + hNR_3D

    rewrite_pdb(ref_pdb, new_CAs, pdbfile[:-4] + "_MIN.pdb")



def main(pdbfile, ref_pdb):
    PDB_ntuple = read_pdb(pdbfile, unit_nm=True)
    PDB_ref = read_pdb(ref_pdb, unit_nm=True)
    CAs = PDB_ntuple.ca_coords
    CAs_ref = PDB_ref.ca_coords
    PDB_ntuple_A = read_pdb(pdbfile, unit_nm=False)
    PDB_ref_A = read_pdb(ref_pdb, unit_nm=False)
    CAs_A = PDB_ntuple_A.ca_coords
    CAs_ref_A = PDB_ref_A.ca_coords

    # From calc_modes
    diss = cdist(CAs_ref, CAs_ref)  # distance matrix of all C-Alpha atoms
    ks = konrad_force_cons(diss)
    h = build_H(CAs_ref, ks, diss, PDB_ntuple.weight)  # build Hessian matrix

    n = h.shape[0]
    n = min(n, 3 * len(CAs) - 6)
    if n > 20:
        e, v = eigh(h)
        e = e[:n + 6]
        v = v[:, :n + 6]
    else:
        e, v = eigsh(h, k=n + 6, which='SA')

    # g = build_g(CAs, CAs_ref, h)
    g = build_g(CAs, CAs_ref, ks)

    gProj = build_gProj(g, v, PDB_ntuple.weight)

    hNR = build_hNR(gProj, v, e)

    project_struct(ref_pdb, pdbfile, CAs_A, hNR)


if __name__ == '__main__':
    main("3tfy_C_m7_a3.00_s1_1_3.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")
    main("3tfy_C_m7_a6.00_s1_1_6.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")
    main("3tfy_C_m7_a9.00_s1_1_9.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")
    main("3tfy_C_m7_a12.00_s1_1_12.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")
    main("3tfy_C_m7_a15.00_s1_1_15.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")
    main("3tfy_C_m7_a18.00_s1_1_18.00_ca.pdb", "3tfy_C_m7_a0.00_s1_0_0.00_ca.pdb")

    for file in os.listdir('.'):
        if file.endswith("MIN.pdb"):
            from shutil import move
            import subprocess
            from distutils.spawn import find_executable

            # pulchra_path = shutil.which("pulchra")
            pulchra_path = find_executable('pulchra')

            # p = subprocess.call("{0} -p -q {1}".format(pulchra_path, file),
            p = subprocess.call("{0} {1}".format(pulchra_path, file),
                                shell=True,
                                stderr=subprocess.PIPE)
            if not p == 0:
                raise Exception("PULCHRA BUGGED on " + file)
            # move(file[:-4] + ".rebuilt.pdb", ".")