from collections import namedtuple
from os.path import join, basename
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select, Atom
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from scipy.spatial.distance import cdist

from .webnma_exceptions import *
from .residue_mass import RES_MASS
from .residue_name import RES_NAME

DIG = 3 # digits kept for atom coordinates in pdb files

PDB_ntuple = namedtuple('Pdb_ntuple', 'ca_coords residues residues_full weight')


def is_ca(a):
    '''
    Check if an atom is a c-alpha atom in a pdb file according to Bio.PDB
    Note: the full id of an atom is the tuple:
     (structure id, model id, chain id, residue id, atom name, altloc)
    where a residue id is:
       (hetero-flag, sequence identifier, insertion code)
    1. The hetero-flag: this is 'H_' plus the name of the hetero-residue
    (e.g. 'H_GLC' in the case of a glucose molecule), or 'W' in the case
    of a water molecule.
    2. The sequence identifier in the chain, e.g. 100
    3. The insertion code,
    For more: 
    https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
    '''
    test1 = lambda x: x.get_id() == 'CA'  # check atom name 
    test2 = lambda x: x.get_full_id()[3][0] == ' ' # check hetero flag
    # check reside name
    test3 = lambda x: x.get_parent().get_resname() in RES_NAME.keys()
    
    return test1(a) and test2(a) and test3(a)


is_cif = lambda f: f[-4:] == '.cif'


def set_parser(protein_file):
    '''
    Choose the correct parser according to the protein file's format
    '''
    if is_cif(protein_file):
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    return parser


# Support .pdb and .cif
def read_pdb(pdb, unit_nm = False) -> namedtuple:
    '''
    Extract the coordinates of C-alpha atoms, residue names and the mass,
    all stored in numpy.array
    '''
    parser = set_parser(pdb)
    protein = parser.get_structure(pdb[:4], pdb)

    CAs = []
    Rs = []
    Rs_full = []
    for a in protein.get_atoms():
        if is_ca(a):
            if unit_nm:
                CAs.append(a.get_coord() / 10)
            else:
                CAs.append(a.get_coord())
            r = a.get_parent()
            r_name = r.get_resname()
            r_id = r.get_full_id()
            r_fullname = r_id[2] + '.' + r_name + str(r_id[3][1]) # e.g, A.Gly123
            Rs.append(r_name)
            Rs_full.append(r_fullname)

    if len(CAs) == 0:
        raise PDBFILE_INVALID
    else:
        w = mass_protein(Rs)    
        arrays = [np.array(l) for l in [CAs, Rs, Rs_full, w]]
        return PDB_ntuple(*arrays)


def mass_protein(rs, full=False, weighted=True):
    '''
    Retrieve the mass for each residue in 'rs'
    set 'full' True if residue name is in the format A.Met100 (like in modefile)
    set 'weighted' True if mass needs square-rooted
    '''
    if full:
        rs = [r[2:5] for r in rs]

    rs = [RES_MASS[r.upper()] for r in rs]

    if weighted:
        rs = [r**0.5 for r in rs]
    return rs
        


def calc_dist(pdbfile):
    '''
    Calculate the distance matrix of the CA atoms (unit: angstrom Å)
    '''
    CAs = read_pdb(pdbfile).ca_coords
    return cdist(CAs, CAs)
   

def calc_ss(pdbfile) -> [str]:
    '''
    Calculate the secondary structure of the protein.
    Code Structure
    H 	Alpha helix (4-12)
    B 	Isolated beta-bridge residue
    E 	Strand
    G 	3-10 helix
    I 	Pi helix
    T 	Turn
    S 	Bend
    - 	None
    '''

    '''
    dssp_dict_from_pdb_file simply Popen 'mkdssp' and then deals with its output,
    src: http://biopython.org/DIST/docs/api/Bio.PDB.DSSP%27-pysrc.html#dssp_dict_from_pdb_file
    '''
    # keys :: [(chainid, res_id)], eg [('A', (' ', 12, ' ')), ...]
    ss_dict, keys = dssp_dict_from_pdb_file(pdbfile)  # this supports .cif also
    
    # make the resides' order consistent as it is in C-alpha file (i.e 'modes_CA.pdb')
    # to plot fluctuations with 2nd structure correctly
    parser = set_parser(pdbfile) 
    protein = parser.get_structure(pdbfile[:4], pdbfile)
    ss_list = []
    for a in protein.get_atoms():
        if is_ca(a):
            full_id = a.get_full_id()
            new_key = (full_id[2], full_id[3])
            if new_key in ss_dict:
                ss_list.append(ss_dict[new_key][1])
    return ss_list


def comp_pdb(pdb1, pdb2):
    pdb1 = read_pdb(pdb1)
    pdb2 = read_pdb(pdb2)
    for x,y in zip(pdb1.ca_coords, pdb2.ca_coords):
        test1 = np.array_equal(x,y)
    test2 = pdb1.residues == pdb2.residues
    return test1 and test2


# Only output .pdb, no .cif (but can use .cif as input also)
# because Bio.Python no longer maiatains MMCIFIO
def rewrite_pdb(pdb, CA_coords, new_name):
    parser = set_parser(pdb)
    protein = parser.get_structure(pdb[:4], pdb)

    i = 0
    for a in protein.get_atoms():
        if is_ca(a):
            try:
                a.set_coord([round(c, DIG) for c in CA_coords[i]])
                i = i + 1
            except IndexError:
                raise Exception('Unequal number of C-alpha atoms.')
  
    if i != len(CA_coords):
        raise Exception('Unequal number of C-alpha atoms.')

    io = PDBIO()
    io.set_structure(protein)
    io.save(new_name)


SERVER = "https://www.ebi.ac.uk/pdbe/entry-files/download/"
CIF_FORM_URL = SERVER + "{}.cif"
PDB_FORM_URL = SERVER + "pdb{}.ent"
# TODO: support compressed(.gz) download?
def download_pdb(pdb_id, tar_dir='.', file_format='pdb'):
    '''
    Download pdb from PDBe(Protein Data Bank in Europe)
    Doc: https://www.ebi.ac.uk/pdbe/api/doc/
    '''
    pdb_id = pdb_id.lower()
    if file_format.lower() in ['cif', 'mmcif']:
        url = CIF_FORM_URL.format(pdb_id)
    else:
        url = PDB_FORM_URL.format(pdb_id)        
    dl_path = join(tar_dir, basename(url))
    try:
        urlretrieve(url, dl_path)
        print("Downloaded %s successfully." % pdb_id)
        return dl_path
    except URLError as e:
        raise PDB_DOWNLOAD_FAIL('Fail downloading %s: %s' % (pdb_id, e.reason))


def download_pdbs(pdb_ids:list, tar_dir='.', file_format='pdb'):
    return [download_pdb(i, tar_dir, file_format) for i in pdb_ids]

        
# Subclass PDBIO.Select to select only c-alpha atoms
class CA_select(Select):              
    def accept_atom(self, atom):
        if is_ca(atom):
            return 1
        else:
            return 0

        
def save_CAs(pdb, target):
    parser = set_parser(pdb)
    protein = parser.get_structure(pdb[:4], pdb)
    io = PDBIO()    
    io.set_structure(protein)
    io.save(target, CA_select(), preserve_atom_numbering=False)



class Chain_select(Select):
    def __init__(self, chains):
        self.chains = chains
        
    def accept_chain(self, chain):        
        if chain.get_id() in self.chains:
            return 1
        else:
            return 0
    
def save_chains(pdb, chains, target):
    parser = set_parser(pdb)
    protein = parser.get_structure(pdb[:4], pdb)
    io = PDBIO()    
    io.set_structure(protein)
    io.save(target, Chain_select(chains), preserve_atom_numbering=False)
    
    
    
if __name__ == '__main__':
    import sys
    save_CAs('../../data/pdbs/1su4.pdb', '../test/1su4_ca.pdb')
