# Python wrapper for calling mustang to do PDB alignment
# e.g
# mustang -i 2vl0_A.pdb 3eam_A.pdb 3rhw_A.pdb 3tlu_A.pdb -o webnma -F fasta

import subprocess
from os.path import join

from .testing import timing


@timing 
def main(pdbs, tar_dir=".", identifier='webnma3'):
    '''
    when the alignment is done successfully, the following will be generated:
    1. <identifier>.afasta : alignment file 
    2. <identifier>.pdb : superposition file for proteins visualiztion 
    '''
    pdbs = '-i ' + ' '.join(pdbs)
    alignment_format = '-F fasta'  # must specify the format 
    output_identifier = '-o ' + join(tar_dir, identifier)
    cmd = ' '.join(['mustang', pdbs, alignment_format, output_identifier])
    
    status, _, err = launch_cmd(cmd)
    if status != 0:
        raise Exception("Mustang alignment failed.\n" + str(err))

    
def launch_cmd(cmd: str):
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         shell=True,
                         stderr=subprocess.PIPE,
                         bufsize=1)

    stdout, stderr = p.communicate()
    p.wait()
    return (p.returncode, stdout, stderr)    


if __name__ == '__main__':
    import sys
    from os.path import join
    if len(sys.argv) > 2:
        main(sys.argv[1:]) # do not pass identifier
    else:
        INPUT = join('webnma_api','tests', 'data_profile_alignment', 'input')
        OUTPUT = INPUT[:-5] + 'output'
        pdbs = ['2vl0_A.pdb', '3eam_A.pdb', '3rhw_A.pdb', '3tlu_A.pdb']
        pdbs_full = [join(INPUT, p) for p in pdbs]
        main(pdbs_full, OUTPUT)
