# Customize sys.exit code for supporting cmd line execution in job runner
import sys

class PDBFILE_INVALID(Exception):
    def __init__(self, exc='Empty or invalid protein file'):
        super().__init__(exc)
        sys.exit(2)

class PDB_DOWNLOAD_FAIL(Exception): 
    def __init__(self, exc='Fail downloading PDB ID'):
        super().__init__(exc)
        sys.exit(3)


class FASTA_INVALID(Exception):
    def __init__(self, exc='Fasta file can not be parsed.'):
        super().__init__(exc)
        # print(exc)
        sys.exit(4)
