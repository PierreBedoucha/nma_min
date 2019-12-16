from shutil import copy



if __name__ == '__main__':

    import os

    directory_list = list()
    files_list = list()
    paths_list = list()
    scripts_list = list()
    scripts_build_list = list()
    scripts_mini_list = list()
    for root, dirs, files in os.walk("../../compnma_api/data/output/python_output", topdown=False):
        for name in dirs:
            if name == "dcd_SC":
                directory_list.append(os.path.join(root, name))

    pdb_align_list = ['3tfy', '5isv', '4pv6', '2z0z', '1s7l', '2x7b', '3igr', '5k18',
                      '2cns', '5hh0', '5wjd', '5icv', '4kvm', '4u9v']

    for root in directory_list:
        # files_list.extend([x for x in os.listdir(dir) if os.path.isfile(x)])
        for x in os.listdir(root):
            if os.path.isfile(os.path.join(root, x)) and not x.startswith(".")\
                    and x.split("_")[0] in pdb_align_list:
                files_list.append(x)
                paths_list.append(os.path.join(root, x))

    for file in paths_list:
        copy(file, "./")