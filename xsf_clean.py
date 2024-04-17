import os
import os.path as path
import csv
import numpy as np

def get_max_elements():
    return 95

def get_num_structs():
    return 7815

# region energy_compilation
def get_energy_(dataset_path, s_no):
    file = path.join(dataset_path, f"structure{str(s_no).zfill(4)}.xsf")
    energy_value = 0.0
    with open(file, "r") as f:
        energy_value = float(f.readlines(1)[0].split(" ")[4])

    return energy_value

def write_energies_(target_path, energies):
    full_path = path.join(target_path, r"energies.csv")
    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["energy_eV"])
        for energy in energies:
            writer.writerow([energy])

def create_targets():
    dataset_path = r"./dat/data-set-2016-TiO2"
    target_path  = r"./dat/"
    
    n_structures = get_num_structs()
    energies = np.array([
        get_energy_(dataset_path, i) for i in range(1, n_structures + 1)
    ])

    write_energies_(target_path, energies)
# endregion

# region position_compilation
def get_positions_(dataset_path, s_no):
    file = path.join(
        dataset_path,
        f"structure{str(s_no).zfill(4)}.xsf"
    )
    size_info_idx_ = 8
    start_idx_     = 9
    dim_           = 3
    
    with open(file, "r") as f:
        lines = f.readlines()

        # get number of atoms in structure
        n_atoms = int(lines[size_info_idx_].split(" ")[0])

        # initialise position array
        positions = np.zeros((n_atoms, dim_))
        
        for idx, l in enumerate(lines[start_idx_ : start_idx_ + n_atoms]):
            positions[idx, :] = np.array(
                l.split()[1 : 1 + dim_]
            ).astype("float")

        if is_in_pos := 10 * np.ones(3) in positions:
            print(f"{is_in_pos = }")
    return positions

if __name__ == "__main__":
    create_targets()