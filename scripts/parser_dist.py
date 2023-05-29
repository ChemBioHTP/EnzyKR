import numpy as np
import os, sys

def calculate_distance_map(pdb_file):
    # Read the PDB file and extract relevant information
    ligand_atoms = []
    protein_atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                if atom_name == 'CA' and residue_name not in ('HOH', 'SOL', 'WAT'):  # Exclude water molecules
                    protein_atoms.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
                elif atom_name not in ('H', 'H1', 'H2', 'H3'):  # Exclude ligand hydrogen atoms
                    ligand_atoms.append(np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))

    ligand_com = np.mean(ligand_atoms, axis=0)

    # Calculate distance map
    distance_map = []
    for protein_atom in protein_atoms:
        distance = np.linalg.norm(ligand_com - protein_atom)
        distance_map.append(distance)

    return np.array(distance_map)


def calculate_distance_map_folder(folder_path):
    distance_maps = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a PDB file
        if file_name.endswith('.pdb'):
            # Construct the full path of the PDB file
            pdb_file = os.path.join(folder_path, file_name)

            # Calculate the distance map for the PDB file
            distance_map = calculate_distance_map(pdb_file)


    return distance_map

pdb_files = sys.argv[1:]

for pdb_file in pdb_files:
    # Calculate distance map
    distance_map = calculate_distance_map(pdb_file)

    # Determine the output directory and create it if it doesn't exist
    output_dir = os.path.dirname(pdb_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save distance map as NPY file in the same subfolder as the PDB file
    #npy_file = os.path.join(output_dir, os.path.splitext(os.path.basename(pdb_file))[0] + 'avg_com'+'.npy')
    #np.save(npy_file, distance_map)
    #print(f"Distance map saved as {npy_file}")

