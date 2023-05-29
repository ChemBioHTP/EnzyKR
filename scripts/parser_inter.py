import os
import sys
import numpy as np
#import matplotlib.pyplot as plt

def calculate_distance_map(coords1, coords2):
    distance_map = np.zeros((len(coords1), len(coords2)))

    for i, coord1 in enumerate(coords1):
        for j, coord2 in enumerate(coords2):
            distance = np.sqrt(np.sum((np.array(coord1) - np.array(coord2)) ** 2))
            distance_map[i, j] = distance

    return distance_map

def stack_and_pad_maps(map1, map2):
    # Determine the maximum size of the maps
    max_size = max(map1.shape[0], map2.shape[0])

    # Pad the maps to the maximum size
    padded_map1 = np.pad(map1, ((0, max_size - map1.shape[0]), (0, max_size - map1.shape[1])), mode='constant')
    padded_map2 = np.pad(map2, ((0, max_size - map2.shape[0]), (0, max_size - map2.shape[1])), mode='constant')
    # Stack the maps together
    stacked_map = np.stack((padded_map1, padded_map2), axis=-1)
    return stacked_map

def pad_distance_map(distance_map, target_length):
    padded_map = np.zeros((target_length, target_length))
    padded_map[:distance_map.shape[0], :distance_map.shape[1]] = distance_map
    return padded_map

def save_distance_map(distance_map, output_file):
    np.save(output_file, distance_map)
'''
def plot_distance_map(distance_map):
    plt.imshow(distance_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    plt.title('Distance Map')
    plt.show()
'''

def process_pdb_files(pdb_folder):
    # Iterate over the PDB files in the folder
    for file_name in os.listdir(pdb_folder):
        if file_name.endswith('.pdb'):
            pdb_file = os.path.join(pdb_folder, file_name)
            output_file = os.path.join(pdb_folder, f'{os.path.splitext(file_name)[0]}_distance_map.npy')
            output_name = os.path.join(pdb_folder,
                                       f'{os.path.splitext(file_name)[0]}_distance_all.npz')

            # Load coordinates of alpha-carbon, beta-carbon, and ligand atoms
            alpha_carbon_coords = []
            beta_carbon_coords = []
            ligand_coords = []

            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        alpha_carbon_coords.append([x, y, z])

                    if line.startswith('ATOM') and line[12:16].strip() == 'CB':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        beta_carbon_coords.append([x, y, z])

                    if line.startswith('HETATM'):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        ligand_coords.append([x, y, z])

            # Calculate distance maps
            alpha_carbon_map = calculate_distance_map(alpha_carbon_coords, alpha_carbon_coords)
            ligand_protein_map = calculate_distance_map(ligand_coords, alpha_carbon_coords)

            beta_carbon_map = calculate_distance_map(beta_carbon_coords, beta_carbon_coords)
            beta_ligand_protein_map = calculate_distance_map(ligand_coords, beta_carbon_coords)

            alpha_beta_carbon_map = calculate_distance_map(beta_carbon_coords, alpha_carbon_coords)

            # Pad the distance maps to the same length
            #target_length = max(len(alpha_carbon_coords), len(ligand_coords))
            #alpha_carbon_map = pad_distance_map(alpha_carbon_map, target_length)
            #ligand_protein_map = pad_distance_map(ligand_protein_map, target_length)
            #beta_carbon_map = pad_distance_map(beta_carbon_map, target_length)

            # Concatenate the distance maps
            #concatenated_map = np.concatenate((alpha_carbon_map, ligand_protein_map, beta_carbon_map), axis=0)
            concatenated_map_alpha = np.concatenate((alpha_carbon_map, ligand_protein_map), axis=0)
            concatenated_map_beta = np.concatenate((beta_carbon_map, beta_ligand_protein_map), axis=0)
            concatenated_map_pl = np.concatenate((alpha_beta_carbon_map, alpha_beta_carbon_map), axis=0)

            #plot_distance_map(concatenated_map_alpha)
            #plot_distance_map(concatenated_map_beta)

            stacked_map = stack_and_pad_maps(concatenated_map_alpha, concatenated_map_beta)
            #print(stacked_map[:,:,0].shape)
            #plot_distance_map(stacked_map[:, :, 1])
            distance_maps = {'map_alpha': concatenated_map_alpha, 'map_beta':
                             concatenated_map_beta,
                             'map_pl':concatenated_map_pl, 'stacked_map': stacked_map, 'caca_map': alpha_carbon_map, 'cbcb_map': beta_carbon_map, 'alpha_pl_map': ligand_protein_map, 'beta_pl_map': beta_ligand_protein_map}
            # Save the concatenated distance map
            #save_distance_map(concatenated_map_alpha, output_file)
            #np.savez(output_name, **distance_maps)
            # Plot the concatenated distance map
            #plot_distance_map(concatenated_map_alpha)
            return concatenated_map_alpha
'''
# Check if the pdb_folder argument is provided
if len(sys.argv) < 2:
    print("Please provide the path to the PDB folder as an argument.")
    print("Usage: python distance.py pdb_folder")
    sys.exit(1)
'''
# Get the pdb_folder from the command-line argument
#pdb_folder = sys.argv[1]

# Process the PDB files in the folder
#data = process_pdb_files(pdb_folder)

