'''
The major part of preprocess at the substrates for the nodes and edges are adopted from https://github.com/guaguabujianle/MGraphDTA
'''
import os.path as osp
import numpy as np
import torch, os
import pandas as pd
from scripts.parser_dist import *
from scripts.parser_inter import *
from scripts.parser_msa import *
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit.Chem import MolFromSmiles, ChemicalFeatures
import networkx as nx
from rdkit import Chem, RDConfig
from tqdm import tqdm

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


AA_DICT = { "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,
                 "G": 7, "H": 8, "I": 9, "-": 10, "K": 11, "L": 12,
                 "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18,
                 "S": 19, "T": 20, "U": 21,
                 "V": 22, "W": 23, "X": 24,
                 "Y": 25, "Z": 26 }



def seqs2int(target):

    return [AA_DICT[s] for s in target]


class DataCreate(InMemoryDataset):

    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['tutorial.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_tutorial.pt']


    def process_data(self, data_path, graph_dict, msa_folder, pdb_folder):
        data = pd.read_csv(data_path)
        data_list = []
        for i, row in data.iterrows():
            ids = row['ids']
            smi = row['substrates']
            sequence = row['enzymes']
            label = row['dg++']

            x, edge_index, edge_attr = graph_dict[smi]

            # caution
            x = (x - x.min()) / (x.max() - x.min())

            target_len = 1254
            #process the msa
            msa_path = osp.join(msa_folder, f'{ids}.a3m')
            with open(msa_path, 'r') as f:
                a3m_file = f.read()
            alns = parse_a3m(a3m_file)
            a3m_int = []
            for seq in alns:
                seqs_tmp = seqs2int(seq)
                a3m_int.append(seqs_tmp)
            padded_msas =torch.LongTensor(np.array([resize_2d(np.array(a3m_int),target_len)])) #convert the msas into longtensor

            target_size = (1254, 1254)

            #The module use to load the avg distance map
            #avg_dist_path = osp.join(distance_map_folder, f'{ids}/positive/positiveavg_com.npy')
            pdb_path = osp.join(pdb_folder)
            #avg_dist = np.load(avg_dist_path)
            avg_dist = calculate_distance_map_folder(pdb_path)
            if len(avg_dist) < target_size[0]:
                padded_avg_distance_map = np.pad(avg_dist, (0, target_size[0] - len(avg_dist)))
            else:
                padded_avg_distance_map = padded_avg_distance_map[:target_size[0]]
            padded_avg_distance_map = torch.FloatTensor(np.array([padded_avg_distance_map]))

            #The module use to add the interaction map
            #distance_map_path = osp.join(distance_map_folder, f'{ids}/positive/positive_distance_map.npy')
            #distance_map = np.load(distance_map_path)
            distance_map = process_pdb_files(pdb_path)
            padded_distance_map = np.pad(distance_map, ((0, target_size[0] - distance_map.shape[0]),(0, target_size[1] - distance_map.shape[1])))
            padded_distance_map = torch.from_numpy(padded_distance_map).unsqueeze(0)

            # Get Labels
            try:
                data = DATA.Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.FloatTensor([label]),
                    msas=padded_msas,
                    distance_map = padded_distance_map,
                    avg_distance_map=padded_avg_distance_map
                )
            except:
                    print("unable to process: ", smi)

            data_list.append(data)

        return data_list

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = df['substrates'].unique()
        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            g = self.mol2graph(mol)
            graph_dict[smile] = g

        test_list = self.process_data(self.raw_paths[0],graph_dict,msa_folder="./msa/", pdb_folder="./structures")


        print('All features prepared!')


        data, slices = self.collate(test_list)
        # save preprocessed test data:
        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])

        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    g.nodes[n]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        return node_attr, edge_index, edge_attr

if __name__ == "__main__":
    DataCreate('./')

