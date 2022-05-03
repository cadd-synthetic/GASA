from rdkit import Chem
import torch
import dgl
from dgl.data.utils import save_graphs
import pandas as pd
from functools import partial
from dgllife.utils import mol_to_bigraph
from dgllife.utils import ConcatFeaturizer, BaseAtomFeaturizer, BaseBondFeaturizer, atom_type_one_hot, atom_total_degree_one_hot, atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_implicit_valence_one_hot, atom_chiral_tag_one_hot, atom_is_aromatic, atom_is_in_ring
from dgllife.utils import bond_stereo_one_hot, atom_formal_charge_one_hot, atom_total_num_H_one_hot, bond_is_in_ring, bond_is_conjugated, bond_type_one_hot
from sklearn.preprocessing import LabelEncoder


class AtomF(BaseAtomFeaturizer):
    """
    extract atom and bond feature
    """
    def __init__(self, atom_data_field='hv'):
        super(AtomF, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                     'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
                 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'], encode_unknown=False),
                 atom_total_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge_one_hot,
                 atom_num_radical_electrons_one_hot,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_is_in_ring,
                 atom_total_num_H_one_hot,
                 atom_chiral_tag_one_hot])})


class BondF(BaseBondFeaturizer):
    def __init__(self, bond_data_field='he', self_loop=False):
        super(BondF, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE])])}, self_loop=self_loop)


def generate_graph(smiles):
    """
    Converts SMILES into graph with features.
    Parameters
    smiles: SMILES representation of the moelcule of interest
            type smiles: list
    return: DGL graph with features
            rtype: list
            
    """
    atom = AtomF(atom_data_field='hv')
    bond = BondF(bond_data_field='he', self_loop=True)
    graph = []
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        Chem.SanitizeMol(mol)
        g = mol_to_bigraph(mol,
                        node_featurizer=atom,
                        edge_featurizer=bond,
                        add_self_loop=True)
        graph.append(g)
    return graph




