import torch
import logging

from glyles import convert
from collections import defaultdict

from torchdrug import utils
from torchdrug.data import Molecule, PackedMolecule, Graph, PackedGraph
from torchdrug.utils import pretty

from module.custom_data.glycan import Glycan

logger = logging.getLogger(__name__)


class AllAtomGlycan(Molecule):
    """
    Baseline data structure of All-Atom Glycan.
    """

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, atom_feature=None, bond_feature=None,
                 mol_feature=None, formal_charge=None, explicit_hs=None, chiral_tag=None, radical_electrons=None,
                 atom_map=None, bond_stereo=None, stereo_atoms=None, node_position=None, **kwargs):
        super(AllAtomGlycan, self).__init__(edge_list, atom_type, bond_type, atom_feature, bond_feature, mol_feature,
                                            formal_charge, explicit_hs, chiral_tag, radical_electrons, atom_map,
                                            bond_stereo, stereo_atoms, node_position, **kwargs)

    @classmethod
    def from_iupac(cls, iupac, unit_feature="default", link_feature="default", **kwargs):
        smiles_sequence = convert(iupac)[0][1]
        if smiles_sequence == "":
            logger.warning("Unsupported IUPAC %s, contain 0 node" % iupac)
        mol = cls.from_smiles_new(smiles=smiles_sequence, unit_feature=unit_feature, link_feature=link_feature,
                                  **kwargs)
        return mol, smiles_sequence

    @classmethod
    def from_smiles_new(cls, smiles, unit_feature="default", link_feature="default", **kwargs):
        mol = None
        try:
            mol = cls.from_smiles(smiles=smiles, atom_feature=unit_feature, bond_feature=link_feature, **kwargs)
        except ValueError:
            logger.warning("Invalid SMILES %s, treat as None" % smiles)
        return mol


class PackedAllAtomGlycan(PackedMolecule, Molecule):
    """
    Baseline data structure of All-Atom Packed Glycan.
    """

    unpacked_type = AllAtomGlycan

    def __init__(self, edge_list=None, atom_type=None, bond_type=None, num_nodes=None, num_edges=None, offsets=None,
                 out_degrees=None, in_degrees=None, spatial_poses=None, len_edge_pathss=None, padded_edge_pathss=None,
                 **kwargs):
        super().__init__(edge_list, atom_type, bond_type, num_nodes, num_edges, offsets, **kwargs)

    @classmethod
    def from_iupac(cls, iupacs, atom_feature="default", bond_feature="default", **kwargs):
        smiles_list = [x[1] for x in convert(glycan_list=iupacs)]
        packed_mol = None
        try:
            packed_mol = cls.from_smiles(smiles_list=smiles_list, atom_feature=atom_feature, bond_feature=bond_feature,
                                         **kwargs)
        except ValueError:
            logger.warning("Unsupported SMILES list, treat as None")
        return packed_mol


class HeterogeneousAllAtomGlycan(Graph):
    """
    Implementation of heterogeneous graph of all-atom glycan structure proposed in the paper.
    """

    glycan2smiles = {'dHex': '', 'Gal': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'AltN': 'O1C(O)[C@H](N)[C@@H](O)[C@@H](O)[C@@H]1CO',
                     'FucN': 'O1C(O)[C@@H](N)[C@H](O)[C@H](O)[C@@H]1C',
                     'Rha': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@@H]1C',
                     'GalNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Glc': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'GlcN': 'O1C(O)[C@H](N)[C@@H](O)[C@H](O)[C@H]1CO',
                     'GlcNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Man': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'ManNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Xyl': 'O1C[C@@H](O)[C@H](O)[C@@H](O)C1O', 'Monosaccharide': '',
                     'Alt': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1CO', 'Ery-ol': 'OC[C@H](O)[C@H](O)CO',
                     'All': 'O1C(O)[C@H](O)[C@H](O)[C@H](O)[C@H]1CO', 'Rib': 'O1C[C@@H](O)[C@@H](O)[C@@H](O)C1O',
                     'Tal': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Gul': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Kdo': 'O1[C@H]([C@H](O)CO)[C@H](O)[C@H](O)CC1(O)C(=O)O',
                     'Kdof': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](O)CC1(O)C(=O)O',
                     'Fruf': 'O1[C@H](CO)[C@@H](O)[C@H](O)C1(O)CO', 'Fuc': 'O1C(O)[C@@H](O)[C@H](O)[C@H](O)[C@@H]1C',
                     'LyxHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(O)CO',
                     'Neu5Ac': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](NC(C)=O)[C@@H](O)CC1(O)C(=O)O',
                     'Ara': 'O1C[C@H](O)[C@H](O)[C@@H](O)C1O', 'HexNAc': 'O1C(O)C(NC(C)=O)C(O)C(O)C1CO',
                     '4eLeg': 'O1[C@@H]([C@H](N)[C@@H](C)O)[C@H](N)[C@H](O)CC1(O)C(=O)O',
                     'Araf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1CO', 'Lyxf': 'O1C(O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     '6dAlt': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1C',
                     '6dAltNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@@H]1C',
                     '6dGul': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1C', 'dHexNAc': '',
                     'ManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(O)CO',
                     '6dTal': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1C',
                     'TalNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Aci': 'O1[C@@H]([C@@H](N)[C@H](C)O)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Leg': 'O1[C@@H]([C@H](N)[C@@H](C)O)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Abe': 'O1C(O)[C@H](O)C[C@@H](O)[C@H]1C',
                     'AcoNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](OC)[C@@H](O)[C@@H]1C',
                     'AllN': 'O1C(O)[C@H](N)[C@H](O)[C@H](O)[C@H]1CO',
                     'AllNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@H](O)[C@H]1CO',
                     'AltA': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1C(=O)O', 'Api': 'O1C[C@](O)(CO)[C@@H](O)C1O',
                     'Apif': 'O1C[C@](O)(CO)[C@@H](O)C1O', 'AraN': 'O1C[C@H](O)[C@H](O)[C@@H](N)C1O',
                     'Asc': 'O1C(O)[C@H](O)C[C@@H](O)[C@@H]1C', 'Bac': 'O1C(O)[C@H](N)[C@@H](O)[C@H](N)[C@H]1C',
                     'Col': 'O1C(O)[C@@H](O)C[C@H](O)[C@@H]1C', 'HexA': 'O1C(O)C(O)C(O)C(O)C1C(=O)O',
                     'Hex': 'O1C(O)C(O)C(O)C(O)C1CO', 'HexN': 'O1C(O)C(N)C(O)C(O)C1CO',
                     'IdoHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1C(O)CO', 'Lyx': 'O1C[C@@H](O)[C@H](O)[C@H](O)C1O',
                     'FucNAc': 'O1C(O)[C@@H](NC(C)=O)[C@H](O)[C@H](O)[C@@H]1C',
                     'Fucf': 'O1C(O)[C@@H](O)[C@H](O)[C@H]1[C@H](C)O',
                     'FucfNAc': 'O1C(O)[C@@H](NC(C)=O)[C@H](O)[C@H]1[C@H](C)O',
                     'Ido': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1CO',
                     'IdoA': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1C(=O)O',
                     'IdoNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@@H]1CO',
                     'RhaN': 'O1C(O)[C@H](N)[C@H](O)[C@@H](O)[C@@H]1C',
                     'RhaNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@@H](O)[C@@H]1C',
                     'Sor': 'O1C[C@H](O)[C@@H](O)[C@H](O)C1(O)CO', 'Thre-ol': 'OC[C@@H](O)[C@H](O)CO',
                     'DDAltHep': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GalHep': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1C(O)CO',
                     'DDGlcHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO',
                     'GalN': 'O1C(O)[C@H](N)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'DLGlcHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO',
                     'Dha': 'O1[C@H](C(=O)O)[C@H](O)[C@H](O)CC1(O)C(=O)O', 'Dig': 'O1C(O)C[C@H](O)[C@H](O)[C@H]1C',
                     'Erwiniose': 'O1C(O)[C@H](O)C[C@](O)([C@@H](O)C[C@@H](C)O)[C@H]1C',
                     'Fru': 'O1C[C@@H](O)[C@@H](O)[C@H](O)C1(O)CO',
                     'Fus': 'O1[C@@H]([C@@H](O)[C@H](C)O)[C@@H](N)[C@H](O)CC1(O)C(=O)O',
                     'GalA': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1C(=O)O',
                     'Galf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GalfNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GlcA': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1C(=O)O',
                     'GulA': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1C(=O)O',
                     'GulN': 'O1C(O)[C@H](N)[C@H](O)[C@@H](O)[C@H]1CO',
                     'GulNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Ins': 'O[C@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O',
                     'Kdn': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](O)[C@@H](O)CC1(O)C(=O)O',
                     'Ko': 'O1[C@H]([C@H](O)CO)[C@H](O)[C@H](O)[C@H](O)C1(O)C(=O)O',
                     'Oli': 'O1C(O)C[C@@H](O)[C@H](O)[C@H]1C', 'Psi': 'O1C[C@@H](O)[C@@H](O)[C@@H](O)C1(O)CO',
                     'Qui': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1C', 'QuiN': 'O1C(O)[C@H](N)[C@@H](O)[C@H](O)[C@H]1C',
                     'QuiNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1C', 'Ribf': 'O1C(O)[C@H](O)[C@H](O)[C@H]1CO',
                     'ManA': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(=O)O',
                     'ManN': 'O1C(O)[C@@H](N)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Manf': 'O1C(O)[C@@H](O)[C@@H](O)[C@H]1[C@H](O)CO',
                     'Mur': 'O1C(O)[C@H](N)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     'MurNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     'Neu': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Neu5Gc': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](NC(=O)CO)[C@@H](O)CC1(O)C(=O)O',
                     'Neu4Ac': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](N)[C@@H](OC(C)=O)CC1(O)C(=O)O',
                     'Par': 'O1C(O)[C@H](O)C[C@H](O)[C@H]1C', 'Pau': 'O1C(O)C[C@@H](O)[C@@H](O)[C@H]1C',
                     'Pse': 'O1[C@@H]([C@@H](N)[C@H](C)O)[C@@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Sedf': 'O1[C@H]([C@H](O)CO)[C@@H](O)[C@H](O)C1(O)CO',
                     'Sia': 'O1C(C(O)C(O)CO)C(NC(C)=O)C(O)CC1(O)C(=O)O', 'Tag': 'O1C[C@@H](O)[C@H](O)[C@H](O)C1(O)CO',
                     'TalA': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1C(=O)O', 'Tyv': 'O1C(O)[C@@H](O)C[C@H](O)[C@H]1C',
                     'Xluf': 'O1C[C@@H](O)[C@H](O)C1(O)CO', 'Xylf': 'O1C(O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Yer': 'O1C(O)[C@H](O)C[C@](O)([C@H](C)O)[C@H]1C',
                     'ManfNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@H]1[C@H](O)CO', 'Unknown': '',
                     'LDManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1[C@@H](O)CO',
                     'AllA': 'O1C(O)[C@H](O)[C@H](O)[C@H](O)[C@H]1C(=O)O', 'ddHex': '',
                     'DDManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO', 'ddNon': '', 'Assigned': '',
                     'GlcfNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H]1[C@H](O)CO',
                     'MurNGc': 'O1C(O)[C@H](NC(=O)CO)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     '6dTalNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1C',
                     'TalN': 'O1C(O)[C@@H](N)[C@@H](O)[C@@H](O)[C@H]1CO', 'dNon': '',
                     '6dAltf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1[C@H](C)O',
                     'AltNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@@H]1CO', 'IdoNGlcf': '',
                     'Aco': 'O1C(O)[C@H](O)[C@H](OC)[C@@H](O)[C@@H]1C'}
    glycan2mol = {}

    offset_atom_id_domain = len(Glycan.units)
    offset_bond_id_domain = len(Glycan.links)
    length_bond_id_domain = len(Molecule.bond2id)

    def __init__(self, edge_list=None, unit_type=None, link_type=None,
                 unit_feature=None, link_feature=None, cumsum=None, **kwargs):
        super().__init__(edge_list=edge_list, **kwargs)
        unit_type, link_type = self._standarize_unit_link(unit_type, link_type)

        self.cumsum = cumsum

        with self.unit():
            if unit_feature is not None:
                self.unit_feature = torch.as_tensor(unit_feature, device=self.device)
            self.unit_type = unit_type

        with self.link():
            if link_feature is not None:
                self.link_feature = torch.as_tensor(link_feature, device=self.device)
            self.link_type = link_type

    @classmethod
    def from_iupac(cls, iupac, unit_feature="default", link_feature="default",
                   atom_feature="default", bond_feature="default", **kwargs):
        """
        Create a heterogeneous all-atom glycan graph from IUPAC-condensed sequence.
        """
        unit_feature = cls._standarize_option(unit_feature)
        link_feature = cls._standarize_option(link_feature)
        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)

        offset_atom_feature, offset_unit_feature = 0, 0
        glycan = Glycan.from_iupac("Man(a1-4)Man", unit_feature=unit_feature, link_feature=link_feature)
        offset_unit_feature = len(glycan.unit_feature[0])
        offset_link_feature = len(glycan.link_feature[0])
        _ext_atom = [0 for _ in range(offset_unit_feature)]
        _ext_bond = [0 for _ in range(offset_link_feature)]
        num_relation_glycan = glycan.num_relation

        smiles = cls.glycan2smiles["Gal"]
        mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
        offset_atom_feature = len(mol.atom_feature[0])
        offset_bond_feature = len(mol.bond_feature[0])
        _ext_unit = [0 for _ in range(offset_atom_feature)]
        _ext_link = [0 for _ in range(offset_bond_feature)]

        cross_feature = [0 for _ in range(offset_bond_feature + offset_link_feature)] + [1]
        cross_type = cls.offset_bond_id_domain + cls.length_bond_id_domain
        num_relation_atom = mol.num_relation

        glycan = Glycan.from_iupac(iupac, unit_feature=unit_feature, link_feature=link_feature)

        num_node_glycan = glycan.num_node
        num_node_atom = 0

        if glycan is None or num_node_glycan == 0:
            return None

        offset_unit_index = len(glycan.unit_type)
        cumsum = [offset_unit_index]

        all_unit_type = glycan.unit_type.tolist()
        all_unit_feature = [x + _ext_unit for x in glycan.unit_feature.tolist()]
        all_link_type = glycan.link_type.tolist()
        all_link_feature = [x + _ext_link + [0] for x in glycan.link_feature.tolist()]
        all_edge_list = glycan.edge_list.tolist()

        for index, unit_type_id in enumerate(glycan.unit_type.tolist()):
            mol = None
            glycan = Glycan.id2unit[unit_type_id]
            if glycan in cls.glycan2mol:
                mol = cls.glycan2mol[glycan]
            else:
                smiles = cls.glycan2smiles[glycan]
                if smiles != "":
                    mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
                    cls.glycan2mol[glycan] = mol
            if mol is None:
                cumsum.append(offset_unit_index)
                continue
            num_node_atom += mol.num_node

            _atom_type = [x + cls.offset_atom_id_domain for x in mol.atom_type.tolist()]
            all_unit_type.extend(_atom_type)

            _atom_feature = [_ext_atom + x for x in mol.atom_feature.tolist()]
            all_unit_feature.extend(_atom_feature)

            _bond_type = [x + cls.offset_bond_id_domain for x in mol.bond_type.tolist()]
            _bond_feature = [_ext_bond + x + [0] for x in mol.bond_feature.tolist()]

            all_link_type.extend(_bond_type)
            all_link_feature.extend(_bond_feature)

            for edge in mol.edge_list:
                src_ = edge[0] + offset_unit_index
                tgt_ = edge[1] + offset_unit_index
                type_ = edge[2] + cls.offset_bond_id_domain
                all_edge_list.append([src_, tgt_, type_])

            for atom_index in range(len(mol.atom_type)):
                all_edge_list.extend([[index, atom_index + offset_unit_index, cross_type],
                                      [atom_index + offset_unit_index, index, cross_type]])
                for _ in range(2):
                    all_link_type.append(cross_type)
                    all_link_feature.append(cross_feature)

            offset_unit_index += len(mol.atom_type)
            cumsum.append(offset_unit_index)

        all_unit_type = torch.tensor(all_unit_type)
        all_link_type = torch.tensor(all_link_type)
        if len(all_unit_feature) > 0:
            all_unit_feature = torch.tensor(all_unit_feature)
        else:
            all_unit_feature = None
        if len(all_link_feature) > 0:
            all_link_feature = torch.tensor(all_link_feature)
        else:
            all_link_feature = None
        num_node = num_node_glycan + num_node_atom
        num_relation = num_relation_glycan + num_relation_atom + 1

        return cls(all_edge_list, all_unit_type, all_link_type, all_unit_feature, all_link_feature,
                   num_node=num_node, num_relation=num_relation, cumsum=cumsum)

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        cumsums = []
        num_relation = -1
        num_cum_node = 0
        num_cum_edge = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            cumsums.append(graph.cumsum)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = v + num_cum_node
                    elif type == "edge reference":
                        v = v + num_cum_edge
                    elif type == "graph reference":
                        v = v + num_graph
                data_dict[k].append(v)
            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError("Inconsistent `num_relation` in graphs. Expect %d but got %d."
                                 % (num_relation, graph.num_relation))
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_relation=num_relation,
                               num_nodes=num_nodes, num_edges=num_edges, meta_dict=meta_dict,
                               cumsums=cumsums, **data_dict)

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _standarize_unit_link(self, unit_type, link_type):
        if unit_type is None:
            raise ValueError("`unit_type` should be provided")
        if link_type is None:
            raise ValueError("`link_type` should be provided")

        unit_type = torch.as_tensor(unit_type, dtype=torch.long, device=self.device)
        link_type = torch.as_tensor(link_type, dtype=torch.long, device=self.device)
        return unit_type, link_type

    def unit(self):
        """
        Context manager for unit attributes.
        """
        return self.node()

    def link(self):
        """
        Context manager for link attributes.
        """
        return self.edge()

    @property
    def num_node(self):
        return self.num_unit

    @num_node.setter
    def num_node(self, value):
        self.num_unit = value

    @property
    def num_edge(self):
        return self.num_link

    @num_edge.setter
    def num_edge(self, value):
        self.num_link = value

    unit2graph = Graph.node2graph
    link2graph = Graph.edge2graph

    @property
    def node_feature(self):
        return self.unit_feature

    @node_feature.setter
    def node_feature(self, value):
        self.unit_feature = value

    @property
    def edge_feature(self):
        return self.link_feature

    @edge_feature.setter
    def edge_feature(self, value):
        self.link_feature = value

    @property
    def graph_feature(self):
        return self.glycan_feature

    @graph_feature.setter
    def graph_feature(self, value):
        self.glycan_feature = value


class PackedHeterogeneousAllAtomGlycan(PackedGraph, HeterogeneousAllAtomGlycan):

    unpacked_type = HeterogeneousAllAtomGlycan
    unit2graph = PackedGraph.node2graph
    link2graph = PackedGraph.edge2graph
    _check_attribute = HeterogeneousAllAtomGlycan._check_attribute

    def __init__(self, edge_list=None, unit_type=None, link_type=None,
                 num_nodes=None, num_edges=None, offsets=None, cumsums=None, **kwargs):
        super(PackedHeterogeneousAllAtomGlycan, self).__init__(
            edge_list=edge_list, unit_type=unit_type, link_type=link_type,
            num_nodes=num_nodes, num_edges=num_edges, offsets=offsets, **kwargs
        )

        self.cumsums = cumsums

    @classmethod
    def from_iupac(cls, iupacs, unit_feature="default", link_feature="default",
                   atom_feature="default", bond_feature="default", **kwargs):
        """
        Create pack heterogeneous graph from a list of IUPAC-Condensed glycan sequences.
        """
        unit_feature = cls._standarize_option(unit_feature)
        link_feature = cls._standarize_option(link_feature)
        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)

        offset_atom_feature, offset_unit_feature = 0, 0
        glycan = Glycan.from_iupac("Man(a1-4)Man", unit_feature=unit_feature, link_feature=link_feature)
        offset_unit_feature = len(glycan.unit_feature[0])
        offset_link_feature = len(glycan.link_feature[0])
        _ext_atom = [0 for _ in range(offset_unit_feature)]
        _ext_bond = [0 for _ in range(offset_link_feature)]
        num_relation_glycan = glycan.num_relation

        smiles = cls.glycan2smiles["Gal"]
        mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
        offset_atom_feature = len(mol.atom_feature[0])
        offset_bond_feature = len(mol.bond_feature[0])
        _ext_unit = [0 for _ in range(offset_atom_feature)]
        _ext_link = [0 for _ in range(offset_bond_feature)]
        num_relation_atom = mol.num_relation

        cross_feature = [0 for _ in range(offset_bond_feature + offset_link_feature)] + [1]
        cross_type = cls.cross_link_id

        _edge_list = []
        _unit_type = []
        _link_type = []
        _unit_feature = []
        _link_feature = []
        _num_nodes = []
        _num_edges = []
        _cumsums = []

        for iupac in iupacs:
            glycan = Glycan.from_iupac(iupac, unit_feature=unit_feature, link_feature=link_feature)

            num_node_glycan = glycan.num_node
            num_node_atom = 0

            if glycan is None or num_node_glycan == 0:
                continue

            offset_unit_index = len(glycan.unit_type)
            cumsum = [offset_unit_index]

            all_unit_type = glycan.unit_type.tolist()
            all_unit_feature = [x + _ext_unit for x in glycan.unit_feature.tolist()]
            all_link_type = glycan.link_type.tolist()
            all_link_feature = [x + _ext_link + [0] for x in glycan.link_feature.tolist()]
            all_edge_list = glycan.edge_list.tolist()

            for index, unit_type_id in enumerate(glycan.unit_type.tolist()):
                mol = None
                glycan = Glycan.id2unit[unit_type_id]
                if glycan in cls.glycan2mol:
                    mol = cls.glycan2mol[glycan]
                else:
                    smiles = cls.glycan2smiles[glycan]
                    if smiles == "":
                        continue
                    mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
                    cls.glycan2mol[glycan] = mol
                if mol is None:
                    continue
                num_node_atom += mol.num_node

                _atom_type = [x + cls.offset_atom_id_domain for x in mol.atom_type.tolist()]
                all_unit_type.extend(_atom_type)

                _atom_feature = [_ext_atom + x for x in mol.atom_feature.tolist()]
                all_unit_feature.extend(_atom_feature)

                _bond_type = [x + cls.offset_bond_id_domain for x in mol.bond_type.tolist()]
                _bond_feature = [_ext_bond + x + [0] for x in mol.bond_feature.tolist()]

                all_link_type.extend(_bond_type)
                all_link_feature.extend(_bond_feature)

                for edge in mol.edge_list:
                    src_ = edge[0] + offset_unit_index
                    tgt_ = edge[1] + offset_unit_index
                    type_ = edge[2] + cls.offset_bond_id_domain
                    all_edge_list.append([src_, tgt_, type_])

                for atom_index in range(len(mol.atom_type)):
                    all_edge_list.extend([[index, atom_index + offset_unit_index, cross_type],
                                          [atom_index + offset_unit_index, index, cross_type]])
                    for _ in range(2):
                        all_link_type.append(cross_type)
                        all_link_feature.append(cross_feature)

                offset_unit_index += len(mol.atom_type)
                cumsum.append(offset_unit_index)

            _edge_list.extend(all_edge_list)
            _unit_type.extend(all_unit_type)
            _link_type.extend(all_link_type)
            _unit_feature.extend(all_unit_feature)
            _link_feature.extend(all_link_feature)
            _cumsums.append(cumsum)

            _num_nodes.append(num_node_atom + num_node_glycan)
            _num_edges.append(len(all_edge_list))

        _edge_list = torch.tensor(_edge_list)
        _unit_type = torch.tensor(_unit_type)
        _link_type = torch.tensor(_link_type)
        _unit_feature = torch.tensor(_unit_feature) if len(_unit_feature) > 0 else None
        _link_feature = torch.tensor(_link_feature) if len(_link_feature) > 0 else None

        num_relation = num_relation_glycan + num_relation_atom + 1

        return cls(edge_list=_edge_list, unit_type=_unit_type, link_type=_link_type,
                   unit_feature=_unit_feature, link_feature=_link_feature,
                   num_nodes=_num_nodes, num_edges=_num_edges, num_relation=num_relation, cumsum=_cumsums)

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Links are undirected relations, but `add_inverse` is specified")
        return super(PackedHeterogeneousAllAtomGlycan, self).undirected(add_inverse)

    @property
    def num_nodes(self):
        return self.num_units

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_units = value

    @property
    def num_edges(self):
        return self.num_links

    @num_edges.setter
    def num_edges(self, value):
        self.num_links = value

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            cumsums = [torch.as_tensor(x, *args, **kwargs) for x in self.cumsums]
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges,
                              num_relation=self.num_relation, offsets=self._offsets,
                              meta_dict=self.meta_dict, cumsums=cumsums,
                              **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges,
                              num_relation=self.num_relation, offsets=self._offsets, meta_dict=self.meta_dict,
                              **utils.cpu(self.data_dict))

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_units=%s" % pretty.long_array(self.num_units.tolist()),
                  "num_links=%s" % pretty.long_array(self.num_links.tolist())]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


class BiAllAtomGlycan(Graph):
    """
    Implementation of glycans graph structure proposed in
    `Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains`_.

    .. _Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains:
        https://arxiv.org/pdf/2402.01481
    """

    glycan2smiles = {'dHex': '', 'Gal': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'AltN': 'O1C(O)[C@H](N)[C@@H](O)[C@@H](O)[C@@H]1CO',
                     'FucN': 'O1C(O)[C@@H](N)[C@H](O)[C@H](O)[C@@H]1C',
                     'Rha': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@@H]1C',
                     'GalNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Glc': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'GlcN': 'O1C(O)[C@H](N)[C@@H](O)[C@H](O)[C@H]1CO',
                     'GlcNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Man': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'ManNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Xyl': 'O1C[C@@H](O)[C@H](O)[C@@H](O)C1O', 'Monosaccharide': '',
                     'Alt': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1CO', 'Ery-ol': 'OC[C@H](O)[C@H](O)CO',
                     'All': 'O1C(O)[C@H](O)[C@H](O)[C@H](O)[C@H]1CO', 'Rib': 'O1C[C@@H](O)[C@@H](O)[C@@H](O)C1O',
                     'Tal': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Gul': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Kdo': 'O1[C@H]([C@H](O)CO)[C@H](O)[C@H](O)CC1(O)C(=O)O',
                     'Kdof': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](O)CC1(O)C(=O)O',
                     'Fruf': 'O1[C@H](CO)[C@@H](O)[C@H](O)C1(O)CO', 'Fuc': 'O1C(O)[C@@H](O)[C@H](O)[C@H](O)[C@@H]1C',
                     'LyxHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(O)CO',
                     'Neu5Ac': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](NC(C)=O)[C@@H](O)CC1(O)C(=O)O',
                     'Ara': 'O1C[C@H](O)[C@H](O)[C@@H](O)C1O', 'HexNAc': 'O1C(O)C(NC(C)=O)C(O)C(O)C1CO',
                     '4eLeg': 'O1[C@@H]([C@H](N)[C@@H](C)O)[C@H](N)[C@H](O)CC1(O)C(=O)O',
                     'Araf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1CO', 'Lyxf': 'O1C(O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     '6dAlt': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1C',
                     '6dAltNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@@H]1C',
                     '6dGul': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1C', 'dHexNAc': '',
                     'ManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(O)CO',
                     '6dTal': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1C',
                     'TalNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'Aci': 'O1[C@@H]([C@@H](N)[C@H](C)O)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Leg': 'O1[C@@H]([C@H](N)[C@@H](C)O)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Abe': 'O1C(O)[C@H](O)C[C@@H](O)[C@H]1C',
                     'AcoNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](OC)[C@@H](O)[C@@H]1C',
                     'AllN': 'O1C(O)[C@H](N)[C@H](O)[C@H](O)[C@H]1CO',
                     'AllNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@H](O)[C@H]1CO',
                     'AltA': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1C(=O)O', 'Api': 'O1C[C@](O)(CO)[C@@H](O)C1O',
                     'Apif': 'O1C[C@](O)(CO)[C@@H](O)C1O', 'AraN': 'O1C[C@H](O)[C@H](O)[C@@H](N)C1O',
                     'Asc': 'O1C(O)[C@H](O)C[C@@H](O)[C@@H]1C', 'Bac': 'O1C(O)[C@H](N)[C@@H](O)[C@H](N)[C@H]1C',
                     'Col': 'O1C(O)[C@@H](O)C[C@H](O)[C@@H]1C', 'HexA': 'O1C(O)C(O)C(O)C(O)C1C(=O)O',
                     'Hex': 'O1C(O)C(O)C(O)C(O)C1CO', 'HexN': 'O1C(O)C(N)C(O)C(O)C1CO',
                     'IdoHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1C(O)CO', 'Lyx': 'O1C[C@@H](O)[C@H](O)[C@H](O)C1O',
                     'FucNAc': 'O1C(O)[C@@H](NC(C)=O)[C@H](O)[C@H](O)[C@@H]1C',
                     'Fucf': 'O1C(O)[C@@H](O)[C@H](O)[C@H]1[C@H](C)O',
                     'FucfNAc': 'O1C(O)[C@@H](NC(C)=O)[C@H](O)[C@H]1[C@H](C)O',
                     'Ido': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1CO',
                     'IdoA': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@@H]1C(=O)O',
                     'IdoNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@@H]1CO',
                     'RhaN': 'O1C(O)[C@H](N)[C@H](O)[C@@H](O)[C@@H]1C',
                     'RhaNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@@H](O)[C@@H]1C',
                     'Sor': 'O1C[C@H](O)[C@@H](O)[C@H](O)C1(O)CO', 'Thre-ol': 'OC[C@@H](O)[C@H](O)CO',
                     'DDAltHep': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GalHep': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1C(O)CO',
                     'DDGlcHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO',
                     'GalN': 'O1C(O)[C@H](N)[C@@H](O)[C@@H](O)[C@H]1CO',
                     'DLGlcHep': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO',
                     'Dha': 'O1[C@H](C(=O)O)[C@H](O)[C@H](O)CC1(O)C(=O)O', 'Dig': 'O1C(O)C[C@H](O)[C@H](O)[C@H]1C',
                     'Erwiniose': 'O1C(O)[C@H](O)C[C@](O)([C@@H](O)C[C@@H](C)O)[C@H]1C',
                     'Fru': 'O1C[C@@H](O)[C@@H](O)[C@H](O)C1(O)CO',
                     'Fus': 'O1[C@@H]([C@@H](O)[C@H](C)O)[C@@H](N)[C@H](O)CC1(O)C(=O)O',
                     'GalA': 'O1C(O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1C(=O)O',
                     'Galf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GalfNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H]1[C@H](O)CO',
                     'GlcA': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1C(=O)O',
                     'GulA': 'O1C(O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1C(=O)O',
                     'GulN': 'O1C(O)[C@H](N)[C@H](O)[C@@H](O)[C@H]1CO',
                     'GulNAc': 'O1C(O)[C@H](NC(C)=O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Ins': 'O[C@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O',
                     'Kdn': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](O)[C@@H](O)CC1(O)C(=O)O',
                     'Ko': 'O1[C@H]([C@H](O)CO)[C@H](O)[C@H](O)[C@H](O)C1(O)C(=O)O',
                     'Oli': 'O1C(O)C[C@@H](O)[C@H](O)[C@H]1C', 'Psi': 'O1C[C@@H](O)[C@@H](O)[C@@H](O)C1(O)CO',
                     'Qui': 'O1C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1C', 'QuiN': 'O1C(O)[C@H](N)[C@@H](O)[C@H](O)[C@H]1C',
                     'QuiNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H](O)[C@H]1C', 'Ribf': 'O1C(O)[C@H](O)[C@H](O)[C@H]1CO',
                     'ManA': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1C(=O)O',
                     'ManN': 'O1C(O)[C@@H](N)[C@@H](O)[C@H](O)[C@H]1CO',
                     'Manf': 'O1C(O)[C@@H](O)[C@@H](O)[C@H]1[C@H](O)CO',
                     'Mur': 'O1C(O)[C@H](N)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     'MurNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     'Neu': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Neu5Gc': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](NC(=O)CO)[C@@H](O)CC1(O)C(=O)O',
                     'Neu4Ac': 'O1[C@@H]([C@H](O)[C@H](O)CO)[C@H](N)[C@@H](OC(C)=O)CC1(O)C(=O)O',
                     'Par': 'O1C(O)[C@H](O)C[C@H](O)[C@H]1C', 'Pau': 'O1C(O)C[C@@H](O)[C@@H](O)[C@H]1C',
                     'Pse': 'O1[C@@H]([C@@H](N)[C@H](C)O)[C@@H](N)[C@@H](O)CC1(O)C(=O)O',
                     'Sedf': 'O1[C@H]([C@H](O)CO)[C@@H](O)[C@H](O)C1(O)CO',
                     'Sia': 'O1C(C(O)C(O)CO)C(NC(C)=O)C(O)CC1(O)C(=O)O', 'Tag': 'O1C[C@@H](O)[C@H](O)[C@H](O)C1(O)CO',
                     'TalA': 'O1C(O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1C(=O)O', 'Tyv': 'O1C(O)[C@@H](O)C[C@H](O)[C@H]1C',
                     'Xluf': 'O1C[C@@H](O)[C@H](O)C1(O)CO', 'Xylf': 'O1C(O)[C@H](O)[C@@H](O)[C@H]1CO',
                     'Yer': 'O1C(O)[C@H](O)C[C@](O)([C@H](C)O)[C@H]1C',
                     'ManfNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@H]1[C@H](O)CO', 'Unknown': '',
                     'LDManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1[C@@H](O)CO',
                     'AllA': 'O1C(O)[C@H](O)[C@H](O)[C@H](O)[C@H]1C(=O)O', 'ddHex': '',
                     'DDManHep': 'O1C(O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1[C@H](O)CO', 'ddNon': '', 'Assigned': '',
                     'GlcfNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@H]1[C@H](O)CO',
                     'MurNGc': 'O1C(O)[C@H](NC(=O)CO)[C@@H](O[C@H](C)C(=O)O)[C@H](O)[C@H]1CO',
                     '6dTalNAc': 'O1C(O)[C@@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@H]1C',
                     'TalN': 'O1C(O)[C@@H](N)[C@@H](O)[C@@H](O)[C@H]1CO', 'dNon': '',
                     '6dAltf': 'O1C(O)[C@H](O)[C@@H](O)[C@@H]1[C@H](C)O',
                     'AltNAc': 'O1C(O)[C@H](NC(C)=O)[C@@H](O)[C@@H](O)[C@@H]1CO', 'IdoNGlcf': '',
                     'Aco': 'O1C(O)[C@H](O)[C@H](OC)[C@@H](O)[C@@H]1C'}
    glycan2mol = {}

    offset_atom_id_domain = len(Glycan.units)
    offset_bond_id_domain = len(Glycan.links)
    length_bond_id_domain = len(Molecule.bond2id)

    def __init__(self, edge_list=None, unit_type=None, link_type=None,
                 unit_feature=None, link_feature=None, cumsum=None, **kwargs):
        super().__init__(edge_list=edge_list, **kwargs)
        unit_type, link_type = self._standarize_unit_link(unit_type, link_type)

        self.cumsum = cumsum

        with self.unit():
            if unit_feature is not None:
                self.unit_feature = torch.as_tensor(unit_feature, device=self.device)
            self.unit_type = unit_type

        with self.link():
            if link_feature is not None:
                self.link_feature = torch.as_tensor(link_feature, device=self.device)
            self.link_type = link_type

    @classmethod
    def from_iupac(cls, iupac, unit_feature="default", link_feature="default",
                   atom_feature="default", bond_feature="default", **kwargs):

        """
        Create heterogeneous graph with backbone atoms from IUAPC-Condensed String.
        """

        unit_feature = cls._standarize_option(unit_feature)
        link_feature = cls._standarize_option(link_feature)
        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)

        glycan = Glycan.from_iupac("Man(a1-4)Man", unit_feature=unit_feature, link_feature=link_feature)
        offset_unit_feature = len(glycan.unit_feature[0])
        offset_link_feature = len(glycan.link_feature[0])
        _ext_atom = [0 for _ in range(offset_unit_feature)]
        _ext_bond = [0 for _ in range(offset_link_feature)]
        num_relation_glycan = glycan.num_relation

        smiles = cls.glycan2smiles["Gal"]
        mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
        offset_atom_feature = len(mol.atom_feature[0])
        offset_bond_feature = len(mol.bond_feature[0])
        _ext_unit = [0 for _ in range(offset_atom_feature)]
        _ext_link = [0 for _ in range(offset_bond_feature)]
        num_relation_atom = mol.num_relation

        glycan = Glycan.from_iupac(iupac, unit_feature=unit_feature, link_feature=link_feature)

        num_node_glycan = glycan.num_node
        num_node_atom = 0

        if glycan is None or num_node_glycan == 0:
            return None

        offset_unit_index = 0
        cumsum = [offset_unit_index]

        all_unit_type = list()
        all_unit_feature = list()
        all_link_type = list()
        all_link_feature = list()
        edge_list = list()

        backbone_atoms = dict()

        for index, (unit_type, unit_feature) in enumerate(zip(glycan.unit_type.tolist(), glycan.unit_feature.tolist())):
            mol = None
            sub_glycan = Glycan.id2unit[unit_type]
            if sub_glycan in cls.glycan2mol:
                mol = cls.glycan2mol[sub_glycan]
            else:
                smiles = cls.glycan2smiles[sub_glycan]
                if smiles != "":
                    mol = Molecule.from_smiles(smiles, atom_feature=atom_feature, bond_feature=bond_feature, **kwargs)
                    cls.glycan2mol[sub_glycan] = mol
            if mol is None:
                backbone_atoms[index] = offset_unit_index

                all_unit_type.append(unit_type)
                all_unit_feature.append(unit_feature + _ext_unit)

                offset_unit_index += 1
                cumsum.append(offset_unit_index)
                continue

            num_node_atom += mol.num_node - 1
            backbone_atom_index = 0
            backbone_atoms[index] = backbone_atom_index + offset_unit_index

            _atom_type = [unit_type if i == 0 else x + cls.offset_atom_id_domain for i, x in enumerate(mol.atom_type.tolist())]
            all_unit_type.extend(_atom_type)

            _atom_feature = [unit_feature + _ext_unit if i == 0 else _ext_atom + x for i, x in enumerate(mol.atom_feature.tolist())]
            all_unit_feature.extend(_atom_feature)

            _bond_type = [x + cls.offset_bond_id_domain for x in mol.bond_type.tolist()]
            all_link_type.extend(_bond_type)

            _bond_feature = [_ext_bond + x for x in mol.bond_feature.tolist()]
            all_link_feature.extend(_bond_feature)

            for edge in mol.edge_list:
                src_ = edge[0] + offset_unit_index
                tgt_ = edge[1] + offset_unit_index
                type_ = edge[2] + cls.offset_bond_id_domain
                edge_list.append([src_, tgt_, type_])

            offset_unit_index += len(mol.atom_type)
            cumsum.append(offset_unit_index)

        for edge in glycan.edge_list.tolist():
            src_glycan, tgt_glycan, edge_type = edge
            src_atom = backbone_atoms[src_glycan]
            tgt_atom = backbone_atoms[tgt_glycan]
            edge_list.append([src_atom, tgt_atom, edge_type])
        
        all_link_type.extend(glycan.link_type.tolist())
        all_link_feature.extend([x + _ext_link for x in glycan.link_feature.tolist()])
        
        all_unit_type = torch.tensor(all_unit_type)
        all_link_type = torch.tensor(all_link_type)
        if len(all_unit_feature) > 0:
            all_unit_feature = torch.tensor(all_unit_feature)
        else:
            all_unit_feature = None
        if len(all_link_feature) > 0:
            all_link_feature = torch.tensor(all_link_feature)
        else:
            all_link_feature = None

        num_node = num_node_glycan + num_node_atom
        num_relation = num_relation_glycan + num_relation_atom

        return cls(edge_list, all_unit_type, all_link_type, all_unit_feature, all_link_feature,
                   num_node=num_node, num_relation=num_relation, cumsum=cumsum)

    @classmethod
    def pack(cls, graphs):
        edge_list = []
        edge_weight = []
        num_nodes = []
        num_edges = []
        cumsums = []
        num_relation = -1
        num_cum_node = 0
        num_cum_edge = 0
        num_graph = 0
        data_dict = defaultdict(list)
        meta_dict = graphs[0].meta_dict
        for graph in graphs:
            edge_list.append(graph.edge_list)
            edge_weight.append(graph.edge_weight)
            num_nodes.append(graph.num_node)
            num_edges.append(graph.num_edge)
            cumsums.append(graph.cumsum)
            for k, v in graph.data_dict.items():
                for type in meta_dict[k]:
                    if type == "graph":
                        v = v.unsqueeze(0)
                    elif type == "node reference":
                        v = v + num_cum_node
                    elif type == "edge reference":
                        v = v + num_cum_edge
                    elif type == "graph reference":
                        v = v + num_graph
                data_dict[k].append(v)
            if num_relation == -1:
                num_relation = graph.num_relation
            elif num_relation != graph.num_relation:
                raise ValueError("Inconsistent `num_relation` in graphs. Expect %d but got %d."
                                 % (num_relation, graph.num_relation))
            num_cum_node += graph.num_node
            num_cum_edge += graph.num_edge
            num_graph += 1

        edge_list = torch.cat(edge_list)
        edge_weight = torch.cat(edge_weight)
        data_dict = {k: torch.cat(v) for k, v in data_dict.items()}

        return cls.packed_type(edge_list, edge_weight=edge_weight, num_relation=num_relation,
                               num_nodes=num_nodes, num_edges=num_edges, meta_dict=meta_dict,
                               cumsums=cumsums, **data_dict)

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _standarize_unit_link(self, unit_type, link_type):
        if unit_type is None:
            raise ValueError("`unit_type` should be provided")
        if link_type is None:
            raise ValueError("`link_type` should be provided")

        unit_type = torch.as_tensor(unit_type, dtype=torch.long, device=self.device)
        link_type = torch.as_tensor(link_type, dtype=torch.long, device=self.device)
        return unit_type, link_type

    def unit(self):
        """
        Context manager for unit attributes.
        """
        return self.node()

    def link(self):
        """
        Context manager for link attributes.
        """
        return self.edge()

    @property
    def num_node(self):
        return self.num_unit

    @num_node.setter
    def num_node(self, value):
        self.num_unit = value

    @property
    def num_edge(self):
        return self.num_link

    @num_edge.setter
    def num_edge(self, value):
        self.num_link = value

    unit2graph = Graph.node2graph
    link2graph = Graph.edge2graph

    @property
    def node_feature(self):
        return self.unit_feature

    @node_feature.setter
    def node_feature(self, value):
        self.unit_feature = value

    @property
    def edge_feature(self):
        return self.link_feature

    @edge_feature.setter
    def edge_feature(self, value):
        self.link_feature = value

    @property
    def graph_feature(self):
        return self.glycan_feature

    @graph_feature.setter
    def graph_feature(self, value):
        self.glycan_feature = value


class PackedBiAllAtomGlycan(PackedGraph, BiAllAtomGlycan):
    """
    Packed version of BiAllAtomGlycan.
    """

    unpacked_type = BiAllAtomGlycan
    unit2graph = PackedGraph.node2graph
    link2graph = PackedGraph.edge2graph
    _check_attribute = BiAllAtomGlycan._check_attribute

    def __init__(self, edge_list=None, unit_type=None, link_type=None,
                 num_nodes=None, num_edges=None, offsets=None, cumsums=None, **kwargs):
        super(PackedBiAllAtomGlycan, self).__init__(
            edge_list=edge_list, unit_type=unit_type, link_type=link_type,
            num_nodes=num_nodes, num_edges=num_edges, offsets=offsets, **kwargs
        )

        self.cumsums = cumsums

    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Links are undirected relations, but `add_inverse` is specified")
        return super(PackedBiAllAtomGlycan, self).undirected(add_inverse)

    @property
    def num_nodes(self):
        return self.num_units

    @num_nodes.setter
    def num_nodes(self, value):
        self.num_units = value

    @property
    def num_edges(self):
        return self.num_links

    @num_edges.setter
    def num_edges(self, value):
        self.num_links = value

    def cuda(self, *args, **kwargs):
        edge_list = self.edge_list.cuda(*args, **kwargs)

        if edge_list is self.edge_list:
            return self
        else:
            cumsums = [torch.as_tensor(x, *args, **kwargs) for x in self.cumsums]
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges,
                              num_relation=self.num_relation, offsets=self._offsets,
                              meta_dict=self.meta_dict, cumsums=cumsums,
                              **utils.cuda(self.data_dict, *args, **kwargs))

    def cpu(self):
        edge_list = self.edge_list.cpu()

        if edge_list is self.edge_list:
            return self
        else:
            return type(self)(edge_list, edge_weight=self.edge_weight,
                              num_nodes=self.num_nodes, num_edges=self.num_edges,
                              num_relation=self.num_relation, offsets=self._offsets, meta_dict=self.meta_dict,
                              **utils.cpu(self.data_dict))

    def __repr__(self):
        fields = ["batch_size=%d" % self.batch_size,
                  "num_units=%s" % pretty.long_array(self.num_units.tolist()),
                  "num_links=%s" % pretty.long_array(self.num_links.tolist())]
        if self.device.type != "cpu":
            fields.append("device='%s'" % self.device)
        return "%s(%s)" % (self.__class__.__name__, ", ".join(fields))


AllAtomGlycan.packed_type = PackedAllAtomGlycan
HeterogeneousAllAtomGlycan.packed_type = PackedHeterogeneousAllAtomGlycan
BiAllAtomGlycan.packed_type = PackedBiAllAtomGlycan
