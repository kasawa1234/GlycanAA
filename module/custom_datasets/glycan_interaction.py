import os
import sys
import csv
import logging
import warnings
import inspect
import copy
from collections import defaultdict

from tqdm import tqdm

from torchdrug import core, data, utils
from torch.utils import data as torch_data
from torchdrug.core import Registry as R

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.custom_data import glycan, all_atom_glycan

logger = logging.getLogger(__name__)


@R.register("datasets.ProteinGlycanInteraction")
class ProteinGlycanInteraction(torch_data.Dataset, core.Configurable):
    """
    Protein-Glycan Interaction Dataset.

    Statistics:
    - #Train: 442,396
    - #Valid: 58,887
    - #Test: 63,364
    """
    url = "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_interaction.csv"
    md5 = "6a0e55b2441b26088b75cf1d6d2a5b52"
    target_fields = ['interaction']
    splits = ["train", "valid", "test"]

    def __init__(self, path, view, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.csv_file = os.path.join(path, "glycan_interaction.csv")
        if not os.path.exists(self.csv_file):
            self.csv_file = utils.download(self.url, path, md5=self.md5)

        self.view = view
        if view == "atom":
            self.csv_with_smiles_file = os.path.join(path, "glycan_interaction_with_smiles.csv")
            if not os.path.exists(self.csv_with_smiles_file):
                self.generate_csv_with_mol(verbose, **kwargs)
            self.load_csv(self.csv_with_smiles_file, iupac_field="target", target_fields=self.target_fields,
                        verbose=verbose, **kwargs)
        elif view == "glycan" or view == "atom-glycan" or view == "bi":
            self.load_csv(self.csv_file, sequence_field="target", verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Expect view in [\'glycan\', \'atom\', \'atom-gycan\', \'bi\'], but found {view}")
    
    def load_sequence(self, sequences, iupac_list, smiles_list, targets, sample_splits, attributes=None, 
                      transform=None, fast=True, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from sequences and targets.
        """ 
        num_sample = len(sequences)
        if num_sample > 1000000:
            warnings.warn("Preprocessing molecules of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_sequence(lazy=True) to construct molecules in the dataloader instead.")
        if len(iupac_list) != num_sample:
            raise ValueError("Number of iupacs doesn't match with the number of proteins. "
                            "Expect %d but found %d" % (num_sample, len(iupac_list)))
        if self.view == 'atom' and len(smiles_list) != num_sample:
            raise ValueError("Number of smiles doesn't match with the number of proteins. "
                            "Expect %d but found %d" % (num_sample, len(iupac_list)))            
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with the number of proteins. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.transform = transform
        self.fast = fast
        self.lazy = lazy
        self.sequences = []
        self.iupac_list = []
        self.smiles_list = []
        self.data = []
        self.sample_splits = []
        self.targets = defaultdict(list)
        self.target_fields = list(targets.keys())
        self.seq2obj = dict()
        self.iupac2obj = dict()

        p_params = inspect.signature(data.Protein.from_sequence).parameters
        g_params = inspect.signature(glycan.Glycan.from_iupac).parameters
        self.protein_kwargs = {k: v for k, v in kwargs.items() if k in p_params}
        self.glycan_kwargs = {k: v for k, v in kwargs.items() if k in g_params}
        if self.view == 'atom-glycan' or self.view == 'bi':
            glycan_special_params = ["atom_feature_glycan", "bond_feature_glycan"]
            self.glycan_kwargs['atom_feature'] = kwargs.pop(glycan_special_params[0])
            self.glycan_kwargs['bond_feature'] = kwargs.pop(glycan_special_params[1])

        if verbose:
            sequences = tqdm(sequences, "Constructing proteins and glycans")
        for i, sequence in enumerate(sequences):
            if self.fast or not self.lazy or len(self.data) == 0:
                cur_protein = self.seq2obj[sequence] if self.fast and sequence in self.seq2obj \
                                else data.Protein.from_sequence(sequence, **self.protein_kwargs)
                cur_glycan = None
                if self.fast and iupac_list[i] in self.iupac2obj:
                    # for view='atom', smiles will also use iupac list to index
                    cur_glycan = self.iupac2obj[iupac_list[i]]
                else:
                    if self.view == "glycan":
                        cur_glycan = glycan.Glycan.from_iupac(iupac_list[i], **self.glycan_kwargs)
                    elif self.view == "atom":
                        cur_glycan = all_atom_glycan.AllAtomGlycan.from_smiles_new(smiles_list[i], **self.glycan_kwargs)
                    elif self.view == "atom-glycan":
                        cur_glycan = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(iupac_list[i], **self.glycan_kwargs)
                    elif self.view == "bi":
                        cur_glycan = all_atom_glycan.BiAllAtomGlycan.from_iupac(iupac_list[i], **self.glycan_kwargs)
                    else:
                        raise ValueError(f"Unknown View {self.view}")
                if cur_glycan is None:
                    continue
                if attributes is not None:
                    with cur_protein.graph():
                        for field in attributes:
                            setattr(cur_protein, field, attributes[field][i])
                if self.fast:
                    self.seq2obj.setdefault(sequence, cur_protein)
                    self.iupac2obj.setdefault(iupac_list[i], cur_glycan)
                self.data.append([cur_protein, cur_glycan])   
            else:
                self.data.append(None)
            self.sequences.append(sequence)
            self.iupac_list.append(iupac_list[i])
            if self.view == 'atom':
                self.smiles_list.append(smiles_list[i])
            if sample_splits:
                self.sample_splits.append(sample_splits[i])
            for field in targets:
                self.targets[field].append(targets[field][i])

    def load_csv(self, csv_file, sequence_field="target", target_field="interaction", lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from a CSV file.

        Parameters:
            csv_file (str): file name
            sequence_field (str, optional): name of the protein sequence column in the table
            target_field (str, optional): name of the output label
            lazy (bool, optional): if lazy mode is used, the proteins and glycans are processed in the dataloader.
                This may slow down the data loading process, but save a lot of CPU memory and dataset loading time.
            verbose (int, optional): output verbose level
            **kwargs
        """        
        sequences = []
        glycan_list = []
        smiles_list = []
        sample_splits = []
        targets = defaultdict(list)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))            
            fields = next(reader)
            if self.view == "atom":
                smiles = next(reader)
            for values in reader:
                if not any(values):
                    continue
                count = 0
                for idc, (field, value) in enumerate(zip(fields, values)):
                    if field == sequence_field:
                        current_sequence = value
                    elif field in self.splits:
                        if utils.literal_eval(value):
                            current_split = field
                    elif value:
                        if self.view == 'atom':
                            if smiles[idc] == "":
                                continue
                            smiles_list.append(smiles[idc])
                            glycan_list.append(field)
                        else:
                            glycan_list.append(field)
                        targets[target_field].append(utils.literal_eval(value))
                        count += 1
                sequences += ([current_sequence] * count)
                sample_splits += ([current_split] * count)
        self.load_sequence(sequences, glycan_list, smiles_list, targets, sample_splits, lazy=lazy, verbose=verbose, **kwargs)
    
    def split(self):
        """
        Get the train, valid and test split.
        """        
        train_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "train"]
        valid_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "valid"]
        test_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "test"]

        return [
            torch_data.Subset(self, train_indices),
            torch_data.Subset(self, valid_indices),
            torch_data.Subset(self, test_indices)
        ]
    
    def get_item(self, index):
        """
        Get the i-th sample.

        Parameters:
            index (int): index of the sample
        """
        sample = self.data[index]

        if self.lazy and not sample:
            cur_protein = data.Protein.from_sequence(self.sequences[index], **self.protein_kwargs)
            cur_glycan = None
            if self.view == "glycan":
                cur_glycan = glycan.Glycan.from_iupac(self.iupac_list[index], **self.glycan_kwargs)
            elif self.view == "atom":
                cur_glycan = all_atom_glycan.AllAtomGlycan.from_smiles_new(self.smiles_list[index], **self.glycan_kwargs)
            elif self.view == "atom-glycan":
                cur_glycan = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(self.iupac_list[index], **self.glycan_kwargs)
            elif self.view == "bi":
                cur_glycan = all_atom_glycan.BiAllAtomGlycan.from_iupac(self.iupac_list[index], **self.glycan_kwargs)
            else:
                raise ValueError(f"Unknown View {self.view}")
            self.data[index] = [cur_protein, cur_glycan]

        cur_protein = self.data[index][0]
        cur_glycan = self.data[index][1]
        item = {"graph1": cur_protein, "graph2": cur_glycan}
        target = {field: self.targets[field][index] for field in self.targets}
        item.update(target)
        if self.transform:
            item = self.transform(item)

        return item
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index
    
    def generate_csv_with_mol(self, verbose=0, **kwargs):
        logger.warning("Construct SMILES by GlyLES and saved in glycan_interaction_with_smiles.csv")

        with open(self.csv_file, "r", newline='') as fin:
            reader = csv.reader(fin)
            rows = list(reader)

            header = rows[0]
            atom_kwargs = copy.deepcopy(kwargs)
            atom_kwargs.pop('atom_feature')
            atom_kwargs.pop('bond_feature')

            smiles_sequences = []
            iupacs = tqdm(header[: -4]) if verbose else header[: -4]
            smiles_sequences = [all_atom_glycan.AllAtomGlycan.from_iupac(iupac, **atom_kwargs)[1] for iupac in iupacs]
            
            new_row = smiles_sequences + ['' for _ in range(4)]
            rows.insert(1, new_row)

        with open(self.csv_with_smiles_file, "w", newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(rows)
