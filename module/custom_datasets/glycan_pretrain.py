import os
import sys
import csv
import logging
import warnings
from collections import defaultdict

from tqdm import tqdm

from torch.utils import data as torch_data
from torchdrug.core import Registry as R

from torchdrug import core, utils

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.custom_data import glycan, all_atom_glycan

logger = logging.getLogger(__name__)


@R.register("datasets.GlycanPretrain")
class GlycanPretrainDataset(torch_data.Dataset, core.Configurable):
    """
    Glycan Dataset filtered from GlyTouCan for pretrain usage.

    Statistics: 40,781
    """
    url = "https://torchglycan.s3.us-east-2.amazonaws.com/pretrain/unlabeled_glycan.csv"
    md5 = "814b372d72c9931d4d5cf2c3104d8238"
    target_fields = []

    def __init__(self, path, view, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path
        self.csv_file = os.path.join(path, "glycan_pretrain.csv")
        if not os.path.exists(self.csv_file):
            self.csv_file = utils.download(self.url, path, md5=self.md5)
        self.view = view

        if self.view not in {'glycan', 'atom-glycan', 'bi'}:
            raise ValueError(f"Expect [\'glycan\', \'atom-glycan\', \'bi\'] in pre-training phase, but found {view}")

        self.load_csv(self.csv_file, iupac_field="IUPAC Condensed", target_fields=self.target_fields,
                      valid_field='valid_final', verbose=verbose, **kwargs)

    def __len__(self):
        return len(self.data)

    def load_iupac(self, iupac_list, targets, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from IUPAC and targets
        """
        num_sample = len(iupac_list)
        if num_sample > 1000000:
            warnings.warn("Preprocessing molecules of a large dataset consumes a lot of CPU memory and time. "
                          "Use load_smiles(lazy=True) to construct glycan in the dataloader instead.")
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with the number of glycan. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.iupac_list = []
        self.data = []
        self.targets = defaultdict(list)

        if verbose:
            iupac_list = tqdm(iupac_list, "Constructing glycan from iupac condensed data")
        for i, iupac in enumerate(iupac_list):
            try:
                mol = None
                if not self.lazy or len(self.data) == 0:
                    if self.view == "glycan":
                        mol = glycan.Glycan.from_iupac(iupac, **kwargs)
                    elif self.view == "atom-glycan":
                        mol = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(iupac, **kwargs)
                    elif self.view == "bi":
                        mol = all_atom_glycan.BiAllAtomGlycan.from_iupac(iupac, **kwargs)
                    else:
                        raise ValueError(f"Unknown View {self.view}")
                    
                if mol is None:
                    continue

                self.data.append(mol)
                self.iupac_list.append(iupac)
                for field in targets:
                    self.targets[field].append(targets[field][i])
            except KeyError:
                continue

    def load_csv(self, csv_file, iupac_field="IUPAC Condensed", valid_field=None, target_fields=None, verbose=0, **kwargs):
        """
        Load the dataset from a CSV file.

        Parameters:
            csv_file (str): file name
            iupac_field (str, optional): name of the iupac condensed column in the table.
                Use ``None`` if there is no iupac column.
            valid_field (str, optional): name of the valid column in the table, recommend using ``valid_final``. (compulsory when view == atom-glycan)
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the iupac column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            header = next(reader)
            iupac = []
            targets = defaultdict(list)
            index_of_iupac_condensed = header.index(iupac_field)

            if valid_field not in ['valid', 'valid_with_question', 'valid_vocab', 'valid_no_leak', 'valid_final']:
                raise ValueError(f"Valid field {valid_field} not support, please choose from \
                                 [\'valid\', \'valid_with_question\', \'valid_vocab\', \'valid_no_leak\', \'valid_final\'].")
            index_of_valid = header.index(valid_field)

            for values in reader:
                iupac_condensed = values[index_of_iupac_condensed].strip()
                
                if not any(iupac_condensed):
                    continue
                if values[index_of_valid] != '1':
                    continue

                if iupac_condensed is not None:
                    iupac.append(iupac_condensed)

        targets = {}
        self.load_iupac(iupac, targets, verbose=verbose, **kwargs)

    def get_item(self, index):
        """
        Get the i-th sample.

        Parameters:
            index (int): index of the sample
        """
        cur_glycan = self.data[index]

        if self.lazy and cur_glycan is None:
            iupac = self.iupac_list[index]
            if self.view == "glycan":
                cur_glycan = glycan.Glycan.from_iupac(iupac, **self.kwargs)
            elif self.view == "atom":
                cur_glycan = all_atom_glycan.AllAtomGlycan.from_iupac(iupac, **self.kwargs)
            elif self.view == "atom-glycan":
                cur_glycan = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(iupac, **self.kwargs)
            elif self.view == "bi":
                cur_glycan = all_atom_glycan.BiAllAtomGlycan.from_iupac(iupac, **self.kwargs)
            else:
                raise ValueError(f"Unknown View {self.view}")
            self.data[index] = cur_glycan

        item = {"graph": cur_glycan}
        target = {field: self.targets[field][index] for field in self.target_fields}
        item.update(target)
        if self.transform:
            item = self.transform(item)

        return item

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
