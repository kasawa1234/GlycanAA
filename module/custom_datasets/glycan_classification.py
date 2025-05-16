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


@R.register("datasets.GlycanClassification")
class GlycanClassificationDataset(torch_data.Dataset, core.Configurable):
    """
    Glycan Dataset.

    Statistics:
    - #Train: 11,010
    - #Valid: 1,280
    - #Test: 919
    - Classification Tasks: 8
    """

    url = "https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_classification.csv"
    md5 = "d8f9b3a73d274936f39eaa7ae4f4a32b"
    target_fields = ["species", "genus", "family", "order", "class", "phylum", "kingdom", "domain"]
    splits = ["train", "validation", "test"]

    def __init__(self, path, view, target_fields=None, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if target_fields is not None:
            self.target_fields = target_fields

        self.csv_file = os.path.join(path, "glycan_classification.csv")
        if not os.path.exists(self.csv_file):
            self.csv_file = utils.download(self.url, path, md5=self.md5)

        self.view = view
        if view == "atom":
            self.csv_with_smiles_file = os.path.join(path, "glycan_classification_with_smiles.csv")
            if not os.path.exists(self.csv_with_smiles_file) and view == "atom":
                self.generate_csv_with_mol(verbose=verbose, **kwargs)
            self.load_csv(self.csv_with_smiles_file, iupac_field="target", target_fields=self.target_fields,
                        verbose=verbose, **kwargs)
        elif view == "glycan" or view == "atom-glycan" or view == "bi":
            self.load_csv(self.csv_file, iupac_field="target", target_fields=self.target_fields,
                        verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Expect view in [\'glycan\', \'atom\', \'atom-gycan\', \'bi\'], but found {view}")


    def __len__(self):
        return len(self.data)

    def load_iupac(self, iupac_list, smiles_list, targets, sample_splits, transform=None, lazy=False, verbose=0,
                   **kwargs):
        """
        Load the dataset from IUPAC and targets.
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
        self.sample_splits = []
        self.targets = defaultdict(list)

        if verbose:
            iupac_list = tqdm(iupac_list, "Constructing glycan from iupac condensed data")
        for i, iupac in enumerate(iupac_list):
            mol = None
            if not self.lazy or len(self.data) == 0:
                if self.view == "glycan":
                    mol = glycan.Glycan.from_iupac(iupac, **kwargs)
                elif self.view == "atom-glycan":
                    mol = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(iupac, **kwargs)
                elif self.view == "atom":
                    mol = all_atom_glycan.AllAtomGlycan.from_smiles_new(smiles_list[i], **kwargs)
                elif self.view == "bi":
                    mol = all_atom_glycan.BiAllAtomGlycan.from_iupac(iupac, **kwargs)
                else:
                    raise ValueError(f"Unknown View {self.view}")

            if self.view == "atom" and (mol is None or mol.num_atom == 0):
                continue

            self.data.append(mol)
            self.iupac_list.append(iupac)
            if sample_splits:
                self.sample_splits.append(sample_splits[i])
            for field in targets:
                self.targets[field].append(targets[field][i])

    def load_csv(self, csv_file, iupac_field="target", target_fields=None, verbose=0, **kwargs):
        """
        Load the dataset from a CSV file.

        Parameters:
            csv_file (str): file name
            iupac_field (str, optional): name of the iupac condensed column in the table.
                Use ``None`` if there is no iupac column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the iupac column.
            verbose (int, optional): output verbose level
            **kwargs
        """

        sample_splits = []
        self.field_mappings = {}
        self.unique_values = {}

        smiles_field = "smiles_sequence"

        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            iupac = []
            smiles = []
            targets = defaultdict(list)
            for field in fields:
                self.field_mappings[field] = {}
                self.unique_values[field] = set()

            for values in reader:
                if not any(values):
                    continue
                if iupac_field is None:
                    iupac.append("")
                for field, value in zip(fields, values):
                    if field == iupac_field:
                        iupac.append(value)
                    elif field == smiles_field:
                        smiles.append(value)
                    elif field in self.splits:
                        if utils.literal_eval(value):
                            current_split = field
                    elif target_fields is None or field in target_fields:
                        if value not in self.field_mappings[field]:
                            self.field_mappings[field][value] = len(self.field_mappings[field])
                        targets[field].append(self.field_mappings[field][value])
                        self.unique_values[field].add(value)

                sample_splits += [current_split]

        for field, values in self.unique_values.items():
            print(f"Number of unique values for {field}: {len(values)}")

        self.load_iupac(iupac, smiles, targets, sample_splits, verbose=verbose, **kwargs)

    def split(self):
        """
        Get the train, valid and test split.
        """
        train_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "train"]
        valid_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "validation"]
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
        cur_glycan = self.data[index]

        if self.lazy and cur_glycan is None:
            iupac = self.iupac_list[index]
            if self.view == "glycan":
                cur_glycan = glycan.Glycan.from_iupac(iupac, **self.kwargs)
            elif self.view == "atom-glycan":
                cur_glycan = all_atom_glycan.HeterogeneousAllAtomGlycan.from_iupac(iupac, **self.kwargs)
            elif self.view == "atom":
                cur_glycan, _ = all_atom_glycan.AllAtomGlycan.from_iupac(iupac, **self.kwargs)
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

    def generate_csv_with_mol(self, verbose=0, **kwargs):
        logger.warning(f"Construct SMILES by GlyLES and saved in {self.path}/glycan_classification_with_smiles.csv")

        with open(self.csv_with_smiles_file, "w", newline='') as fout:
            writer = csv.writer(fout)
            with open(self.csv_file, "r") as fin:
                reader = csv.reader(fin)
                if verbose:
                    reader = iter(tqdm(reader, "Loading %s" % self.csv_file, utils.get_line_count(self.csv_file)))
                fields = next(reader)
                writer.writerow(fields + ["smiles_sequence"])
                for values in reader:
                    if not any(values):
                        continue
                    iupac = values[fields.index("target")]
                    mol, smiles_sequence = all_atom_glycan.AllAtomGlycan.from_iupac(iupac, **kwargs)
                    writer.writerow(values + [smiles_sequence])
