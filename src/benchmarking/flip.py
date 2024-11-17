import os
import csv
import math
from collections import defaultdict

from tqdm import tqdm

from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.core import Registry as R


class FLIPDataset(data.ProteinDataset):

    def load_csv(self, csv_file, sequence_field="sequence", target_fields=None, verbose=0, **kwargs):
        if target_fields is not None:
            target_fields = set(target_fields)

        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(tqdm(reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)))
            fields = next(reader)
            train, valid, test = [], [], []
            _sequences = []
            _targets = defaultdict(list)
            for i, values in enumerate(reader):
                for field, value in zip(fields, values):
                    if field == sequence_field:
                        _sequences.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        _targets[field].append(value)
                    elif field == "set":
                        if value == "train":
                            train.append(i)
                        elif value == "test":
                            test.append(i)
                    elif field == "validation":
                        if value == "True":
                            valid.append(i)

        valid_set = set(valid)
        sequences = (
            [_sequences[i] for i in train if i not in valid_set]
            + [_sequences[i] for i in valid]
            + [_sequences[i] for i in test]
        )
        targets = defaultdict(list)
        for key, value in _targets.items():
            targets[key] = (
                [value[i] for i in train if i not in valid_set]
                + [value[i] for i in valid]
                + [value[i] for i in test]
            )
        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)
        self.num_samples = [len(train) - len(valid), len(valid), len(test)]


@R.register("datasets.AAV")
class AAV(FLIPDataset):

    url = (
        "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip"
    )
    md5 = "cabdd41f3386f4949b32ca220db55c58"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]
    region = slice(474, 674)

    def __init__(self, path, split="two_vs_many", keep_mutation_region=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, "aav")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in [
            "des_mut",
            "low_vs_high",
            "mut_des",
            "one_vs_many",
            "sampled",
            "seven_vs_many",
            "two_vs_many",
        ]

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)
        if keep_mutation_region:
            for i in range(len(self.data)):
                self.data[i] = self.data[i][self.region]
                self.sequences[i] = self.sequences[i][self.region]

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.GB1")
class GB1(FLIPDataset):

    url = (
        "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip"
    )
    md5 = "14216947834e6db551967c2537332a12"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]

    def __init__(self, path, split="two_vs_rest", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, "gb1")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ["one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high", "sampled"]

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.Thermostability")
class Thermostability(FLIPDataset):

    url = "https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip"
    md5 = "0f8b1e848568f7566713d53594c0ca90"
    splits = ["train", "valid", "test"]
    target_fields = ["target"]

    def __init__(self, path, split="human_cell", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        path = os.path.join(path, "thermostability")
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        assert split in ["human", "human_cell", "mixed_split"]

        zip_file = utils.download(self.url, path, md5=self.md5)
        data_path = utils.extract(zip_file)
        csv_file = os.path.join(data_path, "splits/%s.csv" % split)

        self.load_csv(csv_file, target_fields=self.target_fields, verbose=verbose, **kwargs)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
