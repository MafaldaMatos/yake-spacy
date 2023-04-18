import json
from pathlib import Path
import random

from spacy.util import minibatch
from thinc.schedules import compounding


class SpacyDataLoader:

    def __init__(self, path: Path):
        with open(path) as fin:
            self.data = json.load(fin)

    def __len__(self):
        return sum([len(annot["entities"]) for _, annot in self.data])

    def __iter__(self):
        return minibatch(self.data, size=compounding(4., 32., 1.001))

    def shuffle(self):
        random.shuffle(self.data)

    @property
    def texts(self):
        return [sent for sent, _ in self.data]

    @property
    def annotations(self):
        return [[tuple(ent[:2]) for ent in annot["entities"]] for _, annot in self.data]
