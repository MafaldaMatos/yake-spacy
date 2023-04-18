from pathlib import Path
from typing import Tuple, List

from tieval import datasets
from tieval.base import Document

from src.utils import split_train_val

Documents = List[Document]


def read_dataset(
        name: str,
        path: Path
) -> Tuple[Documents, Documents, Documents]:
    """Read a tieval dataset."""
    data = datasets.read(name, path)

    if data.test:
        train_docs, val_docs = split_train_val(data.train)
        test_docs = data.test
    else:
        train_val_docs, test_docs = split_train_val(data.documents)
        train_docs, val_docs = split_train_val(train_val_docs, split=0.9)

    return train_docs, val_docs, test_docs


def read_datasets(
        names: List[str],
        path: Path
) -> Tuple[Documents, Documents, Documents]:
    """Read a list of datasets from tieval."""
    train, validation, test = [], [], []
    for name in names:
        dataset_train, dataset_validation, dataset_test = read_dataset(name, path)
        train.extend(dataset_train),
        validation.extend(dataset_validation)
        test.extend(dataset_test)
    return train, validation, test
