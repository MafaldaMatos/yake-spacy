import re
import json
from pathlib import Path
from typing import List

import spacy
import yake
from tqdm import tqdm

from src.data.spacy import SpacyDataLoader
from src.models.spacy import SpacyModel
from src.dataprocess import process
from src.utils import split_train_val


ROOT = Path(__file__).parent
ANNOTATED_DATA_PATH = ROOT / "data" / "annotated"


def eraseOverlapIntervals(intervals: List[List[int]]):
    intervals.sort()
    if len(intervals) == 0:
        return []

    validIntervals = []
    validIntervals.append(intervals[0])
    n = len(intervals)
    lastEnd = intervals[0][1]
    for i in range(1, n):
        s, e = intervals[i]
        if s > lastEnd:
            lastEnd = e
            validIntervals.append(intervals[i])
        else:
            lastEnd = min(lastEnd, e)

    return validIntervals


def weak_labelling(loaders, mode):

    kw_extractor = yake.KeywordExtractor(lan="pt")

    data = []
    for text in tqdm(loaders):

        # TODO: check why texts are empty.
        if not text:
            continue

        keywords = kw_extractor.extract_keywords(text)
        spans = []
        for keyword, _ in keywords:

            matches = re.finditer(keyword, text)
            for match in matches:
                start, end = match.span()
                if text[start:end] != keyword:
                    print(f"ERROR: {text[start:end]} != {keyword}")
                else:
                    spans.append([start, end])

        spans = eraseOverlapIntervals(spans)

        entities = [[s, e, "TIMEX"] for s, e in spans]
        data.append([text, {"entities": entities}])

    filepath = ANNOTATED_DATA_PATH / f"{mode}.json"
    json.dump(data, filepath.open("w"), indent=4)


def annotate():
    path = ROOT / "data/processed/portuguese/"

    list_of_files = list(path.glob('**/*.json'))

    loaders, loadersval = [], []

    # randomized file read
    train, valid = split_train_val(list_of_files)
    # ss[0] = train, ss[1] = validation

    for data in train:
        f = open(data)
        data = json.load(f)
        loaders.append(data["text"])

    for data in valid:
        f = open(data)
        data = json.load(f)
        loadersval.append(data["text"])

    weak_labelling(loaders[:100], "train")
    weak_labelling(loadersval[:100], "validation")


def train():
    model = SpacyModel("pt")

    print("done loading model")

    import json
    train_path = ANNOTATED_DATA_PATH / r'loadersdicttrain.json'
    content = json.loads(train_path.read_text())
    for text, annots in content:
        for s, e, _ in annots["entities"]:
            print(text[s:e])

    train_data = SpacyDataLoader(
        ANNOTATED_DATA_PATH / r'loadersdicttrain.json')
    eval_data = SpacyDataLoader(
        ANNOTATED_DATA_PATH / r'loadersdictvalidation.json')

    metrics = model.fit(
        train_data=train_data,
        validation_data=eval_data,
        n_epochs=10,
        dropout=0.05,
        model_path=ROOT / "models/test"
    )
    print(metrics)

    with open(ROOT / r'fit.json', "w") as fp:
        json.dump(metrics, fp)
        print("Done json dict")


def load_annotated_data():
    a1 = ANNOTATED_DATA_PATH / r'loadersdicttrain.json'
    a2 = ANNOTATED_DATA_PATH / r'loadersdictvalidation.json'

    f1 = open(a1)
    data1 = json.load(f1)

    textdata1 = []
    entitiesdata1 = []
    for datai in data1:
        textdata1.append(datai[0])
        entitiesdata1.append(datai[1]["entities"])

    f2 = open(a2)
    data2 = json.load(f2)

    textdata2 = []
    entitiesdata2 = []
    for datai in data2:
        textdata2.append(datai[0])
        entitiesdata2.append(datai[1]["entities"])

    return textdata1, entitiesdata1, textdata2, entitiesdata2


def check_annotation():
    textdata1, entitiesdata1, textdata2, entitiesdata2 = load_annotated_data()

    nlp = spacy.blank("pt")

    for i in range(len(textdata1)):
        print(spacy.training.offsets_to_biluo_tags(
            nlp.make_doc(textdata1[i]), entitiesdata1[i]))

    for i in range(len(textdata2)):
        print(spacy.training.offsets_to_biluo_tags(
            nlp.make_doc(textdata2[i]), entitiesdata2[i]))


def evaluate(model_path):
    textdata1, entitiesdata1, textdata2, entitiesdata2 = load_annotated_data()

    model = SpacyModel("pt")
    model.load(ROOT / model_path)

    print(entitiesdata2)

    print(model.evaluate(textdata2, entitiesdata2))


if __name__ == "__main__":
    annotate()
    # train()
    # check_annotation()
    # evaluate("models/test/20230511181036")
