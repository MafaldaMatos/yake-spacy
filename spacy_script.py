import glob
import json
import os
from pathlib import Path

import yake
from tqdm import tqdm

from highlight import TextHighlighter
from src.data.spacy import SpacyDataLoader
from src.models.spacy import SpacyModel
from src.dataprocess import process
from src.utils import split_train_val


ROOT = Path(__file__).parent
ANNOTATED_DATA_PATH = ROOT / "data" / "annotated"


def eraseOverlapIntervals(intervals):
    intervals.sort()

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
            # Choose the interval that has the shorter end point.
            lastEnd = min(lastEnd, e)

    return validIntervals


def weak_labelling(loaders, mode):
    actual_loaders = []
    startendloaders = []
    to_save = []

    # does weak labelling
    count = 0
    for text in tqdm(loaders):
        # define the text language as portuguese
        kw_extractor = yake.KeywordExtractor(lan="pt")
        keywords = kw_extractor.extract_keywords(text)

        # if count > 4:
        # break
        # print(count+1)
        # ordered read: 1880? 11276? 12234? 14780? 16248? 20218? 21015? 25725? 35117? 38474?
        try:
            th = TextHighlighter(
                max_ngram_size=3, highlight_pre="", highlight_post="")
            newtext, startends = th.highlight(text, keywords)
            new_list = []
            startends = set(startends)  # remove duplicates
            startends = list(startends)
            # removes overlaps, doesnt work for all of them
            startends = eraseOverlapIntervals(startends)
            for tup in range(len(startends)):
                # add the label for fitting
                temp_list = list(startends[tup])
                temp_list.append("NOUN")  # what label to use here?
                new_list.append(temp_list)
            startendloaders.append(new_list)
            actual_loaders.append(text)
            to_save.append([text, {"entities": new_list}])
            count = count + 1
            pass
        except:
            pass

    with open(ROOT / r'/startendloaders'+mode+'.txt', 'w') as fp:
        for item in startendloaders:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done start end values')

    loads = {}
    with open(ROOT / r'/loaders'+mode+'.json', "w") as fp:
        idx = 0
        for item in actual_loaders:
            # write each item on a new line
            loads[idx] = item
            idx = idx + 1
        json.dump(loads, fp)
        print('Done text values')

    print(to_save)
    with open(ROOT / r'/loadersdict'+mode+'.json', "w") as fp:
        json.dump(to_save, fp)
        print("Done json dict")


def annotate():
    path = ROOT / "data/processed/portuguese/"

    list_of_files = list(path.glob('**/*.json'))

    loaders, loadersval = [], []

    # randomized file read
    ss = split_train_val(list_of_files)
    # ss[0] = train, ss[1] = validation

    for data in ss[0]:
        # print(data)
        # Opening JSON file
        f = open(data)

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        # print(data["text"])
        loaders.append(data["text"])

    for data in ss[1]:
        # print(data)
        # Opening JSON file
        f = open(data)

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        # print(data["text"])
        loadersval.append(data["text"])

    weak_labelling(loaders, "train")
    weak_labelling(loadersval, "validation")


def train():

    model = SpacyModel("pt")
    model.load(ROOT / "models/base/spacy/portuguese")

    print("done loading model")

    train_data = SpacyDataLoader(
        ANNOTATED_DATA_PATH / r'/loadersdicttrain.json')
    eval_data = SpacyDataLoader(
        ANNOTATED_DATA_PATH / r'/loadersdictvalidation.json')

    val = model.fit(train_data, eval_data, 10, 0.1, ROOT / "models/test")
    print(val)

    with open(ROOT / r'/fit.json', "w") as fp:
        json.dump(val, fp)
        print("Done json dict")


if __name__ == "__main__":
    train()
