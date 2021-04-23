import mlmc
import pandas as pd
import json
from tqdm import tqdm
import pathlib
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import pathlib
import mlmc
import tempfile
from urllib import error
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO



def get_gs1_hierarchy():
    url = "https://www.gs1.org/sites/default/files/docs/gpc/en_2020-06.zip"

    try:
        resp = urlopen(url)
    except error.HTTPError:
        print(error.HTTPError)
        return None
    assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
    # with tempfile.TemporaryDirectory() as tmpdir:

    zf = ZipFile(BytesIO(resp.read()))

    # , encoding = "ISO-8859-1"
    content = []

    for file in  [x for x in zf.namelist() if x.endswith(".txt")]:
        with zf.open(file, "r", )as f:
            content.append(pd.read_csv(f, delimiter="\t", encoding="ISO-8859-1"))

    all = pd.concat(content, axis=0)[
        ["Segment Code", "Segment Description", "Family Code", "Family Description", "Class Code",
         "Class Description"]].dropna()

    triples = pd.concat(
        [
            all["Segment Code"].astype(int).astype(str) + "_" + all["Segment Description"],
            all["Family Code"].astype(int).astype(str) + "_" + all["Family Description"],
            all["Class Code"].astype(int).astype(str) + "_" + all["Class Description"]
        ], axis=1
    )
    return triples

def get_wiki_descriptions(directory):
    descriptions = [{}, {}, {}]
    for i, descriptions_file in enumerate((
            directory / "lvl1_entities_7.json", directory / "lvl2_entities_7.json",
            directory / "lvl3_entities_7.json")):
        with open(descriptions_file, "r") as f:
            content = json.loads(f.read())
            descriptions[i].update(
                {x: " ".join((v["text"], "is a", ", ".join(sum([k["summary"] for k in v["entities"]], [])))) for x, v in
                 content.items()})

    return descriptions


def get_webdatacommons():
    url = "http://webdatacommons.org/structureddata/2014-12/products/data/goldstandard_eng_v1.csv"
    resp = urlopen(url)
    return pd.read_csv(resp, sep=";", encoding="ISO-8859-1").fillna("")




def get_gs1_dataset(directory = pathlib.Path("productclassification/data"), separator=" \n ", fields=["Name", "Description", "URL"]):



    cl_train = pd.read_json(directory / "mwpd" / 'train.json', lines=True)
    cl_val = pd.read_json(directory / "mwpd" /  'validation.json', lines=True)
    cl_test = pd.read_json(directory / "mwpd" /  'test_public.json', lines=True)

    webdatacommons = get_webdatacommons()#pd.read_csv(directory / "webdatacommons"/'goldstandard_eng_v1.csv', sep=";", encoding="ISO-8859-1").fillna("")
    triples = get_gs1_hierarchy()
    descriptions = get_wiki_descriptions(directory=directory / "wiki")

    # Mapping the column names of webdatacommons to the mwpd training data
    map = {"Name":"s:name", "URL":"URL", "Description":"s:description", "CategoryText":'s:category',}


    # Real encoded classes of the dataset
    list_of_classes = {}
    for l in ("lvl1", "lvl2", "lvl3"):
        list_of_classes[l] = []
        for data in (cl_train, cl_val,):
            list_of_classes[l].extend(data[l].to_list())

    list_of_classes = {x: list(set(y)) for x, y in list_of_classes.items()}
    dataset_classes = [dict(zip(classes, range(len(classes)))) for classes in list_of_classes.values()]


    extra_data = {"text": [], "labels": []}
    for x, y, z in [x.split("|") for x in set(list(["|".join(x) for x in triples.values.tolist()]))]:
        if x in dataset_classes[0].keys() and y in dataset_classes[1].keys() and z in dataset_classes[2].keys():
            extra_data["text"].append(descriptions[2][z.split("_")[0]])
            extra_data["labels"].append([[x], [y], [z]])

    f = pd.DataFrame([[x] + [l[0] for l in y] for x,y in zip(extra_data["text"], extra_data["labels"])])
    f.to_csv("mwpd_entity_wiki_data.csv")

    ### package data
    wiki_data = mlmc.data.MultiOutputSingleLabelDataset(x=extra_data["text"], y=extra_data["labels"], classes=dataset_classes)

    webdata = mlmc.data.MultiOutputSingleLabelDataset(
        classes=dataset_classes,
        x=webdatacommons[[map[s] for s in fields]].agg(separator.join, axis=1).to_list(),
        y=[[[l] for l in x] for x in
           webdatacommons[["GS1_Level1_Category", "GS1_Level2_Category", "GS1_Level3_Category"]].values.tolist()]
    )
    webdata.reduce(dataset_classes)

    train = mlmc.data.MultiOutputSingleLabelDataset(
        classes=dataset_classes,
        x=cl_train[fields].agg(separator.join, axis=1).to_list(),
        y=[[[l] for l in x] for x in cl_train[["lvl1", "lvl2", "lvl3"]].values.tolist()]
    )

    valid = mlmc.data.MultiOutputSingleLabelDataset(
        classes=dataset_classes,
        x=cl_val[fields].agg(separator.join, axis=1).to_list(),
        y=[[[l] for l in x] for x in cl_val[["lvl1", "lvl2", "lvl3"]].values.tolist()]
    )
    test = mlmc.data.MultiOutputSingleLabelDataset(
        classes=dataset_classes,
        x=cl_test[fields].agg(separator.join, axis=1).to_list(),
        y=None
    )

    return {"train": train,
            "valid":valid,
            "test": test,
            "webdata": webdata,
            "wikidata": wiki_data,
            "classes":dataset_classes}
