def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import mlmc
from build_pretraining_dataset import *

if __name__ == "__main__":
    import argparse

    a = argparse.ArgumentParser()

    a.add_argument("--fields", default=["Name", "Description", "CategoryText"], help="Fields to concatenate as input.")
    a.add_argument("--sep", default="\n", help="Separator for  the concatentation of fields.")
    a.add_argument("--device", default="cuda:0", help="Device to train on. (cuda:x or cpu)")
    a.add_argument("--input", default="data/",
                   help="folder containing the data (train.json, validation.json, test_public.json).")
    a.add_argument("--lr", default=0.00005, type=float,
                   help="folder containing the data (train.json, validation.json, test_public.json).")
    a.add_argument("--representation", default="distilroberta-base", type=str,
                   help="representation (default: distilroberta-base).")
    a.add_argument("--epochs", default=25, type=int,
                   help="epochs on the real train data")
    a.add_argument("--pre", default=10, type=int,
                   help="epochs on pretraining data.")
    a.add_argument("--classweights", default="None", type=str)
    args = a.parse_args()
    print(args)

    fields = args.fields
    separator = args.sep
    device = args.device
    directory = pathlib.Path(args.input)
    lr = args.lr
    representation = args.representation
    epochs=args.epochs
    pre_epochs = args.pre

    print(directory)
    print(pathlib.Path(".").absolute())
    data = get_gs1_dataset(directory, separator=separator, fields=fields)

    tc = mlmc.models.MoKimCNN(n_outputs=len(data["train"].classes), classes=data["train"].classes,
                              threshold="max",
                              representation=representation, n_layers=1, target="single",
                              device=device, finetune=True,max_len=480)

    # Pretraining
    cb = mlmc.callbacks.CallbackSaveAndRestore(metric="valid_loss", mode="min")
    pretrain_history = tc.fit(data["webdata"]+data["wikidata"], data["valid"], epochs=pre_epochs, batch_size=20, valid_batch_size=24, return_report=True, callbacks=[cb])

    # Actual Training
    cb = mlmc.callbacks.CallbackSaveAndRestore(metric="accuracy_2", mode="max")
    history = tc.fit(data["train"], data["valid"], epochs=epochs, batch_size=20, valid_batch_size=24, return_report=True,callbacks = [cb])


    validation_prediction_file = directory / f"prediction_validation.txt"
    prediction = [
        (x,) +
        tc.predict(data["cl_val"][data["cl_val"]["ID"] == x][fields].agg(separator.join, axis=1).to_list())[0] for
        x in
        tqdm(data["cl_val"]["ID"].tolist())]
    with open(validation_prediction_file, "w", encoding="utf-8") as f:
        f.write("\n".join([",".join([str(y)] + sum(x, [])) for y, *x in prediction]))


    prediction = [
        (x,) + tc.predict(data["cl_test"][data["cl_test"]["ID"] == x][fields].agg(separator.join, axis=1).to_list())[0] for
        x in
        tqdm(data["cl_test"]["ID"].tolist())]
    with open(directory / f"prediction_test.txt", "w",encoding="utf-8") as f:
        f.write("\n".join(
            [",".join([str(y)] + [w for i, w in enumerate(sum(x, []))]) for y, *x in
             prediction]))
