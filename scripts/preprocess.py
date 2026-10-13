import json
from pathlib import Path
from src.datasets.preprocess import prepare

if __name__ == "__main__":
    for dataset in ["ASSIST2009", "ASSIST2015", "Algebra2005"]:
        path = Path("data/raw")/(dataset+(".tsv" if dataset == "Algebra2005" else ".csv"))
        print(json.dumps(prepare(dataset, path)))
