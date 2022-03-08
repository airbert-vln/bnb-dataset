from pathlib import Path
import csv
import sys
from typing import List
import pickle
import lmdb
import pickle
from tqdm.auto import tqdm
import argtyped
from helpers import get_key

csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    output: str
    datasets: List[Path]


def extract_lmdb_keys(filename: Path) -> List[str]:
    env = lmdb.open(str(filename))
    with env.begin(write=False) as txn:
        bkeys = txn.get("keys".encode())
        if bkeys is None:
            raise ValueError("No keys were found")
        return [key.decode() for key in pickle.loads(bkeys)]


def extract_tsv_keys(filename: Path) -> List[str]:
    with open(filename) as fid:
        reader = csv.DictReader(
            fid, delimiter="\t", fieldnames=["listing_id", "photo_id", "url", "caption"]
        )
        return [get_key(item["listing_id"], item["photo_id"]) for item in reader]


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    all_keys = []
    for filename in tqdm(args.datasets):
        if filename.suffix == ".lmdb":
            all_keys += extract_lmdb_keys(filename)
        elif filename.suffix == ".tsv":
            all_keys += extract_tsv_keys(filename)
        else:
            raise ValueError(f"Unknown extension {filename.suffix}")

    with open(args.output, "w") as fid:
        fid.write("\n".join(all_keys))
