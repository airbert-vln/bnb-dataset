from pathlib import Path
import csv
import base64
from multiprocessing import Pool
import sys
from glob import glob
from typing import List, Optional, Iterator, Dict, Union, Tuple, Any, Iterable
import pickle
import lmdb
import pickle
from tqdm.auto import tqdm
import argtyped
import numpy as np
from helpers import load_json

csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    output: str
    datasets: List[str]


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    all_keys = []
    for lmdb_file in tqdm(args.datasets):
        env = lmdb.open(lmdb_file)
        with env.begin(write=False) as txn:
            bkeys = txn.get("keys".encode())
            if bkeys is None:
                raise ValueError("No keys were found")
            all_keys += [key.decode() for key in pickle.loads(bkeys)]

    with open(args.output, "w") as fid:
        fid.write("\n".join(all_keys))
