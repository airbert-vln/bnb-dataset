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
from scripts.helpers import load_txt, get_key

csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    output: str
    keys: Path  # contains the photo_id, listin_id pairs we want to put on the LMDB files
    tsv_folder: Path = Path("img_feature")
    map_size: int = int(5e11)
    num_parts: int = 1
    start: int = 0
    num_workers: int = 0
    buffer_size: int = 1000

    tsv_fieldnames: Tuple[str, ...] = (
        "listing_id",
        "photo_id",
        "image_w",
        "image_h",
        "vfov",
        "features",
        "boxes",
        "cls_prob",
        "attr_prob",
        "featureViewIndex",
        "featureHeading",
        "featureElevation",
    )


class LMDBWriter:
    def __init__(self, path: Union[Path, str], map_size: int, buffer_size: int):
        self._env = lmdb.open(str(path), map_size=map_size)
        self._buffer: List[Tuple[bytes, bytes]] = []
        self._buffer_size = buffer_size
        with self._env.begin(write=False) as txn:
            value = txn.get("keys".encode())
            self._keys: List[bytes] = [] if value is None else pickle.loads(value)

    def write(self, key: str, value: bytes):
        if key in self._keys:
            return
        bkey = key.encode()
        self._keys.append(bkey)
        self._buffer.append((bkey, value))
        if len(self._buffer) == self._buffer_size:
            self.flush()

    def flush(self):
        with self._env.begin(write=True) as txn:
            txn.put("keys".encode(), pickle.dumps(self._keys))
            for bkey, value in self._buffer:
                txn.put(bkey, value)

        self._buffer = []


def load_photo_ids(filename: Path) -> List[int]:
    keys = load_txt(filename)
    return [int(k.split("-")[1]) for k in keys]


def features_to_lmdb(args: Arguments, part_id: int):
    lmdb_file = str(args.output) % part_id
    writer = LMDBWriter(lmdb_file, args.map_size, args.buffer_size)

    tsv_files = list(glob(str(args.tsv_folder / "**/*.tsv.*"), recursive=True))
    if part_id == 0:
        print("Found", len(tsv_files), "files")

    list_photo_ids = load_photo_ids(args.keys)
    if part_id == 0:
        print("Found", len(list_photo_ids), "photos")
    list_photo_ids = list_photo_ids[part_id :: args.num_parts]
    if part_id == 0:
        print("Worker:", len(list_photo_ids), "photos")
    photo_ids = set(list_photo_ids)
    if part_id == 0:
        print("Worker:", len(photo_ids), "photos")

    for path in tqdm(tsv_files):
        with open(path, "rt") as fid:
            reader = csv.DictReader(fid, delimiter="\t", fieldnames=args.tsv_fieldnames)
            for item in reader:
                photo_id = int(item["photo_id"])
                if photo_id not in photo_ids:
                    continue
                photo_ids.remove(photo_id)

                key = get_key(int(item["listing_id"]), photo_id)
                boxes = np.frombuffer(base64.b64decode(item["boxes"]))
                if boxes.size == 0:
                    continue
                writer.write(key, pickle.dumps(item))

        writer.flush()

    writer.flush()


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    if args.num_workers == 0:
        features_to_lmdb(args, args.start)

    else:
        p = Pool(args.num_workers)
        p.starmap(
            features_to_lmdb,
            [
                (args, proc_id)
                for proc_id in range(args.start, args.start + args.num_workers)
            ],
        )
