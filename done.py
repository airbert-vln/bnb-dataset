from pathlib import Path
from glob import glob
import csv
import sys
from multiprocessing import Pool
from typing import List
from tqdm.auto import tqdm
import argtyped
from helpers import save_json, load_json

csv.field_size_limit(sys.maxsize)

TSV_FIELDNAMES = [
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
]


def search_locations(image_folder: Path) -> List[Path]:
    return [f for f in image_folder.iterdir() if f.is_dir()]


def load_photo_paths(cache: Path) -> List[str]:
    if not cache.is_file():
        raise RuntimeError("Please cache paths first")

    with open(cache, "r") as fid:
        photos = [p.strip() for p in fid.readlines()]
    return photos


def cache_photo_paths(image_folder: Path, cache: Path):
    if cache.is_file():
        return

    locations = search_locations(image_folder)
    photos = []
    for location in tqdm(locations):
        for photo in location.glob("*.jpg"):
            photos.append(str(photo))

    photo_id_to_path = {
        int(Path(path.strip()).stem.split("-")[1]): path for path in tqdm(photos)
    }

    save_json(photo_id_to_path, cache)


class Arguments(argtyped.Arguments):
    start: int = 0
    num_workers: int = 1
    num_splits: int = 1
    images: Path = Path("images")
    cache: Path = Path(".photo_id_to_path.json")


def extraction(args: Arguments, start: int):
    if start == 0:
        print("Extracting photo id")
    photo_id_to_path = load_json(args.cache)
    if start == 0:
        print(f"Found {len(photo_id_to_path)} images")

    tsv_files = list(glob(f"{args.images}/**/*.tsv.*", recursive=True))
    if start == 0:
        print(f"Found {len(tsv_files)}")
    tsv_files = tsv_files[start :: args.num_splits]

    output = f".done.{start}.txt"
    # empty file
    with open(output, "w") as fid:
        print(f"Creating {output}")

    for f in tqdm(tsv_files):
        with open(f".done.{start}.txt", "a") as fout:
            with open(f) as fid:
                reader = csv.DictReader(fid, TSV_FIELDNAMES, delimiter="\t")
                for item in reader:
                    done = photo_id_to_path[item["photo_id"]]
                    fout.write(f"{done}\n")


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    cache_photo_paths(args.images, args.cache)

    if args.num_workers == 0:
        extraction(args, args.start)

    elif args.num_workers > 1:
        p = Pool(args.num_workers)
        p.starmap(
            extraction,
            [
                (args, proc_id)
                for proc_id in range(args.start, args.start + args.num_workers)
            ],
        )
