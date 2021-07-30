import json
import csv
from pathlib import Path
from tqdm import tqdm
import argtyped


class Arguments(argtyped.Arguments):
    keys: Path = Path("indoor-keys.txt")
    input: Path = Path("airbnb-train.tsv")
    output: Path = Path("airbnb-train-indoor.tsv")


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    with open(args.keys) as fid:
        photo_ids = {list(map(int, k.strip().split("-")))[1] for k in fid.readlines()}

    with open(args.input, newline="") as fid:
        reader = csv.DictReader(
            fid, delimiter="\t", fieldnames=("listing_id", "photo_id", "url", "caption")
        )
        total = 0
        for _ in reader:
            total += 1

    with open(args.input, newline="") as fid:
        with open(args.output, "w", newline="") as out:
            reader = csv.DictReader(
                fid,
                delimiter="\t",
                fieldnames=("listing_id", "photo_id", "url", "caption"),
            )
            writer = csv.DictWriter(
                out,
                delimiter="\t",
                fieldnames=("listing_id", "photo_id", "url", "caption"),
            )
            counter = 0
            for row in tqdm(reader, total=total):
                photo_id = int(row["photo_id"])
                if photo_id in photo_ids:
                    writer.writerow(row)
                    photo_ids.remove(photo_id)
                    counter += 1

            print("Add", counter, "items")
