"""
We apply basic rules to remove some captions
"""
from operator import itemgetter
from itertools import groupby
import csv
from pathlib import Path
from tqdm import tqdm
import argtyped
from scripts.helpers import is_empty

class Arguments(argtyped.Arguments):
    input: Path = Path("airbnb-train.tsv")
    output: Path = Path("airbnb-train-filtered.tsv")




if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    with open(args.input, newline="") as fid:
        reader = csv.DictReader(
            fid, delimiter="\t", fieldnames=("listing_id", "photo_id", "url", "caption")
        )
        rows = list(reader)

    rows = sorted(rows, key=itemgetter("listing_id"))
    rows_by_listing_id = {
        listing_id: list(items)
        for listing_id, items in groupby(rows, key=itemgetter("listing_id"))
    }

    counter = 0
    rows = []
    for  items in tqdm(list(rows_by_listing_id.values())):
        uniq = set([])
        for row in items:
            if not is_empty(row["caption"]):
                if row["caption"] in uniq:
                    row["caption"] = ""
                    counter += 1
                else:
                    uniq.add(row["caption"])
            rows.append(row)

    print(f"Empty {counter} captions")

    with open(args.output, "w", newline="") as out:
        writer = csv.DictWriter(
            out, delimiter="\t", fieldnames=("listing_id", "photo_id", "url", "caption")
        )
        writer.writerows(rows)
