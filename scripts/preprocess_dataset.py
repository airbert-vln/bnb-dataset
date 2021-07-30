"""
Create bnb_train.json and bnb_test.json
"""
from pathlib import Path
from typing import List, Optional, Any, Dict
import csv
from operator import itemgetter
from itertools import groupby
import argtyped
from scripts.helpers import load_json, save_json, save_txt, get_key


def flatten(seq):
    ret = []
    for sub in seq:
        ret += sub
    return ret


def is_empty(stc: str):
    return stc.strip() == ""


class Arguments(argtyped.Arguments):
    csv: Path  # listing_id, photo_id, url, caption
    name: str
    min_caption: int = 2
    min_length: int = 4
    captionless: bool = True


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    # add captions from downloaded images
    with open(args.csv) as fid:
        reader = csv.DictReader(
            fid, delimiter="\t", fieldnames=["listing_id", "photo_id", "url", "caption"]
        )
        captions = [
            {
                "listing_id": int(r["listing_id"]),
                "photo_id": int(r["photo_id"]),
                "instructions": [r["caption"] if not is_empty(r["caption"]) else ""],
            }
            for r in reader
        ]
    print("Loaded captions", len(captions))

    # filter out captionless images
    if not args.captionless:
        captions = [c for c in captions if not is_empty(c["instructions"][0])]
    print("After filtering out captionless images", len(captions))

    # group by listings
    captions = sorted(captions, key=itemgetter("listing_id"))
    captions_by_listing = {
        listing: list(items)
        for listing, items in groupby(captions, key=itemgetter("listing_id"))
    }
    print("Listings", len(captions_by_listing))

    # filter out listings not having enough captions
    captions_by_listing = {
        listing: items
        for listing, items in captions_by_listing.items()
        if sum(not is_empty(c["instructions"][0]) for c in items) >= args.min_caption
        and len(items) >= args.min_length
    }
    print("Listings with enough captions", len(captions_by_listing))
    print("Number of photos", sum(len(items) for items in captions_by_listing.values()))

    # export
    save_json(flatten(captions_by_listing.values()), f"{args.name}.json")
    save_txt(captions_by_listing.keys(), f"{args.name}-listings.txt")
