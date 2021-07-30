"""
Check if images/listings were not downloaded or are corrupted
"""
from pathlib import Path
import json
from dataclasses import dataclass, field
import sys
from typing import List, Dict, Tuple, Iterator, Union, Optional
from tqdm.auto import tqdm
import numpy as np
from PIL import Image, ExifTags
from helpers import save_json, load_json
import argtyped


class Arguments(argtyped.Arguments):
    num_procs: int = 10
    proc_id: int = 0
    images: Path = Path("images")
    merlin: Path = Path("merlin")


def check_listings(merlin: Path, images: Path) -> List[Tuple[str, str, int]]:
    """
    Check that the listing has correctly been downloaded
    """
    missing = []
    listings = [f for f in merlin.rglob("*") if f.is_dir() and f.parent != merlin]

    for listing in tqdm(listings):
        location = listing.parent.name
        listing_id = int(listing.name)

        if not (images / location / str(listing_id)).is_dir():
            missing.append(("no folder", location, listing_id))

        try:
            data = load_json(listing / "photos.json")
            _ = data["data"]["merlin"]["pdpPhotoTour"]["images"]
        except FileNotFoundError:
            missing.append(("no photos.json", location, listing_id))
        except (TypeError, json.decoder.JSONDecodeError) as _:
            missing.append(("corrupted photos.json", location, listing_id))

        try:
            data = load_json(listing / "room1.json")
            _ = data["data"]["merlin"]["pdpSections"]
        except FileNotFoundError:
            missing.append(("no room1.json", location, listing_id))
        except (TypeError, json.decoder.JSONDecodeError) as _:
            missing.append(("corrupted room1.json", location, listing_id))

    return missing


def check_missing_images(merlin: Path, images: Path) -> List[Tuple[str, str, int]]:
    """
    Check that the listing has correctly been downloaded
    """
    missing = []
    locations = list(images.iterdir())

    for location in tqdm(locations):
        for listing in location.iterdir():
            if not listing.is_dir() or listing.parent == merlin:
                continue

            location = listing.parent.name
            listing_id = int(listing.name)

            try:
                data = load_json(merlin / location / str(listing_id) / "photos.json")
                photos = data["data"]["merlin"]["pdpPhotoTour"]["images"]
            except FileNotFoundError:
                missing.append(("no photos.json", location, listing_id))
                continue
            except (TypeError, json.decoder.JSONDecodeError) as _:
                missing.append(("corrupted photos.json", location, listing_id))
                continue

            for photo in photos:
                filename = listing / f"{photo['id']}.jpg"
                if not filename.is_file():
                    missing.append(("no photo", location, listing_id))

    return missing


def load_exif(im: Image.Image) -> Dict:
    if im.getexif() is not None:
        return {
            ExifTags.TAGS[k]: v for k, v in im.getexif().items() if k in ExifTags.TAGS
        }
    else:
        return {}


def check_corrupted_images(images: Path) -> List[Tuple[str, str, int]]:
    """
    Check that the listing has correctly been downloaded
    """
    corrupted = []
    locations = list(images.iterdir())

    for location in tqdm(locations):
        for image in location.rglob("*.jpg"):
            location = image.parents[1].name
            listing_id = int(image.parent.name)

            try:
                im = Image.open(image)
                load_exif(im)
                im.convert("RGB")
            except OSError:
                corrupted.append(("corrupted", location, listing_id))

    return corrupted


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string())

    missing_images = check_missing_images(args.merlin, args.images)
    save_json(missing_images, "missing_images.json")

    corrupted_images = check_corrupted_images(args.images)
    save_json(corrupted_images, "corrupted_images.json")

    missing = check_listings(args.merlin, args.images)
    save_json(missing, "missing.json")

