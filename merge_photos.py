"""
Create image merging datasets
"""
import csv
import sys
from glob import glob
import json
import math
import logging
import random
import json
from typing import DefaultDict, Tuple, Dict, Iterable, List, Set, Union
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from tqdm.auto import tqdm
import argtyped


random.seed(1)
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


def load_json(f):
    with open(f) as fid:
        return json.load(fid)


def save_json(data, f):
    with open(f, "w") as fid:
        json.dump(data, fid, indent=2)


class Arguments(argtyped.Arguments):
    source: Path
    output: Path
    detection_dir: Path = Path("./places365")
    fieldnames: Tuple[str, ...] = (
        "listing_id",
        "photo_id",
        "category",
        "attributes",
        "is_indoor",
    )
    hierarchy: Path = Path("places365.hierarchy.txt")
    min_captioned: int = 2  # minimum of captioned images once merged
    min_length: int = 4  # minimum of images once merged


def load_hierarchy(hierarchy: Path) -> Dict[str, str]:
    classes = {}
    with open(hierarchy) as class_file:
        for line in class_file:
            name, cat = line.strip().split("\t")
            classes[name] = cat
    return classes


def init_categories(classes):
    return {c: 0 for c in classes.keys()}


def init_hoc(classes):
    return {c: 0 for c in classes.values()}


def get_category(item, classes):
    values, keys = zip(*eval(item["category"]))
    categories = init_hoc(classes)
    for k, v in zip(keys, values):
        if k not in classes:
            continue
        cat = classes[k]
        categories[cat] += v
    name, score = sorted(list(categories.items()), key=itemgetter(1), reverse=True)[0]
    return name, score


Detection = DefaultDict[int, Dict[int, Tuple[str, float]]]


def load_detections(
    detection_dir: Path,
    hierarchy: Path,
    photo_ids: Iterable[int],
    fieldnames=(
        "listing_id",
        "photo_id",
        "category",
        "attributes",
        "is_indoor",
    ),
) -> Detection:
    listings: Detection = defaultdict(dict)
    classes = load_hierarchy(hierarchy)
    files = list(glob(f"{detection_dir}/**/*.tsv", recursive=True))
    logger.info("Loading detections...")

    for filename in tqdm(files):
        with open(filename, "rt") as fid:
            reader = csv.DictReader(fid, delimiter="\t", fieldnames=fieldnames)
            for item in reader:
                if not int(item["photo_id"]) in photo_ids:
                    continue
                is_indoor = eval(item["is_indoor"])[1]
                if not is_indoor:
                    continue

                name, score = get_category(item, classes)
                listings[int(item["listing_id"])][int(item["photo_id"])] = (name, score)

    return listings


def is_captionless(instr: str) -> bool:
    return instr.strip() == ""


def get_states(
    photo_ids: Iterable[Union[int, Tuple[int, ...]]], captioned_photos: Iterable[int]
) -> List[bool]:
    """
    Return states (captioned/captionless)
    """
    states = []
    for photo_id in photo_ids:
        if isinstance(photo_id, tuple):
            states.append(any(pid in captioned_photos for pid in photo_id))
        elif isinstance(photo_id, int):
            states.append(photo_id in captioned_photos)
        else:
            raise ValueError("Unexpected type {type(photo_id)}")
    return states


def merging_detections(
    listings: Detection,
    captioned_photos: Iterable[int],
    min_captioned: int,
    min_length: int,
    merging_rooms: Tuple[str, ...] = ("bedroom", "living_room", "bathroom", "kitchen"),
    max_photo_per_merging: int = 10,
) -> Dict[int, Tuple[Tuple[int, str, float], ...]]:

    logger.info("Merging images...")
    merging_by_photo_id = {}
    counter = 0

    for listing_id, photos in tqdm(listings.items()):
        num_photos = len(list(photos.keys()))
        if num_photos < min_length:
            logger.error(f"Not enough images on listing {listing_id}")
            continue

        num_captioned = sum(get_states(photos.keys(), captioned_photos))
        if num_captioned < min_captioned:
            # This should happen only when there's not enough detections
            logger.error(f"{listing_id} does not have enough captioned images")
            continue

        counter += 1

        # group photo ids by class names and sort them by number of photos per class
        by_classes = groupby(
            sorted(list(photos.items()), key=lambda x: x[1][0]), key=lambda x: x[1][0]
        )
        group_photos: List[Tuple[Tuple[int, str, float], ...]] = []
        for name, group in by_classes:
            photo_ids = list(group)
            if name not in merging_rooms:
                for photo_id in photo_ids:
                    pid: int = photo_id[0]
                    weight = photo_id[1][1]
                    group_photos.append(((pid, name, weight),))
            else:
                subgroup = []
                for photo_id in photo_ids:
                    pid = photo_id[0]
                    weight = photo_id[1][1]
                    subgroup.append((pid, name, weight))
                group_photos.append(tuple(subgroup))

        # ungroup groups that contain too many photos
        for idx, grp in enumerate(group_photos):
            if len(grp) <= max_photo_per_merging:
                continue
            num_groups = math.ceil(len(grp) / max_photo_per_merging)
            num_captioned = sum(get_states([p[0] for p in grp], captioned_photos))
            num_groups = min(num_groups, num_captioned)
            num_groups = max(num_groups, 1)
            new_groups: List[List[Tuple[int, str, float]]] = [
                [] for _ in range(num_groups)
            ]

            # each new groups must have at least one captioned image
            # so we start by distributing them
            done: List[int] = []
            for i, photo_id in enumerate(grp):
                if photo_id[0] in captioned_photos:
                    new_groups[len(done)] = [photo_id]
                    done.append(i)
                    if len(done) == num_groups:
                        break

            # then we distribute the remaining photos
            for i, photo_id in enumerate(grp):
                if i in done:
                    continue
                new_groups[i % num_groups].append(photo_id)

            del group_photos[idx]
            for g in new_groups:
                group_photos.append(tuple(g))

        # ungroup captioned photos until we have enough groups with a least 1 captioned photo
        # WARNING: a group might contain zero captioned photo!
        def is_captioned_in_group(grp, captioned_photos):
            return get_states([p[0] for p in grp], captioned_photos)

        # make sure we have enough captioned photos
        captioned_states_grps = [
            is_captioned_in_group(grp, captioned_photos) for grp in group_photos
        ]
        num_captioned_grps = sum([sum(sg) > 0 for sg in captioned_states_grps])
        while num_captioned_grps < min_captioned:
            # pick a group having at least two captioned photos
            candidates = [i for i, sg in enumerate(captioned_states_grps) if sum(sg) > 1]
            assert candidates != [], listing_id
            grp_idx = random.choice(candidates)
            grp = group_photos[grp_idx]

            idx_captioned = [
                (idx, photo_id)
                for idx, photo_id in enumerate(grp)
                if photo_id[0] in captioned_photos
            ]

            photo_idx, photo_id = random.choice(idx_captioned)
            group_photos.append((photo_id,))
            group_photos.append(tuple(p for i, p in enumerate(grp) if i != photo_idx))
            del group_photos[grp_idx]

            captioned_states_grps = [
                is_captioned_in_group(grp, captioned_photos) for grp in group_photos
            ]
            num_captioned_grps = sum([sum(sg) > 0 for sg in captioned_states_grps])

        # make sure we have enough images
        while len(group_photos) < min_length:
            # pick a group having at least two photos
            grp_idx = random.choice(
                [i for i, grp in enumerate(group_photos) if len(grp) > 1]
            )
            grp = group_photos[grp_idx]

            idx_captioned = [(idx, photo_id) for idx, photo_id in enumerate(grp)]
            photo_idx, photo_id = random.choice(idx_captioned)
            group_photos.append((photo_id,))
            group_photos.append(tuple(p for i, p in enumerate(grp) if i != photo_idx))
            del group_photos[grp_idx]

        # update returned dict
        for grp in group_photos:
            for pid, name, weight in grp:
                merging_by_photo_id[pid] = grp

            continue

    print(f"Merged {counter} photos")

    return merging_by_photo_id


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load BnB captions by photo_id
    logging.info(f"Loading BnB JSON from {args.source}")
    data = load_json(args.source)
    photo_ids: Set[int] = set()
    captioned: Set[int] = set()
    listing_ids: Set[int] = set()
    for item in data:
        photo_id = int(item["photo_id"])
        photo_ids.add(photo_id)
        listing_id = int(item["listing_id"])
        listing_ids.add(listing_id)
        if not is_captionless(item["instructions"][0]):
            captioned.add(photo_id)
    logger.info(f"Load {len(photo_ids)} images")
    logger.info(f"Load {len(captioned)} captioned images")

    # Load detections
    detections_by_listing = load_detections(args.detection_dir, args.hierarchy, photo_ids)
    logger.info(f"Load {sum(len(v) for v in detections_by_listing.values())} detections")

    merging_by_photo_ids = merging_detections(
        detections_by_listing,
        captioned,
        min_captioned=args.min_captioned,
        min_length=args.min_length,
    )

    # Export merging
    for item in data:
        pid = int(item["photo_id"])
        if pid in merging_by_photo_ids:
            item["merging"] = [int(m[0]) for m in merging_by_photo_ids[pid]]
            item["weights"] = [float(m[2]) for m in merging_by_photo_ids[pid]]
            item["room"] = merging_by_photo_ids[pid][0][1]
        else:
            item["merging"] = [pid]
            item["weights"] = [1]
            item["room"] = ["unknown"]

    logger.info(f"Outputting to {args.output}")
    save_json(data, args.output)
