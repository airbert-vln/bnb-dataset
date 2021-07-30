"""
Extract listing ids from location files
"""
from pathlib import Path
from typing import Set
import json
import argtyped
from tqdm.auto import tqdm
from helpers import list_to_txt


class Arguments(argtyped.Arguments):
    loc_dir: Path = Path("dora")
    room_dir: Path = Path("merlin")


if __name__ == "__main__":

    args = Arguments()

    locations = list(args.loc_dir.glob("*.json"))
    room_set: Set[int] = set()

    for filename in tqdm(locations):
        with open(filename) as fid:
            data = json.load(fid)

        rooms = []
        for section in data["data"]["dora"]["exploreV3"]["sections"]:
            if section["__typename"] != "DoraExploreV3ListingsSection":
                continue
            for item in section["items"]:
                listing_id = int(item["listing"]["id"])
                if listing_id not in room_set:
                    rooms.append(listing_id)
                    room_set.add(listing_id)
                    room_folder = args.room_dir / filename.stem / str(listing_id)
                    room_folder.mkdir(exist_ok=True, parents=True)

        if rooms != []:
            list_to_txt(rooms, args.room_dir / filename.stem / "rooms.txt")

    print(f"Extracted {len(room_set)} rooms from {len(locations)} locations")
