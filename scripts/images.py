"""
Download images related to a room
"""
import requests
import json
from typing import Dict
from pathlib import Path
from tqdm.auto import tqdm
import dask.multiprocessing
import dask.bag as db
import time


def download_photos(listing_id: int, download_dir: Path):
    path = Path(download_dir) / str(listing_id) / "photos"
    path.mkdir(exist_ok=True, parents=True)

    with open(path / "photos.json", "w") as fid:
        json.dump(photos, fid)


if __name__ == "__main__":

    with open("rooms.json") as fid:
        rooms = json.load(fid)

    download = Path("download")

    def is_not_done(listing_id):
        return not (download / str(listing_id) / "photos.json").is_file()

    errors = []

    for room in tqdm(rooms):
        if is_not_done(room):
            try:
                download_room(room, download)
            except:
                print(f"issue with room {room}")
                errors.append(room)
                with open("errors-room.json", "w") as fid:
                    json.dump(errors, fid)
    # dask.config.set(scheduler="processes", num_workers=20)
    # db.from_sequence(rooms).map(lambda l_id: download_room(l_id, download)).compute()
