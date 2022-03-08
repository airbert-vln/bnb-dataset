"""
Download details of listing
"""
import json
import multiprocessing
from typing import Dict, Tuple
import time
from itertools import product
from multiprocessing.pool import Pool
import multiprocessing
from pathlib import Path
import argtyped
from tqdm.auto import tqdm
from helpers import (
    graphql_request,
    DownloadError,
    save_json,
)
import istarmap


def sleep():
    # We sleep to avoid doing more than 2it/s
    # Otherwise Airbnb sends a TooManyRequestsError
    time.sleep(0.1)


def prepare_path(path: Path, *args: str) -> Path:
    pids: Tuple[int, ...] = multiprocessing.current_process()._identity  # type: ignore
    path /= str(pids[0] if pids else 0)
    for a in args:
        path /= a
    path.mkdir(exist_ok=True, parents=True)
    return path


def _download_info(listing_id: int, path: Path):
    path = prepare_path(path, str(listing_id))
    dest = path / "room1.json"
    if dest.is_file():
        return
    variables: Dict = {
        "request": {
            "id": str(listing_id),
            "layouts": ["SIDEBAR", "SINGLE_COLUMN"],
        }
    }
    key = "PdpPlatformSections"
    query = {
        "operationName": ["PdpPlatformSections"],
        "locale": ["en"],
        "currency": ["EUR"],
        "variables": [json.dumps(variables)],
        "extensions": [
            '{"persistedQuery":{"version":1,"sha256Hash":"625a4ba56ba72f8e8585d60078eb95ea0030428cac8772fde09de073da1bcdd0"}}'
        ],
    }
    room1 = graphql_request(key, query)
    time.sleep(1.0)
    save_json(room1, dest)


def _download_reviews(listing_id: int, path: Path, limit=50):
    path = prepare_path(path, str(listing_id))
    dest = path / "reviews.json"
    if dest.is_file():
        return
    variables = {
        "request": {
            "listingId": str(listing_id),
            "fieldSelector": "for_p3",
            "limit": limit,
        }
    }
    key = "PdpReviews"
    query = {
        "operationName": ["PdpReviews"],
        "locale": ["en"],
        "currency": ["EUR"],
        "variables": [json.dumps(variables)],
        "extensions": [
            '{"persistedQuery":{"version":1,"sha256Hash":"4730a25512c4955aa741389d8df80ff1e57e516c469d2b91952636baf6eee3bd"}}'
        ],
    }
    reviews = graphql_request(key, query)
    num_reviews = reviews["data"]["merlin"]["pdpReviews"]["metadata"]["reviewsCount"]
    time.sleep(0.05)
    save_json(reviews, dest)

    counter = limit
    while counter < num_reviews:
        variables = {
            "request": {
                "listingId": str(listing_id),
                "fieldSelector": "for_p3",
                "limit": limit,
                "offset": counter,
            }
        }
        key = "PdpReviews"
        query = {
            "operationName": ["PdpReviews"],
            "locale": ["en"],
            "currency": ["EUR"],
            "variables": [json.dumps(variables)],
            "extensions": [
                '{"persistedQuery":{"version":1,"sha256Hash":"4730a25512c4955aa741389d8df80ff1e57e516c469d2b91952636baf6eee3bd"}}'
            ],
        }
        counter += limit
        time.sleep(0.05)
        save_json(reviews, dest.parent / f"{dest.stem}_{counter}.json")


def _download_photo_tour(listing_id: int, path: Path):
    path = prepare_path(path, str(listing_id))

    dest = path / "photos.json"
    if dest.is_file():
        return

    variables = {"request": {"id": str(listing_id), "translateUgc": None}}
    query = {
        "operationName": ["PdpPhotoTour"],
        "locale": ["en"],
        "currency": ["EUR"],
        "variables": [json.dumps(variables)],
        "extensions": [
            '{"persistedQuery":{"version":1,"sha256Hash":"db992dc729743692ca024edde095f8594e6009bece640d76bc45cd4c82925b42"}}'
        ],
    }
    key = "PdpPhotoTour"
    try:
        photos = graphql_request(key, query)
    except DownloadError as e:
        # print(listing_id, e)
        photos = {}

    save_json(photos, dest)
    sleep()


def download_listing(listing_id: int, path: Path):
    _download_info(listing_id, path)
    _download_reviews(listing_id, path)
    _download_photo_tour(listing_id, path)


class Arguments(argtyped.Arguments):
    listings: Path
    output: Path = Path("merlin")
    num_splits: int = 10
    start: int = 0
    num_procs: int = 1


def run_downloader(args: Arguments):
    """
    Inputs:
        process: (int) number of process to run
        images_url:(list) list of images url
    """
    print(f"Running {args.num_procs} procs")
    with open(args.listings) as fid:
        listings = [int(listing) for listing in fid.readlines()]

    listings = listings[args.start :: args.num_splits]

    with Pool(args.num_procs) as pool:
        list(
            tqdm(
                pool.istarmap(  # type: ignore
                    _download_photo_tour,
                    product(listings, [args.output]),
                    chunksize=1,
                ),
                total=len(listings),
            )
        )


if __name__ == "__main__":
    args = Arguments()

    if args.num_procs > 0:
        run_downloader(args)

    else:
        with open(args.listings) as fid:
            listings = [int(listing) for listing in fid.readlines()]
        listings = listings[args.start :: args.num_splits]
        for listing in tqdm(listings):
            _download_photo_tour(listing, args.output)
