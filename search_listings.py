"""
Download list of rooms
"""
import math
import csv
import json
from typing import Union, Dict, List
from pathlib import Path
from multiprocessing.pool import Pool
import socket
import signal
import tap
from tqdm.auto import tqdm, trange
from helpers import graphql_request, get_slug, save_json

socket.setdefaulttimeout(1)

# Intercept Ctrl-C to exit gracefully
stop = False


def signal_handler(signal_received, frame):
    global stop
    stop = True


signal.signal(signal.SIGINT, signal_handler)


class Arguments(tap.Tap):
    locations: Path  # txt files containing one location per line
    correspondance: Path = Path("correspondance_listing")
    output: Path = Path("results_listing")
    num_procs: int = 5
    num_splits: int = 40
    start: int = 0


def search_page(name: str, offset: int, limit: int):
    variables = {
        "request": {
            "query": name,
            "metadataOnly": False,
            "version": "1.7.9",
            "itemsPerGrid": limit,
            "itemsOffset": offset,
            "refinementPaths": ["/homes"],
        }
    }

    query = {
        "operationName": ["ExploreSearch"],
        "locale": ["en"],
        "currency": ["EUR"],
        "variables": [json.dumps(variables)],
        "extensions": [
            '{"persistedQuery":{"version":1,"sha256Hash":"1816e0a81cc05d7e63f9a3817dc1e3e8a2e3a9bb5dafdf6dbc212b9aa2880391"}}'
        ],
    }
    results = graphql_request("ExploreSearch", query)

    return results


def extract_listings(data: Dict) -> List[int]:
    for section in data["data"]["dora"]["exploreV3"]["sections"]:
        if section["__typename"] == "DoraExploreV3ListingsSection":
            return [int(l["listing"]["id"]) for l in section["items"]]
    return []


def search_location(name: str, dest: Path, limit: int):
    # print(name)
    if (dest / "listings.txt").is_file():
        return

    data = search_page(name, 0, limit)
    listings = []

    # Sometimes Airbnb says that they are no results, whereas it is going to find some if we retry
    for i in range(2):
        listings += extract_listings(data)
        if listings != []:
            break

    num_listings = 0

    pagination = data["data"]["dora"]["exploreV3"]["metadata"]["paginationMetadata"]
    num_listings = (
        int(pagination["totalCount"]) if pagination["totalCount"] is not None else 0
    )

    # assert pagination["pageLimit"] == limit, pagination

    for counter in range(limit, int(num_listings), limit):
        data = search_page(name, counter, limit)
        listings += extract_listings(data)

    dest.mkdir(exist_ok=True, parents=True)
    with open(dest / "listings.txt", "w") as fid:
        fid.write("\n".join([str(l) for l in listings]))


def search_locations(location_file: Union[Path, str], limit: int = 50):
    # print(location_file)

    with open(location_file, "r") as fid:
        num_rows = sum(1 for _ in fid.readlines())

    with open(location_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=("name", "dest"))
        for row in reader:
            search_location(row["name"], Path(row["dest"]), limit)


def make_correspondance(args: Arguments):
    """
    CSV file: location name\t path/to/location
    """
    print(f"Running {args.num_splits} splits")
    args.correspondance.mkdir(parents=True, exist_ok=True)

    with open(args.locations, "r") as fid:
        locations = [l.strip() for l in fid.readlines()]

    per_split = math.ceil(len(locations) / args.num_splits)
    counter = args.start
    for split_id in trange(args.num_splits):
        if stop:
            break

        correspondance_file = args.correspondance / f"listings.part-{split_id}.tsv"
        end = min(len(locations), counter + per_split)

        with open(correspondance_file, "w") as f:
            for i in range(counter, end):
                if stop:
                    break
                path_to_location = args.output / get_slug(locations[i])
                f.write("\t".join([locations[i], str(path_to_location)]))
                f.write("\n")

            counter += per_split


def run_downloader(args: Arguments):
    """
    Inputs:
        process: (int) number of process to run
        images_url:(list) list of images url
    """
    if args.num_procs == 0:
        print(f"Running without parallelization")
        correspondances = sorted(list(args.correspondance.iterdir()))[args.start :]
        for correspondance in tqdm(correspondances):
            search_locations(correspondance)
    else:
        print(f"Running {args.num_procs} procs")
        correspondances = sorted(list(args.correspondance.iterdir()))[args.start :]
        with Pool(args.num_procs) as pool:
            list(
                tqdm(
                    pool.imap(search_locations, correspondances, chunksize=1),
                    total=len(correspondances),
                )
            )


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    if not args.correspondance.is_dir() or list(args.correspondance.glob("*.tsv")) == []:
        print("Making correspondance")
        make_correspondance(args)

    download = args.output
    download.mkdir(exist_ok=True)

    # search_location("new york city", Path("test"), 50)
    run_downloader(args)
