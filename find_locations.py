"""
Download list of rooms
"""
from pathlib import Path
import json
import argtyped
import requests
from tqdm.auto import tqdm
from helpers import graphql_request


def find_rooms(location):
    variables = {
        "request": {
            "metadataOnly": False,
            "version": "1.7.9",
            "itemsPerGrid": 20,
            "adults": 1,
            "searchType": "autocomplete_click",
            "refinementPaths": ["/homes"],
            "tabId": "home_tab",
            #       'placeId': 'ChIJOwg_06VPwokRYv534QaPC8g',
            "source": "structured_search_input_header",
            "query": location,
            "cdnCacheSafe": False,
            "simpleSearchTreatment": "simple_search_only",
            "treatmentFlags": [
                "simple_search_1_1",
                "simple_search_desktop_v3_full_bleed",
                "flexible_dates_options_extend_one_three_seven_days",
            ],
            "screenSize": "large",
        }
    }

    query = {
        "operationName": ["ExploreSearch"],
        "locale": ["en"],
        "currency": ["EUR"],
        "variables": [json.dumps(variables)],
        "extensions": [
            '{"persistedQuery":{"version":1,"sha256Hash":"13aa9971e70fbf5ab888f2a851c765ea098d8ae68c81e1f4ce06e2046d91b6ea"}}'
        ],
        "_cb": ["1lruxnglh3w7o"],
    }
    results = graphql_request("ExploreSearch", query)

    room_ids = []
    locations = []
    for section in results["data"]["dora"]["exploreV3"]["sections"]:
        for item in section["items"]:
            if "listing" not in item:
                continue
            room_ids.append(item["listing"]["contextualPictures"][0]["id"])
            locations.append(item["listing"]["publicAddress"])

    return room_ids, list(set(locations))


room_ids = set()
with open("cities.txt") as fid:
    cities = json.load(fid)
todo = [f"{city}, United States" for city in cities]
done = []
err = []


class TqdmCustom(tqdm):
    def update_custom(self):
        self.total = len(todo)
        return self.update(len(done) - self.n)


class Arguments(argtyped.Arguments):
    cities: Path
    listings: Path = Path("data/listings.json")


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=100))

    room_ids = set()
    with open(args.cities) as fid:
        cities = json.load(fid)

    todo = []
    done = []
    err = []

    with TqdmCustom() as t:
        counter = 0

        while todo != []:
            location = todo.pop()
            try:
                new_rooms, new_locations = find_rooms(location)
            except:
                err.append(location)
                continue

            done.append(location)

            room_ids.update(new_rooms)

            for location in new_locations:
                if "United States" not in location:
                    continue
                if location not in done and location not in todo and location not in err:
                    todo.append(location)

            t.update_custom()
            counter += 1

            if counter % 50:
                counter = 0

                with open(args.listings, "w") as fid:
                    json.dump(list(room_ids), fid)

    with open(args.listings, "w") as fid:
        json.dump(list(room_ids), fid)

    with open("errors.json", "w") as fid:
        json.dump(err, fid)

    with open("done.json", "w") as fid:
        json.dump(done, fid)
