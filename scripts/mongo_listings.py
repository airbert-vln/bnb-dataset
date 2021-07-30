from pathlib import Path
import json
from typing import List, Dict, Tuple, Iterator, Union, Optional
from dataclasses import field, dataclass, asdict
import pymongo
from tqdm.auto import tqdm
import argtyped


class Arguments(argtyped.Arguments):
    merlin: Path = Path("merlin")
    images: Path = Path("images")
    proc_id: int = 0
    num_procs: int = 20
    reset: bool = False

@dataclass
class Image:
    photo_id: int
    listing_id: int
    location: str
    state: str
    caption: str

@dataclass
class Review:
    listing_id: int
    review_id: int
    comment: str
    created_at: str
    language: str
    is_superçhost: bool

@dataclass
class Record:
    listing_id: int
    state: str
    title: str = ""
    rating: float = -1.
    person_capacity: int = -1
    property_type: str = ""
    description: str = ""
    amenities: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    location: str = ""
    lat: float = float("inf")
    lng: float = float("inf")
    neighborhood: str = ""
    getting_around: str = ""
    sleeping: List[Tuple[str, str]] = field(default_factory=list)


def load_images(room: Path) -> Iterator[Image]:
    if not room.is_dir() or not room.stem.isdigit():
        return

    listing_id = room.name
    location = room.parent.name

    try:
        data = load_json(
            room.parents[2] / "merlin" / location / listing_id / "photos.json"
        )
        photos = data["data"]["merlin"]["pdpPhotoTour"]["images"]
    except (FileNotFoundError, TypeError) as _:
        return



    for photo in photos:
        key = f"{location}/{listing_id}/{photo['id']}"
        filename = room.parents[2] / "images" / f"{key}.jpg"
        if not filename.is_file():
            continue
        caption = photo["imageMetadata"]["caption"]
        yield Image(
            photo["id"], int(room.stem), location, "downloaded", caption,
        )



def load_json(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        return json.load(fid)


def is_room_valid(room: Path) -> bool:
    return room.is_dir() and room.stem.isdigit()


def load_reviews(room: Path) -> Iterator[Review]:
    if not is_room_valid(room):
        return

    listing_id = int(room.name)
    for review_file in room.glob("reviews*.json"):
        data =  load_json(review_file)
        for review in data["data"]["merlin"]["pdpReviews"]["reviews"]:
            yield Review(listing_id, review["id"], review["comments"], review["createdAt"], review["language"], review["reviewee"]["isSuperhost"])

def load_room(room: Path) -> Optional[Record]:
    if not is_room_valid(room):
        return None

    try:
        data = load_json( room / "room1.json")
        record = Record(int(room.name), "downloaded", location=room.parent.name)

        for section in data["data"]["merlin"]["pdpSections"]:
            if section["sectionId"] == "DESCRIPTION_DEFAULT":
                record.description = section["section"]["htmlDescription"]["htmlText"]
            elif section["sectionId"] == "BOOK_IT_FLOATING_FOOTER":
                record.rating = float(section["section"]["reviewItem"]["title"])
            elif section["sectionId"] == "BOOK_IT_SIDEBAR":
                record.rating = float(section["section"]["reviewItem"]["title"])
            elif section["sectionId"] == "NAV_MOBILE":
                record.title = section["section"]["sharingConfig"]["title"]
                record.person_capacity = section["section"]["sharingConfig"]["personCapacity"]
                record.rating = section["section"]["sharingConfig"]["starRating"]
                record.property_type = section["section"]["sharingConfig"]["propertyType"]
            elif section["sectionId"] == "LOCATION_DEFAULT":
                record.location = section["section"]["previewLocationDetails"]["title"]
                record.neighborhood = section["section"]["previewLocationDetails"]["content"]["htmlText"]
                record.lat = float(section["section"]["lat"])
                record.lng = float(section["section"]["lng"])
                for detail in section["section"]["seeAllLocationDetails"]:
                    if detail["id"].startswith("getting"):
                        record.getting_around = detail["content"]["htmlText"]
            elif section["sectionId"] == "SLEEPING_ARRANGEMENT_DEFAULT":
                record.sleeping = [
                    (detail["title"], detail["subtitle"]) for detail in section["arrangementDetails"]
                ]
            elif section["sectionId"] == "AMENITIES_DEFAULT":
                record.amenities = {}
                for group in section["seeAllAmenitiesGroups"]:
                    record.amenities[group["title"]] = [
                        (detail["title"], detail["subtitle"]) for detail in group
                    ]
        return record
    except (FileNotFoundError, TypeError, json.decoder.JSONDecodeError) as _:
        return None


if __name__ == "__main__":
    args = Arguments()
    db = pymongo.MongoClient(host="diffrac").databnb

    if args.reset:
        db.rooms.delete_many({})
        db.reviews.delete_many({})

    locations = list(args.merlin.iterdir())[args.proc_id::args.num_procs]
    for location in tqdm(locations):
        rooms = list(location.iterdir())
        for room in rooms:
            if not room.is_dir() or not room.name.isdigit():
                continue

            record = load_room(room)
            if record is not None:
                db.rooms.insert_one(asdict(record), ordered=False)

            reviews = [asdict(record) for record in load_reviews(room)]
            if reviews != []:
                db.reviews.insert_many(reviews, ordered=False)

            photos = [asdict(photo) for photo in load_images(room)]
            if photos != []:
                db.images.insert_many(photos, ordered=False)
