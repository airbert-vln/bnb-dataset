"""
Build a testset for Airbnb
"""
from pathlib import Path
from typing import List, Tuple, Dict, Callable
from multiprocessing import Pool
import pickle
from itertools import groupby
from operator import itemgetter
from utils.dataset.bnb_dataset import shuffle_two
import argtyped
import lmdb
import transformers
from tqdm.auto import tqdm
from helpers import load_json, save_json
from utils.dataset.bnb_features_reader import PhotoId
from utils.dataset.bnb_dataset import (
    generate_trajectory_from_listing,
    generate_trajectory_out_listing,
    generate_negative_trajectories,
    merge_images,
)
from utils.dataset.common import load_tokens


class Arguments(argtyped.Arguments):
    output: Path
    captions: Path
    num_negatives: int = 2
    min_length: int = 4
    max_length: int = 7
    min_captioned: int = 2
    max_captioned: int = 7
    max_instruction_length: int = 60
    out_listing: bool = False


def filter(items, cond_func=lambda x: True):
    """Yield items from any nested iterable"""
    for x in items:
        if cond_func(x):
            yield x


def generate_photo_ids(
    listing_id: int,
    listing_ids: List[int],
    photo_ids_by_listing: Dict[int, List[PhotoId]],
    photo_id_to_caption: Dict[int, Dict],
    min_length: int,
    max_length: int,
    min_captioned: int,
    max_captioned: int,
    num_negatives: int,
    shuffler: Callable,
    out_listing: bool,
):

    fn = (
        generate_trajectory_out_listing
        if out_listing
        else generate_trajectory_from_listing
    )
    positive_trajectory, captioned = fn(
        listing_id,
        listing_ids,
        photo_ids_by_listing,
        photo_id_to_caption,
        min_length,
        max_length,
        min_captioned,
        max_captioned,
    )

    neg_captions, neg_images, neg_randoms = generate_negative_trajectories(
        positive_trajectory,
        captioned,
        listing_ids,
        photo_ids_by_listing,
        photo_id_to_caption,
        num_negatives,
        shuffler,
    )

    return positive_trajectory, neg_captions, neg_images, neg_randoms


def collect_fixed_samples(args: Arguments):
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loading photo by listing")
    captions = load_tokens(args.captions, tokenizer, args.max_instruction_length)
    photo_id_to_caption = {int(caption["photo_id"]): caption for caption in captions}
    captions = sorted(captions, key=itemgetter("listing_id"))
    photo_ids_by_listing = {
        listing: merge_images(photos)
        for listing, photos in groupby(captions, key=itemgetter("listing_id"))
    }
    listing_ids = list(photo_ids_by_listing.keys())
    print(f"Loaded {len(listing_ids)} listings")

    testset = {
        int(l): generate_photo_ids(
            l,
            listing_ids,
            photo_ids_by_listing,
            photo_id_to_caption,
            args.min_length,
            args.max_length,
            args.min_captioned,
            args.max_captioned,
            args.num_negatives,
            shuffle_two,
            args.out_listing,
        )
        for l, photos in tqdm(photo_ids_by_listing.items())
        if len(photos) >= args.min_length or args.out_listing
    }
    print(f"Testset size is {len(testset)}")

    save_json(testset, args.output)


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string(width=80))

    collect_fixed_samples(args)
