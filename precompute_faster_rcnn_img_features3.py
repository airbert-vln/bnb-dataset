#!/usr/bin/env python3
"""
Script to precompute image features using bottom-up attention
(i.e., Faster R-CNN pretrained on Visual Genome) 
"""
from pathlib import Path
import json
import tarfile
import shutil
from dataclasses import dataclass, field, asdict
import pickle
from typing import List, Dict, Tuple, Iterator, Union, Optional
from tqdm.auto import tqdm
import lmdb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ExifTags
import argtyped
import cv2
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor

Image.MAX_IMAGE_PIXELS = None


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


@dataclass
class Record:
    photo_id: int
    listing_id: int
    image: Optional[Image.Image] = None
    path: Path = Path(".")
    bbox: np.ndarray = field(default_factory=lambda: np.array([]))
    feature: np.ndarray = field(default_factory=lambda: np.array([]))
    cls_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    num_boxes: int = -1
    image_width: int = -1
    image_height: int = -1

    # detectron2
    config_file: str = "config/faster_rcnn_R_101_FPN_3x.yaml"
    confidence_threshold: float = 0.5


class Arguments(argtyped.Arguments):
    dry_run: bool = False
    config_file: str = "data/detectron_config.yaml"
    model_file: str = "data/detectron_model.pth"
    todo: Optional[Path] = None
    batch_size: int = 1
    images: Path = Path("images")
    num_procs: int = 10
    proc_id: int = 0
    num_workers: int = 5

    # settings for the number of features per image
    max_size: int = 1333
    min_size: int = 800
    max_boxes: int = 100
    nms_thresh: float = 0.3  # same as bottom-up
    conf_thresh: float = 0.4  # increased from 0.2 in bottom-up paper


def _build_detection_model(config_file: str, model_file: str) -> nn.Module:
    cfg.merge_from_file(config_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(model_file, map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to("cuda")
    model.eval()
    return model


def _image_transform(
    args: Arguments, img: Image.Image
) -> Tuple[torch.Tensor, float, Dict[str, int]]:
    im = np.array(img).astype(np.float32)
    if len(im.shape) < 3:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    #     im = im[:, :, ::-1] # bgr --> rgb
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_height = im_shape[0]
    im_width = im_shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # Scale based on minimum size
    im_scale = args.min_size / im_size_min

    # Prevent the biggest axis from being more than max_size
    # If bigger, scale it down
    if np.round(im_scale * im_size_max) > args.max_size:
        im_scale = args.max_size / im_size_max

    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    timg = torch.from_numpy(im).permute(2, 0, 1)

    im_info = {"width": im_width, "height": im_height}

    return timg, im_scale, im_info


def _process_feature_extraction(
    output,
    im_scales,
    im_infos,
    num_features: int,
    feature_name="fc6",
    conf_thresh=0,
    background=False,
):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [F.softmax(x, -1) for x in score_list]
    feats = output[0][feature_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    info_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]
        max_conf = torch.zeros((scores.shape[0])).to(cur_device)
        conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
        start_index = 1
        # Column 0 of the scores matrix is for the background class
        if background:
            start_index = 0
        for cls_ind in range(start_index, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(
                # Better than max one till now and minimally greater than conf_thresh
                (cls_scores[keep] > max_conf[keep])
                & (cls_scores[keep] > conf_thresh_tensor[keep]),
                cls_scores[keep],
                max_conf[keep],
            )

        sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
        num_boxes = (sorted_scores[:num_features] != 0).sum()
        keep_boxes = sorted_indices[:num_features]
        bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
        # Predict the class label using the scores
        objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
        # cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

        info_list.append(
            {
                "bbox": bbox.cpu().numpy(),
                "features": feats[i][keep_boxes].cpu().numpy(),
                "num_boxes": num_boxes.item(),
                "objects": objects.cpu().numpy(),
                "image_width": im_infos[i]["width"],
                "image_height": im_infos[i]["height"],
                "cls_prob": scores[keep_boxes].cpu().numpy(),
            }
        )

    return info_list


def get_detectron_features(
    args: Arguments, model: nn.Module, images: List[Image.Image]
) -> List[Dict]:
    img_tensor, im_scales, im_infos = [], [], []

    for image in images:
        im, im_scale, im_info = _image_transform(args, image)
        img_tensor.append(im)
        im_scales.append(im_scale)
        im_infos.append(im_info)

    # Image dimensions should be divisible by 32, to allow convolutions
    # in detector to work
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to("cuda")

    with torch.no_grad():
        output = model(current_img_list)

    info_list = _process_feature_extraction(output, im_scales, im_infos, args.max_boxes)

    return info_list


def load_json(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        return json.load(fid)


def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid)


async def lmdb_batch_write(keys: List, seq: List, env: lmdb.Environment):
    """Writing is faster when the seq is sorted"""
    seq = sorted(seq)
    with env.begin(write=True) as txn:
        for key, el in zip(keys, seq):
            lmdb_write(key, el, txn)


def lmdb_write(key, value, txn: lmdb.Transaction):
    txn.put(key=str(key).encode("utf-8", "ignore"), value=pickle.dumps(value))


def search_locations(image_folder: Path) -> List[Path]:
    return [f for f in image_folder.iterdir() if f.is_dir()]


def load_photo_paths(locations: List[Path]) -> List[str]:
    photos = []
    for location in tqdm(locations):
        listings = list((location).iterdir())
        for listing in listings:
            for photo in listing.glob("*.jpg"):
                if (listing / f"{photo.stem}.pkl").is_file():
                    continue
                photos.append(str(photo))
    return photos


class ImageDataset(Dataset):
    def __init__(self, photos: List[str]):
        self.photos = photos

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, index: int) -> Record:
        path = Path(self.photos[index])
        listing_id = int(path.parent.name)
        photo_id = int(path.stem)
        # image = Image.open(path)
        # image = image.convert("RGB")
        image: np.ndarray = read_image(path, format="BGR")
        return Record(photo_id, listing_id, image, path)


def build_features(args: Arguments):
    # model = _build_detection_model(args.config_file, args.model_file)

    cfg = setup_cfg(args)
    setup_logger(name="fvcore")
    model = DefaultPredictor(cfg)

    if isinstance(args.todo, Path) and args.todo.is_file():
        with open(args.todo, "r") as fid:
            photos = fid.read().splitlines()
    else:
        locations = search_locations(args.images)
        photos = load_photo_paths(locations)
        if args.todo is not None:
            with open(args.todo, "w") as fid:
                fid.writelines(f"{l}\n" for l in photos)

    photos = photos[args.proc_id :: args.num_procs]
    dataset = ImageDataset(photos)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,
        num_workers=args.num_workers,
    )

    records: List[Record]
    for records in tqdm(dataloader):
        images = []
        for record in records:
            if not (record.path.parent / f"{record.path.stem}.pkl").is_file():
                images.append(record.image)
        if images == []:
            continue

        # infos = get_detectron_features(args, model, images)
        infos = model(images)

        for record, info in zip(records, infos):
            item = {
                "listing_id": record.listing_id,
                "photo_id": record.photo_id,
                "bbox": info["bbox"],
                "image_width": info["image_width"],
                "image_height": info["image_height"],
                "feature": info["features"],
                "cls_prob": info["cls_prob"],
            }
            with open(record.path.parent / f"{record.path.stem}.pkl", "wb") as fid:
                pickle.dump(item, fid)


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string())
    build_features(args)
