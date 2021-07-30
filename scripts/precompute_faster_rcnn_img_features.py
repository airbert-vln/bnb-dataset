#!/usr/bin/env python3
"""
Script to precompute image features using bottom-up attention
(i.e., Faster R-CNN pretrained on Visual Genome) 
"""
import glob
import os
from pathlib import Path
import json
from collections import defaultdict
import math
import base64
import csv
import os
import pickle
import sys
from typing import List, Dict, Tuple, Iterator, Union, DefaultDict, Optional
import random
from multiprocessing import Pool
import math
from tqdm.auto import tqdm
import lmdb
import numpy as np
import torch
from torch import nn
from PIL import Image, ExifTags
import argtyped
import cv2
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from timer import Timer

random.seed(1)
csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    dry_run: bool = False
    dataloader: str = "room2room"
    config_file: str = "data/detectron_config.yaml"
    model_file: str = "data/detectron_model.pth"
    databnb: Path = Path(".")
    num_gpus: int = 50
    start_gpu: int = 0
    end_gpu: int = -1
    max_size: int = 600
    min_size: int = 600

    tsv_fieldnames: Tuple[str, ...] = (
        "scanId",
        "viewpointId",
        "image_w",
        "image_h",
        "vfov",
        "features",
        "boxes",
        "cls_prob",
        "featureViewIndex",
        "featureHeading",
        "featureElevation",
        "viewHeading",
        "viewElevation",
    )

    # Camera sweep parameters
    num_sweeps: int = 3
    views_per_sweep: int = 12
    viewpoint_size: int = 3 * 12  # Number of total views from one pano
    heading_inc: int = 360 // 12  # in degrees
    angle_margin: int = 5  # margin of error for deciding if an object is closer to the centre of another view
    elevation_start: int = -30  # Elevation on first sweep
    elevation_inc: int = 30  # How much elevation increases each sweep

    # Filesystem etc
    feature_size: int = 2048
    # You need to download this, see README.md in bottom-up-attention
    outfile: str = "img_features/ResNet-101-faster-rcnn-genome.tsv.%d"
    graphs: str = "connectivity/"

    # simulator image parameters
    width: int = 600  # max size handled by faster r-cnn model
    height: int = 600
    vfov: int = 80
    aspect: float = width / height
    hfov: float = math.degrees(2 * math.atan(math.tan(math.radians(vfov / 2)) * aspect))
    foc: float = (height / 2) / math.tan(math.radians(vfov / 2))  # focal length

    # settings for the number of features per image
    min_local_boxes: int = 5
    max_local_boxes: int = 20
    max_total_boxes: int = 100
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
    args: Arguments, img: Image
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

    assert im_scale == 1.0
    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)

    im_info = {"width": im_width, "height": im_height}

    return img, im_scale, im_info


def _process_feature_extraction(
    output: List[Dict],
    im_scales: List[float],
    im_infos: List[Dict[str, int]],
    feature_name="fc6",
    conf_thresh=0,
    background=False,
) -> List[Dict]:
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feature_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    info_list = []
    # import ipdb

    # ipdb.set_trace()

    for i in range(batch_size):
        proposal = output[0]["proposals"][i]
        #         dets = proposal.bbox / im_scales[i]
        scores = score_list[i]
        bbox = proposal.bbox  # / MAX_SIZE
        # Predict the class label using the scores
        objects = torch.argmax(scores, dim=1)
        cls_prob = torch.max(scores, dim=1)

        info_list.append(
            {
                "feat": feats[i].cpu().numpy(),
                "bbox": bbox.cpu().numpy(),
                "num_boxes": len(bbox),
                "objects": objects.cpu().numpy(),
                "image_width": im_infos[i]["width"],
                "image_height": im_infos[i]["height"],
                "im_scale": im_scales[i],
                "cls_prob": scores.cpu().numpy(),
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

    feat_list = _process_feature_extraction(output, im_scales, im_infos,)

    return feat_list


def load_viewpointids(args: Arguments, proc_id=0):
    viewpointIds = []
    with open(args.graphs + "scans.txt") as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(args.graphs + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    random.shuffle(viewpointIds)
    if args.num_gpus != 1:
        viewpointIds = viewpointIds[proc_id :: args.num_gpus]
    print("%d: Loaded %d viewpoints" % (proc_id, len(viewpointIds)))
    return viewpointIds


def get_detections_from_im(
    args: Arguments, record: Dict, model: nn.Module, im: np.ndarray,
):

    if "features" not in record:
        ix = 0  # First view in the pano
    elif record["featureViewIndex"].shape[0] == 0:
        ix = 0  # No detections in pano so far
    else:
        ix = int(record["featureViewIndex"][-1]) + 1

    # Code from bottom-up and top-down
    # scores, boxes, attr_scores, rel_scores = im_detect(net, im)
    info_list = get_detectron_features(args, model, [im])
    info = info_list[0]
    # import ipdb
    # ipdb.set_trace()

    cls_boxes = info["bbox"]
    cls_prob = info["cls_prob"]
    pool5 = info["feat"]

    # Keep only the best detections
    max_conf = np.zeros((info["num_boxes"]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = cls_prob[:, cls_ind]
        dets = torch.tensor(info["bbox"].astype(np.float32))
        keep = nms(dets, torch.tensor(cls_scores), args.nms_thresh).numpy()
        max_conf[keep] = np.where(
            cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
        )

    keep_boxes = np.where(max_conf >= args.conf_thresh)[0]
    if len(keep_boxes) < args.min_local_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][: args.min_local_boxes]
    elif len(keep_boxes) > args.max_local_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][: args.max_local_boxes]

    # Discard any box that would be better centered in another image
    # threshold for pixel distance from center of image
    hor_thresh = args.foc * math.tan(
        math.radians(args.heading_inc / 2 + args.angle_margin)
    )
    vert_thresh = args.foc * math.tan(
        math.radians(args.elevation_inc / 2 + args.angle_margin)
    )
    center_x = 0.5 * (info["bbox"][:, 0] + info["bbox"][:, 2])
    center_y = 0.5 * (info["bbox"][:, 1] + info["bbox"][:, 3])
    reject = (center_x < args.width / 2 - hor_thresh) | (
        center_x > args.width / 2 + hor_thresh
    )
    heading = record["viewHeading"][ix]
    elevation = record["viewElevation"][ix]
    if ix >= args.views_per_sweep:  # not lowest sweep
        reject |= center_y > args.height / 2 + vert_thresh
    if ix < args.viewpoint_size - args.views_per_sweep:  # not highest sweep
        reject |= center_y < args.height / 2 - vert_thresh
    keep_boxes = np.setdiff1d(keep_boxes, np.argwhere(reject))

    # Calculate the heading and elevation of the center of each observation
    featureHeading = heading + np.arctan2(
        center_x[keep_boxes] - args.width / 2, args.foc
    )
    # normalize featureHeading
    featureHeading = np.mod(featureHeading, math.pi * 2)
    # force it to be the positive remainder, so that 0 <= angle < 360
    featureHeading = np.expand_dims(
        np.mod(featureHeading + math.pi * 2, math.pi * 2), axis=1
    )
    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    featureHeading = np.where(
        featureHeading > math.pi, featureHeading - math.pi * 2, featureHeading
    )
    featureElevation = np.expand_dims(
        elevation + np.arctan2(-center_y[keep_boxes] + args.height / 2, args.foc),
        axis=1,
    )

    # Save features, etc
    if "features" not in record:
        record["boxes"] = cls_boxes[keep_boxes]
        record["cls_prob"] = cls_prob[keep_boxes]
        record["features"] = pool5[keep_boxes]
        record["featureViewIndex"] = (
            np.ones((len(keep_boxes), 1), dtype=np.float32) * ix
        )
        record["featureHeading"] = featureHeading
        record["featureElevation"] = featureElevation
    else:
        record["boxes"] = np.vstack([record["boxes"], cls_boxes[keep_boxes]])
        record["cls_prob"] = np.vstack([record["cls_prob"], cls_prob[keep_boxes]])
        record["features"] = np.vstack([record["features"], pool5[keep_boxes]])
        record["featureViewIndex"] = np.vstack(
            [
                record["featureViewIndex"],
                np.ones((len(keep_boxes), 1), dtype=np.float32) * ix,
            ]
        )
        record["featureHeading"] = np.vstack([record["featureHeading"], featureHeading])
        record["featureElevation"] = np.vstack(
            [record["featureElevation"], featureElevation]
        )
    return


def filter(record: Dict, max_boxes: int):
    # Remove the most redundant features (that have similar heading, elevation and
    # are close together to an existing feature in cosine distance)
    feat_dist = pairwise_distances(record["features"], metric="cosine")
    # Heading and elevation diff
    heading_diff = pairwise_distances(record["featureHeading"], metric="euclidean")
    heading_diff = np.minimum(heading_diff, 2 * math.pi - heading_diff)
    elevation_diff = pairwise_distances(record["featureElevation"], metric="euclidean")
    feat_dist = feat_dist + heading_diff + elevation_diff  # Could add weights
    # Discard diagonal and upper triangle by setting large distance
    feat_dist += 10 * np.identity(feat_dist.shape[0], dtype=np.float32)
    feat_dist[np.triu_indices(feat_dist.shape[0])] = 10.0
    ind = np.unravel_index(np.argsort(feat_dist, axis=None), feat_dist.shape)
    # Remove indices of the most similar features (in appearance and orientation)
    keep = set(range(feat_dist.shape[0]))
    ix = 0
    while len(keep) > max_boxes:
        i = ind[0][ix]
        j = ind[1][ix]
        if i not in keep or j not in keep:
            ix += 1
            continue
        if record["cls_prob"][i, 1:].max() > record["cls_prob"][j, 1:].max():
            keep.remove(j)
        else:
            keep.remove(i)
        ix += 1
    # Discard redundant features
    for k, v in record.items():
        if k in [
            "boxes",
            "cls_prob",
            "features",
            "featureViewIndex",
            "featureHeading",
            "featureElevation",
        ]:
            record[k] = v[sorted(keep)]


def _load_simulator():
    import MatterSim
    # Set up the simulator
    sim = MatterSim.Simulator()
    sim.setCameraResolution(args.width, args.height)
    sim.setCameraVFOV(math.radians(args.vfov))
    sim.setDiscretizedViewingAngles(False)
    sim.setBatchSize(1)
    sim.setPreloadingEnabled(False)
    sim.initialize()
    return sim


class Dataloader:
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Tuple[Dict, List[np.ndarray]]]:
        raise NotImplementedError()


class Room2Room(Dataloader):
    def __init__(self, args: Arguments, proc_id: int):
        self.sim = _load_simulator()
        # Loop all the viewpoints in the simulator
        self.viewpointIds = load_viewpointids(args, proc_id)
        self.args = args

    def __len__(self):
        return len(self.viewpointIds)

    def __iter__(self) -> Iterator[Tuple[Dict, List[np.ndarray]]]:
        for scanId, viewpointId in self.viewpointIds:
            ims = []
            self.sim.newEpisode(
                [scanId], [viewpointId], [0], [math.radians(self.args.elevation_start)]
            )
            for ix in range(args.viewpoint_size):
                state = self.sim.getState()[0]
                img = np.array(state.rgb, copy=True)
                ims.append(img)

                # Build state
                if ix == 0:
                    record = {
                        "scanId": state.scanId,
                        "viewpointId": state.location.viewpointId,
                        "viewHeading": np.zeros(args.viewpoint_size, dtype=np.float32),
                        "viewElevation": np.zeros(
                            args.viewpoint_size, dtype=np.float32
                        ),
                        "image_h": args.height,
                        "image_w": args.width,
                        "vfov": args.vfov,
                    }
                record["viewHeading"][ix] = state.heading
                record["viewElevation"][ix] = state.elevation

                # Move the sim viewpoint so it ends in the same place
                elev = 0.0
                heading_chg = math.pi * 2 / args.views_per_sweep
                view = ix % self.args.views_per_sweep
                sweep = ix // self.args.views_per_sweep
                if view + 1 == self.args.views_per_sweep:  # last viewpoint in sweep
                    elev = math.radians(args.elevation_inc)
                self.sim.makeAction([0], [heading_chg], [elev])
            yield record, ims


def load_json(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        return json.load(fid)


def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid)


def lmdb_batch_write(keys: List, seq: List, env: lmdb.Environment):
    """ Writing is faster when the seq is sorted """
    seq = sorted(seq)
    with env.begin(write=True) as txn:
        for key, el in zip(keys, seq):
            lmdb_write(key, el, txn)

def lmdb_write(key, value, txn: lmdb.Transaction):
    txn.put(key=str(key).encode("utf-8", "ignore"), value=pickle.dumps(value))


def build_databnb_lmdb(lmdb_file: Path, database: Path):
    locations = list((database / "merlin").iterdir())
    env = lmdb.open(str(lmdb_file), map_size=int(1e13), writemap=True, map_async=True, max_dbs=0)
    keys = []

    with env.begin(write=True) as txn:
        for location in tqdm(locations):
            for room in location.iterdir():
                if not room.is_dir() or not room.stem.isdigit():
                    continue

                try:
                    data = load_json(room / "photos.json")
                    photos = data["data"]["merlin"]["pdpPhotoTour"]["images"]
                except FileNotFoundError:
                    continue

                for photo in photos:
                    key = f"{location.stem}/{room.stem}/{photo['id']}"
                    filename = database / "images" / f"{key}.jpg"
                    if not filename.is_file():
                        lmdb_write(key, {"state": "missing"}, txn) 
                        keys.append(key)
                    im = Image.open(filename)
                    exif = {
                        ExifTags.TAGS[k]: v
                        for k, v in im._getexif().items()
                        if k in ExifTags.TAGS
                    } if im._getexif is not None else {}
                    lmdb_write(key, {
                        "state": "downloaded",
                        "caption": photo["imageMetadata"]["caption"],
                        "image": np.array(im),
                        "exif": exif,
                   }, txn)
                    keys.append(key)
        lmdb_write("keys", keys, txn)


class DataBnB(Dataloader):
    def __init__(self, args: Arguments, proc_id: int):
        self.folder = Path(args.databnb)
        self.proc_id = proc_id
        self.num_gpus = args.num_gpus

        lmdb_file = self.folder / "db" / "images.lmdb"
        print(lmdb_file)
        if not lmdb_file.is_dir():
            build_databnb_lmdb(lmdb_file, self.folder)

        self.env = lmdb.open(str(lmdb_file), readonly=True)

    def __len__(self):
        return self.env.stat()["entries"]

    def __iter__(self) -> Iterator[Tuple[Dict, List[np.ndarray]]]:
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, record in cursor:
                image = record.pop("image")
                yield record, [image]

def build_tsv(args: Arguments, proc_id: int):
    if args.dataloader == "room2room":
        dataloader: Dataloader = Room2Room(args, proc_id)
    elif args.dataloader == "databnb":
        dataloader = DataBnB(args, proc_id)
        print(len(dataloader))
    else:
        raise ValueError(f"Unknown dataloader for {args.dataloader}")

    model = _build_detection_model(args.config_file, args.model_file)

    count = 0
    t_net = Timer()
    with open(args.outfile % proc_id, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=args.tsv_fieldnames)
        for record, ims in dataloader:
            t_net.tic()

            # Run detection
            for im in ims:
                get_detections_from_im(args, record, model, im)

            if args.dry_run:
                print(
                    "%d: Detected %d objects in pano"
                    % (proc_id, record["features"].shape[0])
                )

            filter(record, args.max_total_boxes)

            if args.dry_run:
                print(
                    "%d: Reduced to %d objects in pano"
                    % (proc_id, record["features"].shape[0])
                )

            for k, v in record.items():
                if isinstance(v, np.ndarray):
                    record[k] = str(base64.b64encode(v), "utf-8")
            writer.writerow(record)
            count += 1
            t_net.toc()
            if count % 10 == 0:
                print(
                    "%d: Processed %d / %d viewpoints, %.1fs avg net time, projected %.1f hours"
                    % (
                        proc_id,
                        count,
                        len(dataloader),
                        t_net.average_time,
                        (t_net.average_time) * len(dataloader) / 3600,
                    )
                )
                if args.dry_run:
                    return


if __name__ == "__main__":
    args = Arguments()
    end_gpu = args.end_gpu if args.end_gpu > 0 else args.num_gpus
    num_procs = end_gpu - args.start_gpu

    if num_procs == 1:
        build_tsv(args, args.start_gpu)

    elif num_procs > 1:
        p = Pool(args.num_gpus)
        p.starmap(
            build_tsv, [(args, proc_id) for proc_id in range(args.start_gpu, end_gpu)]
        )
