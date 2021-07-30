# :houses: BnB Dataset :houses:

[![MIT](https://img.shields.io/github/license/airbert-vln/bnb-dataset)](./LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-green.svg)](https://arxiv.org/abs/<INDEX>)
[![R2R 1st](https://img.shields.io/badge/R2R-🥇-green.svg)](https://eval.ai/web/challenges/challenge-page/97/leaderboard/270)

This contains a set of scripts for downloading a dataset from Airbnb.


## :hammer_and_wrench: 1. Get started

You need to have a recent version of Python (higher than 3.6) and install dependencies:

```bash
pip install -r requirements
```

Note that typing is extensively used in the script. This was a real time saver for detecting errors before runtime. You might want to setup properly your IDE to play well with mypy. I  recommend the [`coc.nvim`](https://github.com/neoclide/coc.nvim) extension [`coc-pyright`](https://github.com/fannheyward/coc-pyright) for [neovim](https://github.com/neovim/neovim/) users.

Managing a large of images is tricky and usually take a lot of times. Usually, the scripts are splitting the task among several workers. A cache folder is keeping the order list for each worker, while each worker is producing its own output file.
Look for `num_workers` or `num_procs` parameters in the `argtyped Arguments`.

To parallelize among multiple GPUs, you can use the `distributed.launcher` script as is:

```bash
python  -m torch.distributed.launch \
  --nproc_per_node=24 \
  --nnodes=1 \
  --node_rank=0 \
  -m scripts.extract_noun_phrases \
    --start 24 \
    --num_splits 48 \
    --source data/bnb-test-indoor.tsv \
    --output noun_phrases.tsv \
    --num_workers 8 \
    --batch_size 20

```




## :world_map: 2. Download listings from Airbnb

This step is building a TSV file with 4 columns: listing ID, photo ID, image URL, image caption.
A too high request rate would induce a rejection from Airbnb. Instead, it is advised to split the job among different IP addresses.

You can use the pre-computed TSV file used in our paper [for training](./data/bnb-train.tsv) and [for testing](./data/bnb-test.tsv). 
Note that this file contains only a portion from Airbnb listings, and some images might not be available anymore.

### 2.1. Create a list of regions

Airbnb listings are searched among a specific region. 
We need first to initialize the list of regions. A quick hack for that consists in scrapping Wikipedia list of places, as done in the script [cities.py](./scripts/cities.py):

```bash
pip install selenium
# Install a driver following selenium specific instructions for your platform:
# https://selenium-python.readthedocs.io/installation.html
python scripts/cities.py --output data/cities.txt
```

You can other examples  in the [`scripts/locations/`](./scripts/locations/) folder.

### 2.2. Download listings

```bash
# Download a list of listing from the list of cities
python scripts/search_listings.py --locations data/cities.txt --output data/listings
# FIXME I think this script is broken

# Download JSON files for each listing
python scripts/download_listings.py --listings data/listings.txt --output data/merlin

# Extract photo URLs from listing export files
python scripts/extract_photo_metadata.py --merlin data/merlin --output data/bnb-dataset-raw.tsv
```

### 2.3. Filter captions

```bash
# Apply basic rules to remove some captions
python scripts/filter_captions.py --input data/bnb-dataset-raw.tsv --output data/bnb-dataset.tsv
```

## :camera_flash: 3. Get images

Now we want to download images and filter out outdoor images.


### 3.1. Download images

The download rate can be higher before the server kicks us out. However, it is still preferable to use a pool of IP addresses.

```bash
python scripts/download_images.py --csv_file data/bnb-dataset.tsv --output data/images --correspondance /tmp/cache-download-images/
```


### 3.2. Optionally, make sure images were correctly downloaded

```bash
python scripts/detect_errors.py --images data/images --merlin data/merlin
```

### 3.3. Filter out outdoor images

Outdoor images tend to be of lower qualities and captions are often not relevant. 
We first detect outdoor images from a CNN pretrained on the places365 dataset. Later on, we will keep indoor images.

Note that the output of this step is also used for image merging.

```bash
# Detect room types
python scripts/detect_room.py --output data/places365/detect.tsv --images data/images

# Filter out indoor images
python scripts/extract_indoor.py --output data/bnb-dataset-indoor.tsv --detection data/places365/detect.tsv
```





## :minidisc: 4. Build an LMDB database with BnB pictures

Extract visual features and store them on a single file. Several steps are required to achieve that. Unfortunately, we don't own permissions over Airbnb images, and thus we are not permitted to share our own LMDB file.

### 4.1. Split between train and test

```bash
python scripts/preprocess_dataset.py --csv datasets/data/bnb-dataset.indoor.tsv
```

### 4.2. Extract bottom-up top-down features

This step is one of the most annoying one, since the install of bottom-up top-down attention is outdated. I put docker file and Singularity definition file in the folder `container` to help you with that.
Note that this step is also extremely slow and you might want to use multiple GPUs.

```bash
python scripts/precompute_airbnb_img_features_with_butd.py  --images data/images
```

### 4.3. Build an LMDB file


```bash
# Extract keys
python scripts/extract_keys.py --output data/keys.txt --datasets data/bnb-dataset.indoor.tsv
# Create an LMDB
python scripts/convert_to_lmdb.py --output img_features --keys data/keys.txt
```

Note that you can split the LMDB into multiple files by using a number of workers. This could be relevant when your LMDB file is super huge!

## :link: 5. Create dataset files with path-instruction pairs

Almost there! We built  image-caption pairs and now we want to convert them into path-instruction pairs.
Actually, we are just going to produce  JSON files that you can feed into the [training repository](https://github.com/airbert-vln/airbert/).

### :chains: 5.1. Concatenation




### :busts_in_silhouette: 5.2. Image merging

### 👨‍👩‍👧 5.3. Captionless insertion

### 👣 5.4. Instruction rephrasing

```bash
# Extract noun phrases from BnB captions
python scripts/extract_noun_phrases.py --source data/bnb-train.tsv --output data/bnb-train.np.tsv 
python scripts/extract_noun_phrases.py --source data/bnb-test.tsv --output data/bnb-test.np.tsv 

# Extract noun phrases from R2R train set
python scripts/extract_noun_phrases.py --source data/bnb-dataset.indoor.tsv --output data/bnb-train.np.tsv 

```

