# :houses: BnB Dataset :houses:

[![MIT](https://img.shields.io/github/license/airbert-vln/bnb-dataset)](./LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-green.svg)](https://arxiv.org/abs/<INDEX>)
[![R2R 1st](https://img.shields.io/badge/R2R-🥇-green.svg)](https://eval.ai/web/challenges/challenge-page/97/leaderboard/270)

This contains a set of scripts for downloading a dataset from Airbnb.


## :hammer_and_wrench: Getting started

You need to have a recent version of Python (higher than 3.6) and install dependencies:

```bash
pip install -r requirements
```


## :world_map: Downloading listings from Airbnb

This step is building a TSV file with 4 columns: listing ID, photo ID, image URL, image caption.
A too high request rate would induce a rejection from Airbnb. Instead, it is advised to split the job among different IP addresses.

You can use the pre-computed TSV file used in our paper [for training](./data/bnb-train.tsv) and [for testing](./data/bnb-test.tsv). 
Note that this file contains only a portion from Airbnb listings, and some images might not be available anymore.

1. Create a list of regions
2. Download listings
3. Filter out 
4. Split between train and test


## :minidisc: Building an LMDB database with BnB pictures

Now we want to download images, extract visual features and store them on a single file. Several steps are required to achieve that. Unfortunately, we don't own permissions over Airbnb images, and thus we are not permitted to share our own LMDB file.

1. Download images
2. Extract visual features
3. Build an LMDB file


## :link: Create dataset files with path-instruction pairs

Now, we describe how you can create your dataset files to load into the [model repository](https://github.com/airbert-vln/airbert/).

### :chains: Concatenation


### :busts_in_silhouette: Image merging

### 👨‍👩‍👧 Captionless insertion

### 👣 Instruction rephrasing

