from typing import List, Dict, Callable, Optional, Tuple, Iterator, Any, Set
from typing_extensions import Literal
from itertools import combinations, product
import os
import re
import copy
from collections import OrderedDict, defaultdict
import string
import functools
import random
from dataclasses import dataclass, field
from pathlib import Path
import json
import argtyped
from tqdm.auto import tqdm
import networkx as nx
from transformers import PreTrainedTokenizer, AutoTokenizer
import numpy as np
import torch
from torch import nn
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer as AllenTokenizer
import allennlp_models.structured_prediction
from allennlp_models.pretrained import load_predictor
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import spacy
from spacy.tokenizer import Tokenizer as SpacyTokenizer
from spacy.tokens import Token
from utils.dataset.pano_features_reader import PanoFeaturesReader
from highlight_sentence import POSHighlighter
from vln_bert import VLNBert
from vilbert.vilbert import BertConfig
from utils.dataset.common import get_headings, load_nav_graphs

random.seed(0)
nlp = spacy.load("en")
# nltk.download('stopwords')


def read_dataset(infile: Path):
    with open(infile) as fid:
        return json.load(fid)


def save_dataset(dataset: Any, outfile: Path):
    with open(outfile, "w") as fid:
        return json.dump(dataset, fid, indent=2)


@dataclass
class Sample:
    instr: str
    path: List[str]
    scan: str
    heading: float


@dataclass
class TokenPerturbation:
    text: str
    span: Tuple[int, int]
    mode: str = "NONE"
    cand: List[str] = field(default_factory=list)


def fill_with_none(
    sentence: str, tokens: List[TokenPerturbation]
) -> List[TokenPerturbation]:
    filled: List[TokenPerturbation] = []
    cursor = 0
    for token in tokens:
        start, end = token.span
        if start > cursor:
            filled.append(
                TokenPerturbation(sentence[cursor:start], (cursor, start - 1), "NONE")
            )
        cursor = end + 1
        filled.append(token)
    if cursor < len(sentence):
        filled.append(
            TokenPerturbation(sentence[cursor:], (cursor, len(sentence) - 1), "NONE")
        )
    return filled


def random_order_cartesian_product(*factors):
    """https://stackoverflow.com/a/53895551/4986615"""
    amount = functools.reduce(lambda prod, factor: prod * len(list(factor)), factors, 1)
    index_linked_list = [None, None]
    for max_index in reversed(range(amount)):
        index = random.randint(0, max_index)
        index_link = index_linked_list
        while index_link[1] is not None and index_link[1][0] <= index:
            index += 1
            index_link = index_link[1]
        index_link[1] = [index, index_link[1]]
        items = []
        for factor in factors:
            items.append(factor[index % len(factor)])
            index //= len(factor)
        yield items


@dataclass
class Perturbation:
    num_perturbations: int = 1
    tokenizer: AllenTokenizer = AllenTokenizer()

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        raise NotImplementedError()

    def __call__(self, sample: Sample) -> Iterator[str]:
        sentence = sample.instr
        # We need to add space between each word to avoid a mismatch
        tokens = self.tokenizer.tokenize(sentence.lower().rstrip())
        fake_sentence = " ".join([str(token) for token in tokens])
        segments = self.segment(fake_sentence)

        corruptable = [i for i, tok in enumerate(segments) if tok.mode != "NONE"]
        random.shuffle(corruptable)
        candidates = combinations(corruptable, self.num_perturbations)
        cand_tokens = [list(range(len(segment.cand))) for segment in segments]

        iterators = {}
        for candidate in candidates:
            tokens = [cand_tokens[i] for i in candidate]
            iterators[candidate] = random_order_cartesian_product(*tokens)

        while True:
            if not iterators:
                return
            candidate, it = random.choice(list(iterators.items()))
            try:
                indexes = next(it)
            except StopIteration:
                del iterators[candidate]
                continue

            words = []
            j = 0
            for i, segment in enumerate(segments):
                if i in candidate:
                    words.append(segment.cand[indexes[j]])
                    j += 1
                else:
                    words.append(segment.text)
            yield "".join(words)


@dataclass
class MaskPerturbation(Perturbation):
    """
    Candidates are replaced by [MASK] for being replaced later by BERT
    """

    perturbator: Perturbation = field(default_factory=lambda: Perturbation())

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        segments = self.perturbator.segment(sentence)
        for segment in segments:
            if len(segment.cand) == 0:
                continue
            mask = re.sub(r"^(\W)*.*\w+(\W)*$", r"\1[MASK]\2", segment.cand[0])
            segment.cand = [mask]
        return segments


def lm_replacements(masked_sentence: str, predictor: Predictor) -> Iterator[str]:
    assert "[MASK]" in masked_sentence
    if not isinstance(
        predictor,
        allennlp_models.lm.predictors.masked_language_model.MaskedLanguageModelPredictor,
    ):
        raise NotImplementedError()
    convert_tokens_to_string = (
        predictor._dataset_reader._tokenizer.tokenizer.convert_tokens_to_string
    )
    predictions = predictor.predict(masked_sentence)
    tokens = predictions["tokens"]
    # import ipdb

    # ipdb.set_trace()
    for words in product(*predictions["words"]):
        cand_tokens = copy.deepcopy(tokens)
        for i, word in enumerate(words):
            cand_tokens[cand_tokens.index("[MASK]", i)] = word
        yield convert_tokens_to_string(cand_tokens[1:-1])


@dataclass
class BertPerturbation(Perturbation):
    """
    Candidates are replaced by the best BERT prediction for being replaced later by BERT
    """

    perturbator: Perturbation = field(default_factory=lambda: Perturbation())
    predictor: Predictor = field(
        default_factory=lambda: load_predictor("lm-masked-language-model")
    )

    def __call__(self, sample: Sample) -> Iterator[str]:
        for masked_sentence in self.perturbator(sample):
            for fixed_sentence in lm_replacements(masked_sentence, self.predictor):
                yield fixed_sentence


class Graphs:
    def __init__(self):
        self._graphs: Dict[str, nx.Graph] = {}

    def __getitem__(self, scan: str):
        if scan not in self._graphs:
            self._graphs[scan] = load_nav_graphs([scan])[scan]
        return self._graphs[scan]


def is_punctuation(s: str):
    return (
        s != ""
        and s.translate(str.maketrans("", "", string.punctuation + string.whitespace))
        == ""
    )


def load_vilbert() -> nn.Module:
    config = BertConfig.from_json_file("data/config/bert_base_6_layer_6_connect.json")
    model = VLNBert.from_pretrained("data/model_zoos/vlnbert.123mine.bin", config)
    model = model.cuda()
    model.eval()
    return model


def synonyms(name: str):
    return set([l.name() for s in wn.synsets(name.strip()) for l in s.lemmas()])


def hypernyms(name: str):
    return set(
        [
            l.name()
            for s in wn.synsets(name.strip())
            for k in s.hypernyms()
            for l in k.lemmas()
        ]
    )


def same_meaning(a: str, b: str):
    return b in synonyms(a) or a in synonyms(b) or b in hypernyms(a) or a in hypernyms(b)


@dataclass
class VilbertPerturbation(Perturbation):
    """
    Candidates are replaced by the best VilBERT prediction for being replaced later by BERT

    FIXME this perturbator should not be combined directly with any other perturbator
    """

    perturbator: Perturbation = field(default_factory=lambda: Perturbation())
    model: nn.Module = field(default_factory=load_vilbert)
    tokenizer: AutoTokenizer = field(
        default_factory=lambda: AutoTokenizer.from_pretrained("bert-base-uncased")
    )
    features: PanoFeaturesReader = field(
        default_factory=lambda: PanoFeaturesReader(
            "data/matterport-ResNet-101-faster-rcnn-genome.lmdb"
        )
    )
    highlighter: POSHighlighter = field(default_factory=POSHighlighter)
    graphs: Graphs = field(default_factory=Graphs)

    def get_model_input(self, sample: Sample) -> Tuple[Optional[torch.Tensor], ...]:
        tokens = self.tokenizer.tokenize(sample.instr)
        instr_tokens = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)]).cuda()
        segment_ids = torch.zeros_like(instr_tokens)
        instr_masks = instr_tokens > 0

        # get path features
        features, boxes, probs, masks = self.get_path_features(
            sample.scan, sample.path, sample.heading
        )

        # convert data into tensors
        image_features = torch.tensor([features]).float().cuda()
        image_locations = torch.tensor([boxes]).float().cuda()
        image_masks = torch.tensor([masks]).long().cuda()

        co_attention_mask = torch.zeros(2, 8 * 101, 60).long()

        return (
            instr_tokens,
            image_features,
            image_locations,
            segment_ids,
            instr_masks,
            image_masks,
            co_attention_mask,
            None,
        )

    def get_path_features(self, scan_id: str, path: List[str], first_heading: float):
        """Get features for a given path."""
        headings = get_headings(self.graphs[scan_id], path, first_heading)
        # for next headings duplicate the last
        next_headings = headings[1:] + [headings[-1]]

        path_length = min(len(path), 8)
        path_features, path_boxes, path_probs, path_masks = [], [], [], []
        for path_idx, path_id in enumerate(path[:path_length]):
            key = scan_id + "-" + path_id

            # get image features
            features, boxes, probs = self.features[
                key.encode(),
                headings[path_idx],
                next_headings[path_idx],
            ]
            num_boxes = min(len(boxes), 101)

            # pad features and boxes (if needed)
            pad_features = np.zeros((101, 2048))
            pad_features[:num_boxes] = features[:num_boxes]

            pad_boxes = np.zeros((101, 12))
            pad_boxes[:num_boxes, :11] = boxes[:num_boxes, :11]
            pad_boxes[:, 11] = np.ones(101) * path_idx

            pad_probs = np.zeros((101, 1601))
            pad_probs[:num_boxes] = probs[:num_boxes]

            box_pad_length = 101 - num_boxes
            pad_masks = [1] * num_boxes + [0] * box_pad_length

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        # pad path lists (if needed)
        for path_idx in range(path_length, 8):
            pad_features = np.zeros((101, 2048))
            pad_boxes = np.zeros((101, 12))
            pad_boxes[:, 11] = np.ones(101) * path_idx
            pad_probs = np.zeros((101, 1601))
            pad_masks = [0] * 101

            path_features.append(pad_features)
            path_boxes.append(pad_boxes)
            path_probs.append(pad_probs)
            path_masks.append(pad_masks)

        return (
            np.vstack(path_features),
            np.vstack(path_boxes),
            np.vstack(path_probs),
            np.hstack(path_masks),
        )

    @torch.no_grad()
    def vlm_replacement(self, orig_word: str, sample: Sample) -> str:
        mask_token = self.tokenizer.convert_tokens_to_ids("[MASK]")
        inputs = self.get_model_input(sample)
        output = self.model(*inputs)

        instr_token = inputs[0]
        if instr_token is None:
            raise RuntimeError()
        instr_token = instr_token[0]
        word_idx = (instr_token.cpu() == mask_token).int().argmax()

        linguistic_predictions = output[2].view(-1, output[2].shape[-1])
        values, indices = torch.sort(linguistic_predictions[word_idx], descending=True)

        for index in indices.tolist():
            token = self.tokenizer.convert_ids_to_tokens(index)
            if (
                not token in stopwords.words()
                and not is_punctuation(token)
                and same_meaning(token, orig_word.strip())
            ):
                instr_token[word_idx] = index
                break

        tokens = self.tokenizer.convert_ids_to_tokens(instr_token)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def mask(self, sentence: str) -> Iterator[Tuple[str, str]]:
        words_of_interest = self.highlighter(sentence)
        cursor = 0
        for word in words_of_interest:
            position = sentence[cursor:].find(word)
            if position == -1:
                continue
            masked_sentence = sentence[:cursor] + sentence[cursor:].replace(
                word, "[MASK]", 1
            )
            cursor += position
            yield (word, masked_sentence)

    def __call__(self, sample: Sample) -> Iterator[str]:
        for orig_word, masked_sentence in self.mask(sample.instr):
            masked_sample = copy.deepcopy(sample)
            masked_sample.instr = masked_sentence
            fixed_sentence = self.vlm_replacement(orig_word, masked_sample)
            yield fixed_sentence


@dataclass
class RemainPerturbation(Perturbation):
    perturbator: Perturbation = field(default_factory=lambda: Perturbation())
    num_clues: int = 1

    def __call__(self, sample: Sample) -> Iterator[str]:
        sentence = sample.instr
        # We need to add space between each word to avoid a mismatch
        tokens = self.tokenizer.tokenize(sentence.lower().rstrip())
        fake_sentence = " ".join([str(token) for token in tokens])
        segments = self.perturbator.segment(fake_sentence)

        corruptable = [i for i, tok in enumerate(segments) if tok.mode != "NONE"]
        random.shuffle(corruptable)
        cand_size = len(corruptable) - self.num_clues

        if cand_size <= 0:
            return

        candidates = combinations(corruptable, cand_size)

        for candidate in candidates:
            yield "".join(
                [
                    random.choice(s.cand) if i in candidate else s.text
                    for i, s in enumerate(segments)
                ]
            )


@dataclass
class CombinePerturbation(Perturbation):
    perturbators: List[Perturbation] = field(default_factory=list)

    @staticmethod
    def add_segments(
        proposals: List[TokenPerturbation], segments: List[TokenPerturbation]
    ) -> List[TokenPerturbation]:
        # Assuming that the segments are sorted
        merged: List[TokenPerturbation] = []

        for segment in segments:
            if segment.mode != "NONE":
                merged.append(segment)
                continue
            start, end = segment.span
            subsegment = False
            for proposal in proposals:
                if proposal.mode == "NONE":
                    continue
                if proposal.span[0] < start or proposal.span[1] > end:
                    continue
                # a proposal is inside a NONE segment --> we divide the NONE segment

                # keep the part of the segment before
                if proposal.span[0] > start:
                    length = proposal.span[0] - start
                    merged.append(
                        TokenPerturbation(
                            segment.text[start : start + length],
                            (start, proposal.span[0] - 1),
                            "NONE",
                        )
                    )
                    start = proposal.span[1] + 1

                subsegment = True
                # add the proposal
                merged.append(proposal)

            # add the end of the current segment
            if subsegment and merged[-1].span[1] < end:
                merged.append(
                    TokenPerturbation(
                        segment.text[merged[-1].span[1] + 1 :],
                        (merged[-1].span[1] + 1, end),
                        "NONE",
                    )
                )

            if not subsegment:
                merged.append(segment)

        return merged

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        segments: List[TokenPerturbation] = []
        for perturbator in self.perturbators:
            proposals = perturbator.segment(sentence)
            if segments == []:
                segments = proposals
            else:
                segments = CombinePerturbation.add_segments(proposals, segments)
        return segments


@dataclass
class DeafPerturbation(Perturbation):
    def __call__(self, sample: Sample) -> Iterator[str]:
        yield ""


@dataclass
class StopWordPerturbation(Perturbation):
    tokenizer: SpacyTokenizer = SpacyTokenizer(nlp.vocab)

    @staticmethod
    def extended_is_stop(token: Token) -> bool:
        stop_words = nlp.Defaults.stop_words
        return token.is_stop or token.lower_ in stop_words or token.lemma_ in stop_words

    def __call__(self, sample: Sample) -> Iterator[str]:
        sentence = sample.instr
        doc = self.tokenizer(sentence)
        yield "".join(
            [
                token.text_with_ws
                for token in doc
                if not StopWordPerturbation.extended_is_stop(token)
            ]
        )


@dataclass
class DirectionPerturbation(Perturbation):
    predictor: Predictor = field(
        default_factory=lambda: Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
        )
    )
    keywords: Tuple[List[str], ...] = (
        ["left"],
        ["right"],
        ["upstairs", "up"],
        ["downstairs", "down"],
        ["forward", "straight"],
        ["inside"],
        ["outside"],
        ["around"],
    )
    action_verbs: Tuple[str, ...] = (
        "turn",
        "veer",
        "walk",
        "go",
        "exit",
        "move",
        "continue",
        "head",
        "stop",
        "enter",
    )

    def __post_init__(self) -> None:
        self._pattern = "|".join(
            [f"(?<!\w){word}(?!\w)" for cat in self.keywords for word in cat]
        )

    @staticmethod
    def _search_real_span(tree: Dict) -> Tuple[int, int]:
        start = tree["spans"][0]["start"]
        end = tree["spans"][0]["end"]
        if "children" in tree:
            for leaf in tree["children"]:
                cstart, cend = DirectionPerturbation._search_real_span(leaf)
                start = min(start, cstart)
                end = max(end, cend)
        return start, end

    @staticmethod
    def _span_pos_to_span_tok(tokens: List[str], start_pos: int, end_pos: int):
        """
        >>> sentence = 'Take a left and go down the stairs.'
        >>> result = re.search(r"go down", sentence)
        >>> span_pos_to_span_tok(sentence.split(" "), *result.span())
        (4, 5)
        """
        counter = 0
        start = -1
        end = -1
        for i, token in enumerate(tokens):
            if start < 0 and counter >= start_pos:
                start = i
            counter += len(str(token)) + 1
            if end < 0 and counter > end_pos - 1:
                end = i
        return start, end

    @staticmethod
    def _get_parent_attr(tree: Dict, start: int, end: int, dep_attr: str = "ROOT"):
        tspan = DirectionPerturbation._search_real_span(tree)
        #     print(tree["spans"][0], tree["word"], tspan)

        if tree["spans"][0]["start"] == start and tree["spans"][0]["end"] == end + 1:
            return dep_attr

        if tspan[0] == start and tspan[1] == end + 1:
            return dep_attr

        if "children" in tree:
            for child in tree["children"]:
                cand_dep_attr = DirectionPerturbation._get_parent_attr(
                    child, start, end, dep_attr=tree["attributes"][0]
                )
                if cand_dep_attr is not None:
                    return cand_dep_attr

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        """
        We split sentence into sub sentence, as it increases the performances of the dependency parser.
        """
        segments = []
        splits = sentence.split(". ")
        offset = 0
        for i, phrase in enumerate(splits):
            new_segments = self._segment_phrase(phrase)
            for segment in new_segments:
                start, end = segment.span
                start += offset
                end += offset
                segment.span = (start, end)
            segments += new_segments
            if i != len(splits) - 1:
                segments[-1].text += ". "
                start, end = segments[-1].span
                end += 2
                segments[-1].span = (start, end)
            offset = segments[-1].span[1] + 1
        return segments

    def _segment_phrase(self, sentence: str) -> List[TokenPerturbation]:
        segments = self._get_detection_tokens(sentence)
        return fill_with_none(sentence, segments)

    def _get_detection_tokens(self, sentence: str) -> List[TokenPerturbation]:
        detection_tokens = []

        tokens = self.tokenizer.tokenize(sentence)
        pred = self.predictor.predict(sentence=sentence)  # type: ignore
        tree = pred["hierplane_tree"]["root"]

        for result in re.finditer(self._pattern, sentence):
            found = result.group()
            start, end = result.span()
            # start += len(found) - len(re.sub("^\W", "", found))
            # end -= len(found) - len(re.sub("\W$", "", found))

            word_idx, _ = DirectionPerturbation._span_pos_to_span_tok(tokens, start, end)
            previous_word = str(tokens[word_idx - 1]).lower() if word_idx > 0 else None
            attr = DirectionPerturbation._get_parent_attr(tree, start, end)
            #             print("       ", previous_word, found, start, end, attr)
            if (
                attr != "VERB"
                #                 and attr != "ROOT"
                and previous_word not in self.action_verbs
            ):
                # print("       ", previous_word, found, start, end, attr)
                continue

            for cat_index, cat in enumerate(self.keywords):
                if found in cat:
                    break
            if found not in self.keywords[cat_index]:
                raise RuntimeError(f"Can't find {found} among {self.keywords}")

            replacements: List[str] = []
            for i, words in enumerate(self.keywords):
                if i != cat_index:
                    replacements += words
            detection_tokens.append(
                TokenPerturbation(found, (start, end - 1), "DIRECTION", replacements)
            )

        return detection_tokens


@dataclass
class LocationPerturbation(Perturbation):
    # next to is not in the list, because it has several meaning (left? right?)
    patterns_adv = r"(?<!\w)(?:on|in|at|to|into|onto)\s?(?:your|the|th)?\s(?:left|right|bottom|top|front|middle)(?: of| side of)?(?!\w)|behind|underneath|\Wover\W|above"
    keywords_adv: Optional[Dict] = None
    patterns_adj = r"(?:left|right)(?:most|side)"
    keywords_adj: Optional[Dict] = None

    @staticmethod
    def _gen_keywords(direction):
        return [
            f"{prep}{conj} {direction}{filler}"
            for prep, conj, filler in product(
                ["into", "onto", "on", "in", "at", "to"],
                [" your", "", " the"],
                ["", " side of", " of"],
            )
        ]

    def __post_init__(self):
        if self.keywords_adv is None:
            self.keywords_adv = defaultdict(list)
            self.keywords_adv["top"] = ["over", "above"]
            self.keywords_adv["bottom"] = ["underneath", "below"]
            self.keywords_adv["behind"] = ["behind"]
            for direction in ("left", "right", "bottom", "top", "front", "middle"):
                self.keywords_adv[direction] += self._gen_keywords(direction)

        if self.keywords_adj is None:
            self.keywords_adj = {
                direction: [f"{direction}{adv}" for adv in ["most", "side"]]
                for direction in ["left", "right"]
            }

    @staticmethod
    def _pick_replacement(found: str, keywords: Dict[str, str]) -> List[str]:
        strip = re.sub("^\W|\W$", "", found)
        strip = re.sub(" +", " ", strip)

        for cat, words in keywords.items():
            if strip in words:
                break
        if strip not in words:
            raise RuntimeError(f"Can't find '{strip}' among '{keywords}'")

        replacements: List[str] = []
        for name, words in keywords.items():
            if name != cat:
                replacements += words
        return replacements

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        if self.keywords_adv is None or self.keywords_adj is None:
            raise RuntimeError("Not correctly initialized")

        location_tokens = []

        for result in re.finditer(self.patterns_adv, sentence.lower()):
            found = result.group()
            start, end = result.span()
            replace = self._pick_replacement(found, self.keywords_adv)
            location_tokens.append(
                TokenPerturbation(found, (start, end - 1), "LOCATION", replace)
            )

        for result in re.finditer(self.patterns_adj, sentence.lower()):
            found = result.group()
            start, end = result.span()
            replace = self._pick_replacement(found, self.keywords_adj)
            location_tokens.append(
                TokenPerturbation(found, (start, end - 1), "LOCATION", replace)
            )

        return fill_with_none(sentence, location_tokens)


@dataclass
class NounPhrasePerturbation(Perturbation):
    num_perturbations: int = 1
    predictor: Predictor = field(
        default_factory=lambda: Predictor.from_path(
            "/gpfsdswork/projects/rech/vuo/uok79zh/.allennlp/elmo/"
            # "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )
    )
    # turn is causing a lot of confusion to the parser
    forbidden_words: Tuple = ("turn",)
    min_len: int = 2
    max_len: int = 6
    mode: str = "NP"

    def _retrieve_noun_phrases(
        self, sentence: str, tree: Dict, pos: int = 0
    ) -> List[TokenPerturbation]:
        """
        Return a dictionary with noun phrases and the spanning positions
        max_len is a protection against parser failures
        """
        noun_phrases: List[TokenPerturbation] = []
        next_char = len(tree["word"]) + pos
        if next_char < len(sentence) and sentence[next_char].isspace():
            tree["word"] += " "

        # print(tree["word"], pos)
        # offset the position as we decode this tree
        inner_pos = 0

        for children in tree["children"]:
            next_char = len(children["word"]) + pos + inner_pos
            if next_char < len(sentence) and sentence[next_char].isspace():
                children["word"] += " "
            # print(
            #     "---",
            #     children["word"],
            #     f"{pos+inner_pos} ({inner_pos}+{pos}) => {pos+inner_pos+len(children['word']) - 1}",
            #     sentence[pos + inner_pos : pos + inner_pos + len(children["word"])],
            # )

            if children["nodeType"] == "NP":
                proposal = children["word"]
                num_tokens = len(self.tokenizer.tokenize(proposal))

                if (
                    "." not in proposal
                    and self.min_len <= num_tokens
                    and num_tokens <= self.max_len
                    and all(word not in proposal for word in self.forbidden_words)
                ):
                    start = tree["word"][inner_pos:].find(proposal) + pos + inner_pos
                    end = start + len(proposal) - 1
                    noun_phrases.append(
                        TokenPerturbation(proposal, (start, end), self.mode)
                    )
                    inner_pos += len(children["word"])
                    continue

            if "children" in children:
                start = tree["word"][inner_pos:].find(children["word"]) + pos + inner_pos
                noun_phrases += self._retrieve_noun_phrases(sentence, children, pos=start)

            inner_pos += len(children["word"])
        return noun_phrases

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        preds = self.predictor.predict(sentence=sentence)  # type: ignore
        noun_phrases = self._retrieve_noun_phrases(
            sentence, preds["hierplane_tree"]["root"]
        )

        # sort the noun phrases by start span
        noun_phrases = sorted(noun_phrases, key=lambda item: item.span[0])

        return fill_with_none(sentence, noun_phrases)


@dataclass
class SwitchPerturbation(NounPhrasePerturbation):
    num_perturbations: int = 2

    def __post_init__(self):
        assert self.num_perturbations == 2, self.num_perturbations

    def __call__(self, sample: Sample) -> Iterator[str]:
        sentence = sample.instr
        tokens = self.tokenizer.tokenize(sentence.lower().rstrip())
        fake_sentence = " ".join([str(token) for token in tokens])
        segments = self.segment(fake_sentence)
        corruptable = [i for i, tok in enumerate(segments) if tok.mode == self.mode]
        random.shuffle(corruptable)
        candidates = combinations(corruptable, self.num_perturbations)

        for (a, b) in candidates:
            a_text = segments[a].text
            b_text = segments[b].text

            a_start = a_text[0] if a_text[0].isspace() else ""
            a_end = a_text[-1] if a_text[-1].isspace() else ""
            b_start = b_text[0] if b_text[0].isspace() else ""
            b_end = b_text[-1] if b_text[-1].isspace() else ""

            if b_text.strip() == a_text.strip():
                continue

            segments[a].text = a_start + b_text.strip() + a_end
            segments[b].text = b_start + a_text.strip() + b_end
            yield "".join([t.text for t in segments])

            # rollback
            segments[b].text = b_text
            segments[a].text = a_text


@dataclass
class NounPhraseFromVocPerturbation(NounPhrasePerturbation):
    vocabulary: List = field(default_factory=list)
    num_cand: int = 10

    def __post_init__(self):
        self.num_cand = min(10, len(self.vocabulary))

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        segments = super().segment(sentence)
        for segment in segments:
            if segment.mode == self.mode:
                words = random.sample(self.vocabulary, self.num_cand)
                start_ws = segment.text[0] if segment.text[0].isspace() else ""
                end_ws = segment.text[-1] if segment.text[-1].isspace() else ""
                segment.cand = [start_ws + cand.strip() + end_ws for cand in words]
        return segments


@dataclass
class NounPhraseReplacedPerturbation(NounPhrasePerturbation):
    replacements: Dict = field(default_factory=dict)

    def __post_init__(self):
        self._categories_to_words = defaultdict(list)
        for name, cat in self.replacements.items():
            self._categories_to_words[cat].append(name)

    def segment(self, sentence: str) -> List[TokenPerturbation]:
        segments = super().segment(sentence)
        for segment in segments:
            if segment.mode != self.mode:
                continue

            for w in self.replacements.keys():
                if w not in segment.text:
                    continue
                segment.cand = []
                for k, words in self._categories_to_words.items():
                    if k != self.replacements[w]:
                        segment.cand += [segment.text.replace(w, cand) for cand in words]
                break
        return segments


def perturbations_training(
    dataset: List[Dict],
    perturbator: Perturbation,
    num_samples: int,
    only_pert: bool,
    max_variants: int = 10,
) -> List[Dict]:
    counter = 0
    if num_samples == -1:
        num_samples = len(dataset)

    perturbed = []
    removed_instr = 0
    removed_path = 0

    for item in tqdm(dataset, total=num_samples):
        item["perturbations"] = []
        for instr in item["instructions"]:
            it = perturbator(Sample(instr, item["path"], item["scan"], item["heading"]))
            item["perturbations"].append([stc for _, stc in zip(range(max_variants), it)])

        if only_pert:
            indices = list(range(len(item["instructions"])))[::-1]
            for i in indices:
                if item["perturbations"][i] == []:
                    del item["instructions"][i]
                    del item["perturbations"][i]
                    removed_instr += 1
            if item["instructions"] == []:
                removed_path += 1
                continue

        perturbed.append(item)

        counter += 1
        if num_samples > -1 and counter >= num_samples:
            return perturbed

    print("Removed instr", removed_instr)
    print("Removed path", removed_path)
    print("Kept instr", sum(len(i["instructions"]) for i in perturbed))
    print("Kept path", len(perturbed))

    return perturbed


def perturbations_testing(
    dataset: List[Dict], perturbator: Perturbation, num_samples: int, only_pert: bool
) -> List[Dict]:
    if num_samples != -1:
        print("Ignoring the parameter num_samples")
    if only_pert:
        raise NotImplementedError()
    perturbed = dataset
    for item in tqdm(dataset):
        item["corrupted"] = [False] * len(item["instructions"])
        for i, instr in enumerate(item["instructions"]):
            it = perturbator(Sample(instr, item["path"], item["scan"], item["heading"]))
            try:
                item["instructions"][i] = next(it)
                item["corrupted"][i] = True
            except StopIteration:
                pass

    return perturbed


def get_perturbator(
    mode: str,
    num_perturbations: int = 1,
    num_clues: int = 1,
    mask: bool = False,
    bert: bool = False,
) -> Perturbation:
    perturbators: List[Perturbation] = []

    if "direction" in mode:
        perturbators.append(DirectionPerturbation(num_perturbations=num_perturbations))

    elif "location" in mode:
        perturbators.append(LocationPerturbation(num_perturbations=num_perturbations))

    elif "deaf" in mode:
        perturbators.append(DeafPerturbation())

    elif "stop" in mode:
        perturbators.append(StopWordPerturbation())

    elif "object" in mode:
        forbidden: Tuple = (
            "turn",
            "room",
            "kitchen",
            "cellar",
            "hall",
            "office",
            "garage",
            "cabinet",
        )
        if "switch" in mode:
            perturbators.append(SwitchPerturbation(forbidden_words=forbidden))
        else:
            with open("data/task/noun_phrases.txt") as fid:
                voc = [w.strip() for w in fid.readlines()]
            perturbators.append(
                NounPhraseFromVocPerturbation(
                    forbidden_words=forbidden,
                    vocabulary=voc,
                    num_perturbations=num_perturbations,
                )
            )

    elif "room" in mode:
        forbidden = ("turn",)
        with open("data/task/rooms.txt") as fid:
            replacements = {}
            for w in fid.readlines():
                name, replace = w.strip().split(", ")
                replacements[name] = replace
        perturbators.append(
            NounPhraseReplacedPerturbation(
                forbidden_words=forbidden,
                replacements=replacements,
                num_perturbations=num_perturbations,
            )
        )

    elif "swap" in mode:
        perturbators.append(SwitchPerturbation(forbidden_words=("turn",)))

    elif mode == "vilbert":
        perturbators.append(VilbertPerturbation())

    if len(perturbators) == 0:
        raise RuntimeError()

    if mask or bert:
        assert num_perturbations == 1, num_perturbations
        perturbators = [MaskPerturbation(perturbator=p) for p in perturbators]
    if bert:
        assert num_perturbations == 1, num_perturbations
        perturbators = [BertPerturbation(perturbator=p) for p in perturbators]

    if len(perturbators) == 1:
        perturbator: Perturbation = perturbators[0]
    else:
        perturbator = CombinePerturbation(
            perturbators=perturbators, num_perturbations=num_perturbations
        )

    if "remain" in mode:
        perturbator = RemainPerturbation(perturbator=perturbator, num_clues=num_clues)

    return perturbator


class Arguments(argtyped.Arguments):
    infile: Path
    outfile: Path
    mode: str
    num_samples: int = -1
    num_perturbations: int = 1
    num_clues: int = 1
    training: bool = False
    mask: bool = False
    only_pert: bool = False
    bert: bool = False


if __name__ == "__main__":
    args = Arguments()

    print("Loading the dataset")
    dataset = read_dataset(args.infile)
    mode = str(args.mode)

    print("Loading the perturbator")
    perturbator = get_perturbator(
        args.mode, args.num_perturbations, args.num_clues, args.mask, args.bert
    )

    print("Perturbating")
    fn = perturbations_training if args.training else perturbations_testing
    dataset = fn(dataset, perturbator, args.num_samples, args.only_pert)  # type: ignore

    print("Exporting to", args.outfile)
    save_dataset(dataset, args.outfile)
