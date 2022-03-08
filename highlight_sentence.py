from typing import List, Dict, Callable, Tuple, Any
import argparse
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
import json
from tqdm.auto import tqdm
from allennlp.predictors.predictor import Predictor
import spacy

random.seed(0)
nlp = spacy.load("en")


def load_json(infile: Path):
    with open(infile) as fid:
        return json.load(fid)


def save_json(dataset: Any, outfile: Path):
    with open(outfile, "w") as fid:
        return json.dump(dataset, fid, indent=2)


Highlighter = Callable[[str], List[str]]


@dataclass
class POS:
    name: str
    span: Tuple[int, int]


@dataclass
class POSHighlighter:
    predictor: Predictor = field(
        default_factory=lambda: Predictor.from_path(
            "/gpfsdswork/projects/rech/vuo/uok79zh/.allennlp/elmo/",
            cuda_device=0
            #             "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
        )
    )
    # turn is causing a lot of confusion to the parser
    forbidden: Tuple[str, ...] = ("next", "then", "there", "just")
    pos: Tuple[str, ...] = ("NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS")

    def __post_init__(self):
        self._tokenizer = self.predictor._tokenizer

    def _retrieve_pos(self, sentence: str, tree: Dict, pos: int = 0) -> List[POS]:
        """
        Return a dictionary with noun phrases and the spanning positions
        """
        inner_pos = 0
        part_of_speeches: List[POS] = []

        for children in tree["children"]:
            next_char = len(children["word"]) + pos + inner_pos
            if next_char < len(sentence) and sentence[next_char].isspace():
                children["word"] += " "

            proposal = children["word"].strip()

            if children["nodeType"] in self.pos and proposal not in self.forbidden:
                start = tree["word"][inner_pos:].find(proposal) + pos + inner_pos
                end = start + len(proposal) - 1
                part_of_speeches.append(POS(proposal, (start, end)))
                inner_pos += len(children["word"])
                continue

            if "children" in children:
                start = tree["word"][inner_pos:].find(children["word"]) + pos + inner_pos
                part_of_speeches += self._retrieve_pos(sentence, children, pos=start)

            inner_pos += len(children["word"])
        return part_of_speeches

    def __call__(self, sentence: str) -> List[str]:
        tokens = self._tokenizer.tokenize(sentence.lower().rstrip())
        fake_sentence = " ".join([str(token) for token in tokens])
        preds = self.predictor.predict(sentence=fake_sentence)  # type: ignore
        pos = self._retrieve_pos(sentence, preds["hierplane_tree"]["root"])

        # sort the pos by start span
        pos = sorted(pos, key=lambda item: item.span[0])
        return [p.name for p in pos]


def highlight_dataset(
    dataset: List[Dict],
    highlighter: Highlighter,
    num_samples: int,
) -> None:
    counter = 0
    for item in tqdm(dataset):
        if "highlights" not in item:
            item["highlights"] = [highlighter(instr) for instr in item["instructions"]]
        if "perturbations" in item and "perturbation_highlights" not in item:
            item["perturbation_highlights"] = []
            for instructions in item["perturbations"]:
                item["perturbation_highlights"].append(
                    [highlighter(instr) for instr in instructions]
                )
        counter += 1
        if num_samples > -1 and counter >= num_samples:
            return


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument(
        "--pos",
        type=str,
        nargs="+",
        default=("NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS"),
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_parser()

    dataset = load_json(args.infile)
    highlighter = POSHighlighter(pos=args.pos)
    highlight_dataset(dataset[args.id :: args.num_procs], highlighter, args.num_samples)  # type: ignore

    print("Exporting to", args.outfile)
    lock = args.outfile.parent / f"{args.outfile.stem}.lock"
    while lock.is_file():
        print(f"lock found at {lock}. sleep 1 sec")
        time.sleep(1)
    lock.touch()
    uptodate = load_json(args.infile)
    uptodate[args.id :: args.num_procs] = dataset[args.id :: args.num_procs]
    save_json(uptodate, args.outfile)
    lock.unlink()
