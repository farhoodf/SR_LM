import argparse
import os
import time

import torch
import transformers
from nltk.metrics import *
import nltk.translate.nist_score as ns
import nltk.translate.bleu_score as bs

from data_utils import SRDataset, get_labels, nlp
from data_paths import data_paths

# torch.manual_seed(0)
# import random

# random.seed(0)
# import numpy as np

# np.random.seed(0)
# torch.use_deterministic_algorithms(True)

MODEL_NAME = "facebook/bart-large"


chencherry = bs.SmoothingFunction()

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--load_path", type=str, help="the path to save models")
my_parser.add_argument("--task", type=str, default="shallow")

args = my_parser.parse_args()

print(vars(args))
save_path = args.load_path

tokenizer = transformers.BartTokenizer.from_pretrained(MODEL_NAME)


def collate(batch):
    inputs, labels = list(zip(*batch))
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer.batch_encode_plus(labels, return_tensors="pt", padding=True, truncation=True)
    return inputs, labels


max_samples = None

val_set = SRDataset(data_paths[args.task]["test"]["ewt"], max_len=100)


val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate)

labels = get_labels(data_paths["labels"]["ewt"])


# Create model and optim
device = torch.device("cuda")
model = transformers.BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.load_state_dict(torch.load(args.load_path))


def gen(model, loader):
    model = model.eval()
    gen_ids = []
    gr = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            o = model.generate(
                input_ids=batch[0].input_ids.to(device),
                attention_mask=batch[0].attention_mask.to(device),
                max_length=100,
            )
            for sample_id in range(o.shape[0]):
                gen_ids.append(o[sample_id].cpu())
    gen = [
        [w.text.lower() for w in nlp(x).iter_words()] for x in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    ]
    return gen


def get_metrics(ref, hyp):
    # NIST score
    nist = ns.corpus_nist(ref, hyp, n=4)
    # BLEU score
    chencherry = bs.SmoothingFunction()
    bleu = bs.corpus_bleu(ref, hyp, smoothing_function=chencherry.method2)
    # DIST
    total_str_len = 0.0
    edits, total_word_edits = 0.0, 0.0
    micro_edits, macro_edits = 0.0, 0.0
    cnt = 0.0
    word_edits = 0.0

    for r, h in zip(ref, hyp):

        cnt += 1

        # String edit distance.
        s1 = " ".join(r[0])
        s2 = " ".join(h)
        total_str_len += max(len(s1), len(s2))
        macro_edits += edit_distance(s2, s1)

        word_edits += edit_distance(r[0], h)
    dist = float(1 - macro_edits / total_str_len)
    return bleu, nist, dist


def write_gens(gens, path):
    with open(path, "w") as f:
        for row in gens:
            f.write(" ".join(row))
            f.write("\n")


gens = gen(model, val_loader)
bleu, nist, dist = get_metrics(labels, gens)
print(f"bleu: {bleu}, nist: {nist}, dist: {dist}", flush=True)
# write_gens(gens, f"{save_path}.txt")
