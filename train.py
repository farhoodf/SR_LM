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

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)
torch.use_deterministic_algorithms(True)

MODEL_NAME = "facebook/bart-large"


chencherry = bs.SmoothingFunction()

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--save_path", type=str, help="the path to save models")

my_parser.add_argument("--bs", type=int, default=8, help="batch_size")

my_parser.add_argument("--lr", type=float, default=0.00001, help="learning rate")
my_parser.add_argument("--ds", type=str, default="ewt")
my_parser.add_argument("--task", type=str, default="shallow")
my_parser.add_argument("--maxsamples", type=int, default=0)
my_parser.add_argument("--maxlen", type=int, default=35)
my_parser.add_argument("--maxepoch", type=int, default=15)

args = my_parser.parse_args()

print(vars(args))
save_path = os.path.join(args.save_path, f"{args.task}-{args.ds}-{args.maxsamples}")

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"path: {save_path} created.", flush=True)
else:
    print(f"path: {save_path} exists and files will be overwritten.", flush=True)

# create data loaders
tokenizer = transformers.BartTokenizer.from_pretrained(MODEL_NAME)


def collate(batch):
    inputs, labels = list(zip(*batch))
    inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer.batch_encode_plus(labels, return_tensors="pt", padding=True, truncation=True)
    return inputs, labels


max_samples = None if args.maxsamples == 0 else args.maxsamples

train_set = SRDataset(data_paths[args.task]["train"][args.ds], max_len=args.maxlen, max_samples=max_samples)
val_set = SRDataset(data_paths[args.task]["test"][args.ds], max_len=1000)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.bs, shuffle=True, drop_last=True, collate_fn=collate
)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate)

labels = get_labels(data_paths["labels"][args.ds])

print(len(train_set), len(val_set), len(labels))

# Create model and optim
device = torch.device("cuda")
model = transformers.BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters,
    lr=args.lr,
    eps=1e-6,
)


def train(model, loader):
    model = model.train()
    losses = []
    for i, batch in enumerate(loader):
        model.zero_grad()
        o = model(
            input_ids=batch[0].input_ids.to(device),
            attention_mask=batch[0].attention_mask.to(device),
            labels=batch[1].input_ids.to(device),
            decoder_attention_mask=batch[1].attention_mask.to(device),
        )
        o.loss.backward()
        optimizer.step()
        losses.append(o.loss.item())
    return model, sum(losses) / len(losses)


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


def write_gens(gens, path):
    with open(path, "w") as f:
        for row in gens:
            f.write(" ".join(row))
            f.write("\n")


for epoch in range(args.maxepoch):
    start_time = time.time()
    model, train_loss = train(model, train_loader)
    gens = gen(model, val_loader)
    bleu, nist, dist = get_metrics(labels, gens)
    running_time = divmod((time.time() - start_time), 60)[0]

    print(f"epoch: {epoch+1}, train loss: {train_loss}, epoch time: {running_time}")
    print(f"bleu: {bleu}, nist: {nist}, dist: {dist}")
    print("", flush=True)
    # torch.save(model.state_dict(), f"./{save_path}/epoch_{epoch}.pt")
    write_gens(gens, f"./{save_path}/epoch_{epoch}.txt")
