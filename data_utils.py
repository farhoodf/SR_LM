import torch
import stanza

nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=False, verbose=False)

id_to_tag = ["id", "lemma", "form", "upos", "xpos", "feats", "head", "deprel"]
tag_to_id = {v: k for k, v in enumerate(id_to_tag)}


def parse_feats(feats):
    feats = feats.split("|")
    parsed_feats = {}
    for feat in feats:
        feat = feat.split("=")
        if len(feat) > 1:
            if not (feat[0] == "original_id" or feat[0].startswith("id")):
                parsed_feats[feat[0]] = feat[1]

    return parsed_feats


def parse_sample(sample, max_len):
    sample = sample.splitlines()
    if len(sample) > max_len:
        return None
    start = 0
    text = list()
    while sample[start].startswith("#"):
        if sample[start].startswith("# text"):
            text = sample[start].split(" = ")[1].split()
        start += 1
    parsed = []
    head = -1
    for row in sample[start:]:
        row = row.split("\t")
        sampled_parsed = {}
        for key, value in enumerate(id_to_tag):
            sampled_parsed[value] = row[key]
        sampled_parsed["id"] = int(sampled_parsed["id"])
        sampled_parsed["head"] = int(sampled_parsed["head"])
        if sampled_parsed["head"] == 0:
            head = sampled_parsed["id"]
        sampled_parsed["feats"] = parse_feats(sampled_parsed["feats"])
        parsed.append(sampled_parsed)
    # text = generate_text(parsed) if get_text else list()
    return parsed, head, text


def read_data(path, max_len=35, max_samples=None):
    with open(path, "r") as f:
        data = f.read()
    data = data.split("\n\n")
    if len(data[-1]) == 0:
        data = data[:-1]
    if max_samples is not None:
        data = data[:max_samples]
    outputs = []
    outputs_ids = []
    for i, sample in enumerate(data):
        output = parse_sample(sample, max_len)
        if output is not None:
            outputs.append(output)
            outputs_ids.append(i)
    return outputs, outputs_ids


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_len, max_samples=None):
        self.data, self.outputs_ids = read_data(path, max_len, max_samples)

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, i):
        return self.get_input(self.data[i][0]), " ".join(self.data[i][2])

    def get_input(self, sample):
        inputs = []
        num_tokens = len(sample)
        indices = torch.randperm(len(sample))
        for i in indices:
            token = sample[i]
            feats = " ".join([y for x, y in token["feats"].items()])
            head = token["head"] if token["head"] != 0 else "ROOT"
            token_text = f"{token['id']}:{token['lemma']} {feats}:{head} <{token['deprel']}>"
            inputs.append(token_text)

        return " # ".join(inputs)


def get_masked(text):
    text = text.split()
    indices = torch.randperm(len(text))
    new_text = []
    for k, i in enumerate(indices):
        new_text.append(f"{i}:{text[i]} :0 <uk>")
    return " # ".join(new_text)


class AEDS(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(
        self,
    ):
        return len(self.ds)

    def __getitem__(self, idx):
        return get_masked(self.ds[idx][1]), self.ds[idx][1]


def get_labels(path):
    with open(path, "r") as f:
        test_text = f.read().split("\n\n")[:-1]
    for i, text in enumerate(test_text):
        test_text[i] = [[x.lower() for x in text.split("\n")[-1].split(" = ")[1].split()]]
    return test_text
