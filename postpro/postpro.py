import json

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import PostProModel

N_CLASSES = 3
N_MEMBERS = 3
LABEL2IDX = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def train(model, train_dataloader, test_dataloader, all_dataloader, n_epochs, device,
          removed_correct=0, removed_count=0):
    train_acc = evaluate(model, train_dataloader, device, removed_correct, removed_count)
    test_acc = evaluate(model, test_dataloader, device, removed_correct, removed_count)
    all_acc = evaluate(model, all_dataloader, device, removed_correct, removed_count)
    print(train_acc, test_acc, all_acc)

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        model.train()
        for probs, labels in train_dataloader:
            probs, labels = probs.to(device), labels.to(device)
            logits = model(probs)

            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        train_acc = evaluate(model, train_dataloader, device, removed_correct, removed_count)
        test_acc = evaluate(model, test_dataloader, device, removed_correct, removed_count)
        all_acc = evaluate(model, all_dataloader, device, removed_correct, removed_count)
        print(train_acc, test_acc, all_acc)

def evaluate(model, dataloader, device, removed_correct=0, removed_count=0):
    model.eval()
    correct = count = 0
    for probs, labels in dataloader:
        probs, labels = probs.to(device), labels.to(device)
        logits = model(probs)

        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        correct += (logits.argmax(dim=-1) == labels).sum()
        count += logits.shape[0]

    return (correct.float() + removed_correct) / (count + removed_count)

def read_tsv(filename, field_idx, has_header, sep='\t'):
    preds = []
    with open(filename) as f:
        seen_header = not has_header
        for line in f:
            if not seen_header:
                seen_header = True
                continue
            pred = line.strip().split(sep)[field_idx]
            preds.append(pred)
    return preds

def read_json(filename):
    with open(filename) as f:
        js = json.load(f)
        return js

def wrap(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def flatten(l):
    return [x for subl in l for x in subl]

def remove_non_conflicts(probs, labels):
    """
    probs: (N_GROUPS, N_MEMBERS, N_CLASSES)
    labels: (N_GROUPS, N_MEMBERS)
    """
    assert probs.shape[:2] == labels.shape[:2]

    new_probs = []
    new_labels = []
    removed_correct = 0
    removed_count = 0
    for group_probs, group_labels in zip(probs, labels):
        group_preds = group_probs.argmax(-1)

        if (group_preds.sort()[0] == torch.arange(3)).all():
            removed_correct += (group_preds == torch.LongTensor(group_labels)).sum()
            removed_count += len(group_preds)
        else:
            new_probs.append(group_probs)
            new_labels.append(group_labels)

    print(removed_correct, removed_count)
    return torch.stack(new_probs, dim=0), torch.stack(new_labels, dim=0), int(removed_correct), int(removed_count)

def accuracy(preds, labels):
    return (preds == labels).sum().float() / len(preds)

def heuristic(probs, labels):
    """
    probs: (N, N_CLASSES)
    labels: (N,)
    """
    probs = probs.reshape(-1, N_MEMBERS, N_CLASSES)  # (N_GROUPS, N_MEMBERS, N_CLASSES)
    preds = probs.argmax(dim=-1)  # (N_GROUPS, N_MEMBERS)
    for group_probs, group_preds in zip(probs, preds):
        if (group_preds.sort()[0] == torch.arange(3)).all():
            continue
        map = {
            (group_probs[0][0], group_probs[1][1], group_probs[2][2]): (0, 1, 2),
            (group_probs[0][0], group_probs[1][2], group_probs[2][1]): (0, 2, 1),
            (group_probs[0][1], group_probs[1][0], group_probs[2][2]): (1, 0, 2),
            (group_probs[0][1], group_probs[1][2], group_probs[2][0]): (1, 2, 0),
            (group_probs[0][2], group_probs[1][0], group_probs[2][1]): (2, 0, 1),
            (group_probs[0][2], group_probs[1][1], group_probs[2][0]): (2, 1, 0)
        }
        max_conf = (-99999999999999, -99999999999999, -999999999999999)

        for k in map.keys():
            if k[0] + k[1] + k[2] > max_conf[0] + max_conf[1] + max_conf[2]:
                max_conf = k

        indices = map[max_conf]
        for j in (0, 1, 2):
            group_preds[j] = indices[j]

    print(f'Heuristic accuracy: {accuracy(preds.reshape(-1), labels)}')

def load_single(pred_json, gold_tsv, batch_size):
    labels = read_tsv(gold_tsv, -1, False)
    labels = torch.LongTensor([LABEL2IDX[l] for l in labels])
    probs = torch.FloatTensor(read_json(pred_json)['scores'])  # (N * N_CLASSES,)
    probs = probs.reshape(-1, N_CLASSES)  # (N, N_CLASSES)
    assert len(probs) == len(labels)

    preds = probs.argmax(dim=-1)
    print(f'Single accuracy: {accuracy(preds, labels)}')
    heuristic(probs, labels)

    return _load(probs, labels)

def load_ensemble(pred_jsons, gold_tsv, batch_size):
    # member analysis
    for pred_json in pred_jsons:
        load_single(pred_json, gold_tsv, batch_size)

    labels = read_tsv(gold_tsv, -1, False)
    labels = torch.LongTensor([LABEL2IDX[l] for l in labels])
    probs_sum = None
    for pred_json in pred_jsons:
        probs = torch.FloatTensor(read_json(pred_json)['scores'])  # (N * N_CLASSES,)
        probs = probs.reshape(-1, N_CLASSES)  # (N, N_CLASSES)
        if probs_sum is None:
            probs_sum = torch.zeros_like(probs)
        probs_sum += torch.FloatTensor(probs)

    preds = probs_sum.argmax(dim=-1)
    print(f'Ensemble accuracy: {accuracy(preds, labels)}')
    heuristic(probs, labels)

    return _load(probs, labels)

def _load(probs, labels):
    """
    probs: (N, N_CLASSES)
    labels: (N,)
    """
    probs = probs.reshape(-1, N_MEMBERS, N_CLASSES)  # (N_GROUPS, N_MEMBERS, N_CLASSES)
    labels = labels.reshape(-1, N_MEMBERS)  # (N_GROUPS, N_MEMBERS)

    # if there's no conflict, then there's no point in doing conflict resolution
    removed_correct = removed_count = 0
    probs, labels, removed_correct, removed_count = remove_non_conflicts(probs, labels)

    zipped = list(zip(probs, labels))
    tenth = int(len(zipped) / 10)

    def generate_dataloader(zipped_data, start, end, shuffle):
        probs, labels = zip(*zipped_data[start:end])
        dataset = TensorDataset(torch.stack(probs, dim=0), torch.stack(labels, dim=0))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_dataloader = generate_dataloader(zipped, 0, 9 * tenth, True)
    test_dataloader = generate_dataloader(zipped, 9 * tenth, len(zipped), False)
    all_dataloader = generate_dataloader(zipped, 0, len(zipped), False)

    return train_dataloader, test_dataloader, all_dataloader, removed_correct, removed_count

if __name__ == '__main__':
    batch_size = 8
    hidden_dim = 16
    n_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_train = True
    model_save_path = 'postpro_model.bin'

    gold_tsv = 'data/dev.tsv'

    # enhancement on single model
    pred_json = 'checkpoints/model/mednli_dev_scores_N.json'  # sub N with best epoch
    train_dataloader, test_dataloader, all_dataloader, removed_correct, removed_count = load_single(pred_json, gold_tsv, batch_size)

    # enhancement on ensemble model
    # pred_jsons = ['checkpoints/model_1/mednli_dev_scores_N.json',  # sub N with best epoch
    #               'checkpoints/model_2/mednli_dev_scores_M.json'
    #               ]
    # train_dataloader, test_dataloader, all_dataloader, removed_correct, removed_count = load_ensemble(pred_jsons, gold_tsv, batch_size)

    if is_train:
        model = PostProModel(N_CLASSES, N_MEMBERS, hidden_dim)
        train(model, train_dataloader, test_dataloader, all_dataloader, n_epochs, device, removed_correct, removed_count)
        torch.save(model, model_save_path)
    else:
        model = torch.load(model_save_path).to(device)
        all_acc = evaluate(model, all_dataloader, device, removed_correct, removed_count)
        print(all_acc)
