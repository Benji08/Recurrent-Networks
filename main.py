import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import csv

pad = -2


def pad_collate(batch, pad_value=0):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)
    yy_tensor = torch.tensor(yy, dtype=torch.long)

    return xx_pad, yy_tensor, x_lens


def pad_collate_test(batch, pad_value=pad):
    xx = batch
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=pad_value)

    return xx_pad, x_lens


def compute_classes_weights(classes):
    classes_count = []
    for i in range(5):
        classes_count.append(len([j for j in classes if j == i]))

    classes_count = torch.tensor(classes_count)
    weights = 1. / classes_count.float()
    sample_weights = torch.tensor([weights[t] for t in classes])
    return sample_weights


class LSTM_Seq_Regressor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, out_size, bidirectional=False, drop_out=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.proj_size = out_size
        if bidirectional:
            self.bidirectional = 2
        else:
            self.bidirectional = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            proj_size=out_size, bidirectional=bidirectional, dropout=drop_out)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.proj_size)
        state = torch.zeros(self.num_layers * self.bidirectional, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        all_outputs, hidden = self.lstm(x, hidden)
        return all_outputs, hidden


class VariableLenDataset(Dataset):
    def __init__(self, in_data, target):
        self.data = [(torch.Tensor(x).float(), torch.tensor(y).long()) for x, y in zip(in_data, target)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data, target = self.data[idx]
        return in_data, target


class VariableLenDatasetTest(Dataset):
    def __init__(self, in_data):
        self.data = [torch.Tensor(x).float() for x in in_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        in_data = self.data[idx]
        return in_data

    def __print__(self):
        print(self.data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

rng = np.random.default_rng(2137)
with open('/kaggle/input/dataset-sieci-rekurencyjne/train.pkl', 'rb') as file:
    train_data = pickle.load(file)

with open('/kaggle/input/dataset-sieci-rekurencyjne/test_no_target.pkl', 'rb') as file:
    test_data = pickle.load(file)


data = [d[0] for d in train_data]
targets = [d[1] for d in train_data]

data = np.array(data, dtype=object)
targets = np.array(targets)

train_indices = rng.random(len(data)) > 0.3
targets_train = [x for i, x in enumerate(targets) if train_indices[i]]
targets_valid = [x for i, x in enumerate(targets) if not train_indices[i]]
data_train = [x for i, x in enumerate(data) if train_indices[i]]
data_valid = [x for i, x in enumerate(data) if not train_indices[i]]

train_set = VariableLenDataset(data_train, targets_train)
valid_set = VariableLenDataset(data_valid, targets_valid)
test_set = VariableLenDatasetTest(test_data)

samples_weights_train = compute_classes_weights(targets_train)
sampler_train = torch.utils.data.WeightedRandomSampler(samples_weights_train, len(samples_weights_train), replacement=True)

train_loader = DataLoader(train_set, batch_size=128, collate_fn=pad_collate, sampler=sampler_train)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=pad_collate)
test_loader = DataLoader(test_set, batch_size=128, drop_last=False, shuffle=False, collate_fn=pad_collate_test)


def eval(loader, model):
    model.eval()

    accuracy_metric = Accuracy(task="multiclass", num_classes=5).to(device)
    precision_metric = Precision(task="multiclass", average='macro', num_classes=5).to(device)
    recall_metric = Recall(task="multiclass", average='macro', num_classes=5).to(device)
    f1_metric = F1Score(task="multiclass", average='macro', num_classes=5).to(device)

    with torch.no_grad():
        for x, targets, x_len in loader:
            x = x.to(device).unsqueeze(2)
            targets = targets.to(device)

            hidden, state = model.init_hidden(x.size(0))
            hidden, state = hidden.to(device), state.to(device)

            x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
            preds_packed, _ = model(x_packed, (hidden, state))
            preds, pred_len = pad_packed_sequence(preds_packed, batch_first=True, padding_value=pad)

            preds = preds.squeeze(2)
            last_outputs = []
            for i, length in enumerate(pred_len):
                last_outputs.append(preds[i, length - 1, :])
            last_outputs = torch.stack(last_outputs)

            _, predictions = torch.max(last_outputs, 1)
            # # collect the correct predictions for each class
            # for label, prediction in zip(targets, predictions):
            #     if label == prediction:
            #         correct_pred += 1
            #     total_pred += 1

            accuracy_metric.update(predictions, targets)
            precision_metric.update(predictions, targets)
            recall_metric.update(predictions, targets)
            f1_metric.update(predictions, targets)

    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")


model = LSTM_Seq_Regressor(1, 50, 2, 5, bidirectional=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fun = nn.CrossEntropyLoss()

for epoch in range(101):
    for x, targets, x_len in train_loader:
        model.train()
        x = x.to(device).unsqueeze(2)
        targets = targets.to(device)

        hidden, state = model.init_hidden(x.size(0))
        hidden, state = hidden.to(device), state.to(device)

        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        preds_packed, _ = model(x_packed, (hidden, state))
        preds, pred_len = pad_packed_sequence(preds_packed, batch_first=True, padding_value=pad)

        preds = preds.squeeze(2)
        last_outputs = []
        for i, length in enumerate(pred_len):
            last_outputs.append(preds[i, length - 1, :])
        last_outputs = torch.stack(last_outputs)

        optimizer.zero_grad()
        loss = loss_fun(last_outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.3}")

print("Started computing for valid dataset...")
eval(valid_loader, model)
print("Started computing for train dataset...")
eval(train_loader, model)

all_predictions_list = []
with torch.no_grad():
    for x, x_len in test_loader:
        x = x.to(device).unsqueeze(2)
        hidden, state = model.init_hidden(x.shape[0])
        hidden, state = hidden.to(device), state.to(device)

        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        preds_packed, _ = model(x_packed, (hidden, state))
        preds, pred_len = pad_packed_sequence(preds_packed, batch_first=True, padding_value=pad)

        preds = preds.squeeze(2)
        last_outputs = []
        for i, length in enumerate(pred_len):
            last_outputs.append(preds[i, length - 1, :])
        last_outputs = torch.stack(last_outputs)

        _, predictions = torch.max(last_outputs, 1)
        for i in predictions.tolist():
            all_predictions_list.append(i)

with open('piatek_Pawlak_Tomczykowski.csv', 'w') as f:
    writer = csv.writer(f)
    for pred in all_predictions_list:
        writer.writerow(str(pred))
