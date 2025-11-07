import sqlite3
import random
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch


def fetch_balanced_samples(db_path, num_samples_per_label=1600):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(samples)")
    columns = [column[1] for column in cursor.fetchall()]
    feature_columns = columns[2:]
    label_column = columns[1]

    cursor.execute(f"SELECT DISTINCT {label_column} FROM samples")
    labels = [row[0] for row in cursor.fetchall()]

    balanced_samples = []
    for label in labels:
        query = f"SELECT {','.join(feature_columns)} FROM samples WHERE {label_column} = ?"
        cursor.execute(query, (label,))
        rows = cursor.fetchall()

        if len(rows) < num_samples_per_label:
            selected_rows = rows
        else:
            if label == 1:
                selected_rows = random.sample(rows, num_samples_per_label * 0)
            elif label == 0:
                selected_rows = random.sample(rows, num_samples_per_label * 1)
            elif label == 2:
                selected_rows = random.sample(rows, num_samples_per_label * 0)
            elif label == 3:
                selected_rows = random.sample(rows, num_samples_per_label * 0)
            else:
                selected_rows = random.sample(rows, num_samples_per_label * 0)

        for row in selected_rows:
            features = list(row)
            balanced_samples.append((features, label))

    conn.close()
    return balanced_samples


class FlowDataset(Dataset):
    def __init__(self, samples, mean, std):
        feature_lengths = [len(s[0]) for s in samples]
        if len(set(feature_lengths)) > 1:
            print("警告：特征长度不一致")
            min_length = min(feature_lengths)
            samples = [(s[0][:min_length], s[1]) for s in samples]

        self.X = np.array([s[0] for s in samples], dtype=np.float32)
        self.y = np.array([s[1] for s in samples], dtype=np.int64)
        self.mean = mean
        self.std = std
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

    def ret_numpy(self, idx):
        return self.X[idx]


def load_dataset():
    samples = fetch_balanced_samples("./case30_samples_PQ_thin.db")
    with open('./best_discriminator_二分类_许_data.pkl', 'rb') as f:
        data_feature = pickle.load(f)
    return FlowDataset(samples, mean=data_feature['mean'], std=data_feature['std'])