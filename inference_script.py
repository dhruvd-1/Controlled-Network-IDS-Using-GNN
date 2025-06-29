import json
import time
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors
import joblib
import pickle
import os

# ==== Load Preprocessing Info ====
with open("models/preprocessing_info.pkl", "rb") as f:
    preproc = pickle.load(f)

scaler = joblib.load("models/scaler.pkl")

feature_cols = preproc["feature_columns"]
numeric_cols = preproc["numeric_cols"]
categorical_cols = preproc["categorical_cols"]
normal_class_index = preproc["normal_class_index"]
input_dim = preproc["input_dim"]

# ==== Define GNN Model ====
import torch.nn as nn
import torch.nn.functional as F

class GNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNN_Model, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNN_Model(input_dim=input_dim, hidden_dim=128, num_classes=2).to(device)
model.load_state_dict(torch.load("models/gnn_intrusion_detection_model.pth", map_location=device))
model.eval()

# ==== Watch JSONL File and Process 5 at a Time ====
def read_jsonl_tail(filepath, start_line):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines[start_line:], len(lines)

def preprocess_logs(log_batch):
    df = pd.DataFrame(log_batch)
    #print(df.columns)
    #print(df.head())

    # Step 1: Map feature column names to index-based column names
    column_name_to_index = {
        'duration': 0, 'protocol_type': 1, 'service': 2, 'flag': 3,
        'src_bytes': 4, 'dst_bytes': 5, 'land': 6, 'wrong_fragment': 7,
        'urgent': 8, 'hot': 9, 'num_failed_logins': 10, 'logged_in': 11,
        'num_compromised': 12, 'root_shell': 13, 'su_attempted': 14,
        'num_root': 15, 'num_file_creations': 16, 'num_shells': 17,
        'num_access_files': 18, 'num_outbound_cmds': 19, 'is_host_login': 20,
        'is_guest_login': 21, 'count': 22, 'srv_count': 23, 'serror_rate': 24,
        'srv_serror_rate': 25, 'rerror_rate': 26, 'srv_rerror_rate': 27,
        'same_srv_rate': 28, 'diff_srv_rate': 29, 'srv_diff_host_rate': 30,
        'dst_host_count': 31, 'dst_host_srv_count': 32,
        'dst_host_same_srv_rate': 33, 'dst_host_diff_srv_rate': 34,
        'dst_host_same_src_port_rate': 35, 'dst_host_srv_diff_host_rate': 36,
        'dst_host_serror_rate': 37, 'dst_host_srv_serror_rate': 38,
        'dst_host_rerror_rate': 39, 'dst_host_srv_rerror_rate': 40,
        'difficulty_level': 42  
    }

    # Fill missing columns
    for col in column_name_to_index:
        if col not in df.columns:
            df[col] = 0.0 if col in numeric_cols else "unknown"
        elif col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = df[col].fillna("unknown")

    # Rename columns to their index names (match what scaler expects)
    df_renamed = df.rename(columns=column_name_to_index)
    #print(df_renamed.columns)

    # Only keep numeric columns for scaling
    numeric_indices = numeric_cols
    df_numeric = df_renamed[numeric_indices]

    # Apply scaler (trained on indexed columns)
    scaled_array = scaler.transform(df_numeric)

    # DEBUG
    #print("✅ Scaled shape:", scaled_array.shape)

    df_scaled = pd.DataFrame(scaled_array, columns=numeric_indices)


    # One-hot encode categorical columns
    #print(categorical_cols)
    df_cat = pd.get_dummies(df_renamed[categorical_cols], dummy_na=False, dtype=np.float32)

    #print(df_cat.head())
    df_cat = df_cat[[col for col in df_cat.columns if col in feature_cols]]  # drop unseen

    for col in feature_cols:
        if col not in df_scaled.columns and col not in df_cat.columns:
            df_cat[col] = 0

    # Combine scaled numerics and one-hot categoricals
    df_final = pd.concat([df_scaled, df_cat], axis=1)
    df_final = df_final[feature_cols]  # ensure order matches model input
 
    return df_final


def create_knn_graph(x_np, k=5):
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(x_np)))
    nbrs.fit(x_np)
    indices = nbrs.kneighbors(x_np, return_distance=False)

    rows = np.repeat(np.arange(len(x_np)), indices.shape[1] - 1)
    cols = indices[:, 1:].flatten()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index

print("🚀 Monitoring logs in real time...")

current_line = 0

OUTPUT_PATH = "latest_predictions.json"

BATCH_SIZE = 25  # Or any number you prefer
current_line = 0
print("\n🧠 GNN Predictions:")
while True:
    try:
        lines, total = read_jsonl_tail("network_logs.jsonl", current_line)
        if len(lines) < BATCH_SIZE:
            time.sleep(2)
            continue

        batch_lines = lines[:BATCH_SIZE]
        current_line += len(batch_lines)

        log_batch = [json.loads(line) for line in batch_lines]
        X = preprocess_logs(log_batch)
        x_tensor = torch.tensor(X.values, dtype=torch.float32)
        edge_index = create_knn_graph(X.values, k=5)

        data = Data(x=x_tensor, edge_index=edge_index)
        data = data.to(device)

        with torch.no_grad():
            out = model(data)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        print("\n")  
        for i, log in enumerate(log_batch):
            result = "ATTACK" if preds[i] == 1 else "NORMAL"
            print(f"📍 Node {log.get('node_id')} → {result} (Confidence: {probs[i]:.2f})")

        result_data = []
        for i, log in enumerate(log_batch):
            result_data.append({
                "timestamp": log.get("timestamp"),
                "node_id": log.get("node_id"),
                "prediction": "ATTACK" if preds[i] == 1 else "NORMAL",
                "confidence": float(f"{probs[i]:.2f}")
            })

        # Save output
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result_data, f, indent=2)

        time.sleep(5)

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
        break
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        time.sleep(2)


