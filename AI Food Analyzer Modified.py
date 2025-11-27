# train_vit_xgb.py
import os
import re
import math
import joblib
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

# PyTorch / torchvision for ViT
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# Model / training
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

#config

DATA_DIR = ""   # path where zip was extracted
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_EXTRACT = 32
IMG_SIZE = 224
RANDOM_STATE = 42
XGB_MODEL_PATH = "xgb_vit_model.joblib"
SCALER_PATH = "scaler.joblib"

#parse

LABEL_RE = re.compile(r".*\((\d+)\s*-\s*(\d+)\).*")
def parse_label_from_folder(foldername):
    """
    Input: foldername string, e.g. "Apple(1-5)" or "Banana (2-7)"
    Returns: float midpoint label, e.g. 3.0
    Raises ValueError if pattern not found.
    """
    m = LABEL_RE.match(foldername)
    if not m:
        raise ValueError(f"Folder name '{foldername}' does not match pattern 'name(min-max)'.")
    a = int(m.group(1))
    b = int(m.group(2))
    return (a + b) / 2.0

#build list

def collect_image_paths_and_labels(root_dir):
    root = Path(root_dir)
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        try:
            label = parse_label_from_folder(sub.name)
        except ValueError:
            # skip folders that don't follow pattern
            print(f"Skipping folder (cannot parse label): {sub}")
            continue
        # collect image files under this folder (non-recursive)
        img_files = [p for p in sub.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        if not img_files:
            print(f"No images found in {sub}, skipping.")
            continue
        for p in img_files:
            rows.append((str(p), float(label)))
    df = pd.DataFrame(rows, columns=["path", "label"])
    return df

#ViT

def build_vit_feature_extractor(device=DEVICE):
    # Try torchvision.models.vit_b_16 (available in torchvision >= 0.12+)
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
    except Exception:
        raise RuntimeError("torchvision ViT model not available in this environment. "
                           "Ensure torchvision >= 0.12 or install timm and adjust code to use timm.models.vit_base_patch16_224.")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights).to(device)
    vit.eval()
    # Remove the classification head: get the representation before final head.
    # torchvision's vit has vit.heads or classifier attribute; we'll create a model that outputs the embedding.
    # Implementation detail depends on torchvision version; this approach uses forward to get features from vit.encoder
    class ViTFeatureExtractor(nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model
            # For newer torchvision the attribute for head is 'heads' or 'head'
            # We'll rely on vit.forward returning logits; instead we replicate forward up to pre_logits.
        def forward(self, x):
            # Use vit._process_input & vit.encoder if available.
            # Simpler: call vit.forward_features if available (returns embeddings). Check attribute:
            if hasattr(self.vit, "forward_features"):
                feats = self.vit.forward_features(x)  # shape (B, C)
            else:
                # fallback: run forward and remove head - but that may compute logits; hope forward_features exists
                feats = self.vit(x)
            return feats
    feat_model = ViTFeatureExtractor(vit).to(device)
    return feat_model, weights.transforms()

#feature extraction
def extract_features_dataframe(df, feature_model, preprocess_transform, batch_size=BATCH_EXTRACT, device=DEVICE):
    n = len(df)
    X_list = []
    y_list = []
    feature_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Extracting features"):
            batch = df.iloc[i:i+batch_size]
            imgs = []
            for p in batch["path"]:
                img = Image.open(p).convert("RGB")
                img = preprocess_transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0).to(device)
            feats = feature_model(imgs)
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            X_list.append(feats)
            y_list.extend(batch["label"].values.tolist())
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)
    return X, y

#eval

def train_xgb(X_train, y_train, X_val, y_val, random_state=RANDOM_STATE):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
        "verbosity": 1,
    }
    watchlist = [(dtrain, "train"), (dval, "val")]
    num_boost_round = 1000
    early_stopping_rounds = 50
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=20,
    )
    return model

def evaluate_model(model, X, y):
    dmat = xgb.DMatrix(X)
    preds = model.predict(dmat)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2, "preds": preds}

def train_and_evaluate_sklearn_model(model, X_train, y_train, X_test, y_test, name="Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    return model, preds

def main():
    print("Collecting image paths and labels...")
    df = collect_image_paths_and_labels(DATA_DIR)
    if df.empty:
        raise RuntimeError(f"No images found under {DATA_DIR}. Check path.")
    print(f"Found {len(df)} images across {df.path.apply(lambda p: Path(p).parent.name).nunique()} folders.")
    print(df.head())

    # split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    print("Building ViT feature extractor...")
    feat_model, preprocess_transform = build_vit_feature_extractor()
    print("Extracting train features...")
    X_train, y_train = extract_features_dataframe(train_df.reset_index(drop=True), feat_model, preprocess_transform)
    print("Extracting test features...")
    X_test, y_test = extract_features_dataframe(test_df.reset_index(drop=True), feat_model, preprocess_transform)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training XGBoost regressor...")
    xgb_model = train_xgb(X_train_scaled, y_train, X_test_scaled, y_test)

    print("Evaluating on test set...")
    eval_res = evaluate_model(xgb_model, X_test_scaled, y_test)
    print("Test MAE:", eval_res["mae"])
    print("Test RMSE:", eval_res["rmse"])
    print("Test R2:", eval_res["r2"])

    joblib.dump(xgb_model, XGB_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved XGBoost model to {XGB_MODEL_PATH} and scaler to {SCALER_PATH}.")

    def predict_shelf_life(image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess_transform(img).unsqueeze(0).to(DEVICE)  # (1,C,H,W)
        feat = feat_model(img_t)
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feat_scaled = scaler.transform(feat)
        dmat = xgb.DMatrix(feat_scaled)
        pred = xgb_model.predict(dmat)[0]
        return float(pred)

    sample_paths = test_df["path"].sample(min(5, len(test_df)), random_state=RANDOM_STATE).tolist()
    print("Sample predictions on test images:")
    for p in sample_paths:
        pred = predict_shelf_life(p)
        true = float(test_df.loc[test_df["path"] == p, "label"].iloc[0])
        print(f"{p} -> predicted: {pred:.3f}, true midpoint: {true}")

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rf_model, rf_preds = train_and_evaluate_sklearn_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")
    joblib.dump(rf_model, "rf_vit_model.joblib")


    ridge_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge_model, ridge_preds = train_and_evaluate_sklearn_model(ridge_model, X_train_scaled, y_train, X_test_scaled, y_test, "Ridge Regression")
    joblib.dump(ridge_model, "ridge_vit_model.joblib")


    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model, knn_preds = train_and_evaluate_sklearn_model(knn_model, X_train_scaled, y_train, X_test_scaled, y_test, "KNN Regressor")
    joblib.dump(knn_model, "knn_vit_model.joblib")

    print("All models trained and saved.")

if __name__ == "__main__":
    main()

XGB_MODEL_PATH = "xgb_vit_model.joblib"
SCALER_PATH = "scaler.joblib"

xgb_model = joblib.load(XGB_MODEL_PATH) #insert model path
scaler = joblib.load(SCALER_PATH)

print("Model and scaler loaded successfully.")

feat_model, preprocess_transform = build_vit_feature_extractor()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
feat_model.to(DEVICE)
feat_model.eval()

def predict_shelf_life(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess_transform(img).unsqueeze(0).to(DEVICE)


    feat = feat_model(img_t)
    if isinstance(feat, torch.Tensor):
        feat = feat.detach().cpu().numpy()


    feat_scaled = scaler.transform(feat)


    dmat = xgb.DMatrix(feat_scaled)
    pred = xgb_model.predict(dmat)[0]

    return float(pred)

new_image_path = "path_to_your_image.jpg"  # replace with actual image path
predicted_shelf_life = predict_shelf_life(new_image_path)
print(f"Predicted shelf-life midpoint: {predicted_shelf_life:.2f}")

