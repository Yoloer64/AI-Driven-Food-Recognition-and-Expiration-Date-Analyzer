## Lightweight Code to Call Model and Demonstrate Prediction ##
import torch
import numpy as np
from PIL import Image
import joblib
import xgboost as xgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

xgb_model = joblib.load("xgb_vit_model.joblib")
scaler = joblib.load("scaler.joblib")
from softwaree import build_vit_feature_extractor
feat_model, preprocess_transform = build_vit_feature_extractor(device=DEVICE)
feat_model.to(DEVICE)
feat_model.eval()

def predict_shelf_life(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = feat_model(img_t)
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    dmat = xgb.DMatrix(features_scaled)
    pred = xgb_model.predict(dmat)[0]

    return float(pred)

if __name__ == "__main__":
    image_path = "" #Insert path to image (i.e. C:/Users/users/Downloads/ApplePicture.jpg)
    result = predict_shelf_life(image_path)
    print(f"Predicted shelf-life midpoint: {result:.2f} days")

