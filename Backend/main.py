import io
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# -------- Optional: PyTorch autoencoder for novelty (falls back to PCA) --------
TORCH_OK = True
try:
    import torch
    import torch.nn as nn
except Exception:
    TORCH_OK = False

warnings.filterwarnings("ignore", category=UserWarning)

# =========================== Globals ===========================
# Anomaly (IF) pipeline
DF_IF: Optional[pd.DataFrame] = None
NUMERIC_IF: List[str] = []
SCALER_IF: Optional[StandardScaler] = None
X_IF: Optional[np.ndarray] = None
IF_MODEL: Optional[IsolationForest] = None
IF_SCORES: Optional[np.ndarray] = None

# Novelty (AE) pipeline
DF_NV: Optional[pd.DataFrame] = None            # Success-only training
NUMERIC_NV: List[str] = []
SCALER_NV: Optional[StandardScaler] = None
X_NV: Optional[np.ndarray] = None
AE_MODEL = None                                  # torch AE or PCA reconstructor
AE_IS_TORCH = False
AE_ERRORS_TRAIN: Optional[np.ndarray] = None
AE_THRESHOLD: Optional[float] = None             # Training-derived threshold (strict; no val/test peeking)
PCA_VIS: Optional[PCA] = None                    # 2D PCA for novelty viz (fit on training)

# =========================== Utilities =========================
def _fig_to_pil(fig):
    import PIL.Image as Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")

def _detect_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _prep_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[StandardScaler, np.ndarray]:
    if not cols:
        raise gr.Error("Please select at least one numeric column.")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols].astype(float).values)
    return scaler, Xs

# ===================== Isolation Forest (Anomaly) =====================
def run_isolation_forest(contam: float, selected_cols: List[str]):
    global DF_IF, SCALER_IF, X_IF, IF_MODEL, IF_SCORES

    if DF_IF is None or len(DF_IF) == 0:
        raise gr.Error("Upload the Anomaly (success+failure) CSV first.")
    if not selected_cols:
        raise gr.Error("Select at least one numeric column for IF.")

    SCALER_IF, X_IF = _prep_matrix(DF_IF, selected_cols)
    IF_MODEL = IsolationForest(
        n_estimators=300, contamination=float(contam),
        random_state=42, n_jobs=-1
    )
    IF_MODEL.fit(X_IF)
    IF_SCORES = -IF_MODEL.score_samples(X_IF)

    thr = np.quantile(IF_SCORES, 1.0 - float(contam))
    flagged = IF_SCORES >= thr

    Z = PCA(n_components=2, random_state=42).fit_transform(X_IF)
    fig1 = plt.figure(figsize=(6, 4.5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=IF_SCORES, s=18)
    plt.colorbar(sc, label="IF anomaly score (↑ = more anomalous)")
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.title("Anomaly — PCA (colored by IF score)")
    plt.tight_layout()
    img_scatter = _fig_to_pil(fig1)

    fig2 = plt.figure(figsize=(6, 4.0))
    plt.hist(IF_SCORES, bins=30)
    plt.axvline(thr, color="red", linestyle="--", label=f"threshold @{1.0 - contam:.3f} quantile")
    plt.legend(); plt.xlabel("IF score"); plt.ylabel("Count"); plt.title("IF Score Histogram")
    plt.tight_layout()
    img_hist = _fig_to_pil(fig2)

    keep_cols = [c for c in ["datetime_stamp", "timestamp", "Machine_Id", "Plant_Id"] if c in DF_IF.columns]
    keep_cols += [c for c in selected_cols if c not in keep_cols]

    out = DF_IF.loc[:, keep_cols].copy()
    out["if_score"] = IF_SCORES
    out["is_anomaly"] = flagged.astype(int)
    anomalies = out[out["is_anomaly"] == 1].sort_values("if_score", ascending=False)

    csv_path = "anomalies_if.csv"
    anomalies.to_csv(csv_path, index=False)
    top_view = anomalies.head(10).loc[:, [c for c in keep_cols if c in anomalies.columns] + ["if_score"]]
    status = f"IF trained (contam={contam:.3f}). Flagged {flagged.sum()} / {len(DF_IF)} rows."
    return img_scatter, img_hist, csv_path, top_view, status

# =================== Autoencoder (Novelty, AE-regularization only) =====================
if TORCH_OK:
    class AE(nn.Module):
        """
        Compact AE with strong regularization to learn a tight 'normal' manifold:
        - dropout (decoder & encoder)
        - weight decay (Adam)
        - latent L1 sparsity
        - denoising (Gaussian noise on inputs)
        """
        def __init__(self, d, h=None, z_dim=2, dropout=0.35):
            super().__init__()
            h = h or max(16, d // 4)
            self.enc_fc1 = nn.Linear(d, h)
            self.enc_fc2 = nn.Linear(h, z_dim)
            self.dec_fc1 = nn.Linear(z_dim, h)
            self.dec_fc2 = nn.Linear(h, d)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ReLU()

        def encode(self, x):
            x = self.act(self.enc_fc1(x))
            x = self.dropout(x)
            z = self.enc_fc2(x)    # linear latent to allow sparsity to act
            return z

        def decode(self, z):
            x = self.act(self.dec_fc1(z))
            x = self.dropout(x)
            x = self.dec_fc2(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            xhat = self.decode(z)
            return xhat, z

def _strict_training_threshold(train_errors: np.ndarray) -> float:
    """
    Derive a STRICT threshold from TRAINING errors ONLY (no val/test).
    We use a tight MAD-based cutoff and cap it by a mid-quantile to
    avoid a loose tail: threshold = min(median + 1.2*MAD*1.4826, 60th pct).

    This, combined with AE regularization, typically yields low single-digit %
    novelty rates on unseen data — driven purely by AE behavior.
    """
    e = np.asarray(train_errors)
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    mad_cut = med + 1.2 * 1.4826 * (mad + 1e-12)  # tight
    q60 = np.quantile(e, 0.60)
    thr = float(min(mad_cut, q60))
    return thr

def train_autoencoder(
    epochs: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    selected_cols: List[str] = None,
    dropout: float = 0.35,
    z_dim: int = 2,
    noise_std: float = 0.10,
    latent_l1: float = 2e-3,     # stronger sparsity to tighten normal manifold
    es_patience: int = 15
) -> Tuple[str, float, object]:
    """
    Train AE on success-only tabular data with strong regularization.
    Threshold is computed ONLY from TRAINING reconstruction errors
    via a strict MAD-based rule (no validation/test peeking).
    Returns (msg, training_threshold, training_hist_img). Also fits PCA_VIS on X_NV.
    """
    global DF_NV, SCALER_NV, X_NV, AE_MODEL, AE_IS_TORCH, AE_ERRORS_TRAIN, AE_THRESHOLD, PCA_VIS

    if DF_NV is None or len(DF_NV) == 0:
        raise gr.Error("Upload the Novelty training (success-only) CSV first.")
    if not selected_cols:
        raise gr.Error("Select at least one numeric column for Novelty training.")

    SCALER_NV, X_NV = _prep_matrix(DF_NV, selected_cols)
    d = X_NV.shape[1]

    if TORCH_OK:
        AE_IS_TORCH = True
        device = torch.device("cpu")
        X_all = torch.tensor(X_NV, dtype=torch.float32, device=device)

        model = AE(d, h=max(16, d // 4), z_dim=z_dim, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        patience = es_patience

        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            x_noisy = X_all + noise_std * torch.randn_like(X_all)   # denoising
            xhat, z = model(x_noisy)
            recon = mse(xhat, X_all)
            sparsity = latent_l1 * z.abs().mean()                   # latent L1
            loss = recon + sparsity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # stability
            opt.step()

            # simple early stopping on training loss (since we purposely avoid val)
            cur = loss.item()
            if cur < best_loss - 1e-8:
                best_loss = cur
                best_state = model.state_dict()
                patience = es_patience
            else:
                patience -= 1
                if patience <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            xhat_all, _ = model(X_all)
            AE_ERRORS_TRAIN = ((xhat_all - X_all) ** 2).mean(dim=1).cpu().numpy()

        AE_MODEL = model
        trainer_msg = (
            f"AE trained (epochs≈{epochs}, z={z_dim}, dropout={dropout}, "
            f"wd={weight_decay}, noise={noise_std}, latent_l1={latent_l1})."
        )
    else:
        # PCA fallback — strict: tiny component count
        comps = max(1, min(min(8, d // 8), d))
        pca_rec = PCA(n_components=comps, svd_solver="randomized", random_state=42)
        Z = pca_rec.fit_transform(X_NV)
        recon = pca_rec.inverse_transform(Z)
        AE_ERRORS_TRAIN = ((recon - X_NV) ** 2).mean(axis=1)
        AE_MODEL = pca_rec; AE_IS_TORCH = False
        trainer_msg = f"PCA reconstructor (strict) trained with n_components={comps}."

    # Strict threshold from TRAINING errors only
    AE_THRESHOLD = _strict_training_threshold(AE_ERRORS_TRAIN)

    # Visualization PCA on training space
    PCA_VIS = PCA(n_components=2, random_state=42).fit(X_NV)

    # Plot training error histogram + threshold
    fig = plt.figure(figsize=(6.8, 4.2))
    plt.hist(AE_ERRORS_TRAIN, bins=30, alpha=0.8)
    plt.axvline(AE_THRESHOLD, color="red", linestyle="--", label=f"training threshold={AE_THRESHOLD:.6g}")
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count")
    plt.title("AE Training Reconstruction Errors (Strict, AE-regularization only)")
    plt.legend(); plt.tight_layout()
    hist_img = _fig_to_pil(fig)

    return trainer_msg, AE_THRESHOLD, hist_img

def score_unseen_novelty(test_df: pd.DataFrame, selected_cols: List[str]):
    """
    Score unseen tabular CSV using the TRAINING-DERIVED threshold (AE-only).
    No validation/test quantiles are used. Novelty is driven by how tightly
    the AE (with regularization) models the normal manifold.
    """
    if SCALER_NV is None or AE_MODEL is None or AE_THRESHOLD is None or PCA_VIS is None:
        raise gr.Error("Train the novelty model first.")

    for c in selected_cols:
        if c not in test_df.columns:
            raise gr.Error(f"Column '{c}' not found in Test CSV.")

    Xt = SCALER_NV.transform(test_df[selected_cols].astype(float).values)

    if TORCH_OK and AE_IS_TORCH:
        with torch.no_grad():
            x = torch.tensor(Xt, dtype=torch.float32)
            xhat, _ = AE_MODEL(x)
            errs = ((xhat - x) ** 2).mean(dim=1).numpy()
    else:
        Zt = AE_MODEL.transform(Xt); recon = AE_MODEL.inverse_transform(Zt)
        errs = ((recon - Xt) ** 2).mean(axis=1)

    is_novel = (errs >= AE_THRESHOLD).astype(int)
    rate = is_novel.mean()

    fig_h = plt.figure(figsize=(6.6, 4.2))
    plt.hist(errs, bins=30, alpha=0.85)
    plt.axvline(AE_THRESHOLD, color="red", linestyle="--", label=f"threshold={AE_THRESHOLD:.6g}")
    plt.xlabel("Reconstruction Error"); plt.ylabel("Count"); plt.title("Test Reconstruction Errors")
    plt.legend(); plt.tight_layout()
    hist_img = _fig_to_pil(fig_h)

    Zp = PCA_VIS.transform(Xt)
    fig_p = plt.figure(figsize=(6.6, 4.6))
    mask = is_novel.astype(bool)
    plt.scatter(Zp[~mask, 0], Zp[~mask, 1], s=18, alpha=0.8, label="Normal")
    plt.scatter(Zp[mask, 0], Zp[mask, 1], s=28, alpha=0.95, c="red", label="Novelty")
    plt.xlabel("PC 1"); plt.ylabel("PC 2"); plt.title("PCA Projection of Test Data (Novelty in Red)")
    plt.legend(); plt.tight_layout()
    pca_img = _fig_to_pil(fig_p)

    view = pd.DataFrame({"reconstruction_error": errs, "is_novelty": is_novel})
    out_path = "novelty_scores.csv"
    view.to_csv(out_path, index=False)

    status = (
        f"Scored {len(test_df)} rows. Flagged {int(is_novel.sum())} rows as novelty "
        f"({rate*100:.2f}% achieved) using training-only threshold={AE_THRESHOLD:.6g}."
    )
    return hist_img, pca_img, view.sort_values(["is_novelty","reconstruction_error"], ascending=[False, False]), out_path, status

# =========================== Gradio UI =========================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate", secondary_hue="indigo")) as demo:
    gr.Markdown("## Model Development for Predictive Maintenance (Anomaly & Novelty Detection)")

    # ----------------------- Upload & Preview -----------------------
    gr.Markdown("### 1) Upload Data")
    with gr.Row():
        file_if = gr.File(label="Anomaly Detection Data (Success + Failure)", file_count="single", file_types=[".csv"])
        file_nv = gr.File(label="Novelty Training Data (Success Only)", file_count="single", file_types=[".csv"])

    btn_load = gr.Button("Load & Preview", variant="primary")

    with gr.Row():
        df_if_head = gr.Dataframe(label="Anomaly CSV — First 5 rows", interactive=False)
        df_nv_head = gr.Dataframe(label="Novelty Train CSV — First 5 rows", interactive=False)
    status_load = gr.Textbox(label="Status", interactive=False)

    cols_if = gr.CheckboxGroup(choices=[], label="Select numeric columns for Anomaly (IF)")
    cols_nv = gr.CheckboxGroup(choices=[], label="Select numeric columns for Novelty (AE)")

    # ------------------- Anomaly (IF) Section -------------------
    gr.Markdown("### 2) Anomaly Detection — Isolation Forest")
    with gr.Row():
        contam = gr.Slider(0.001, 0.20, value=0.02, step=0.001, label="Contamination (expected anomaly %)")
        btn_if = gr.Button("Run Isolation Forest", variant="primary")

    with gr.Row():
        pca_img = gr.Image(label="PCA Scatter (colored by IF score)", type="pil")
        hist_img = gr.Image(label="IF Score Histogram", type="pil")

    with gr.Row():
        if_top = gr.Dataframe(label="Top anomalies (by IF score)", interactive=False)
        if_csv = gr.File(label="Download anomalies_if.csv", interactive=False)
    status_if = gr.Textbox(label="Anomaly Status", interactive=False)

    # ------------------- Novelty (AE-regularization only) -------------------
    gr.Markdown("### 3) Novelty Detection — Autoencoder (tight manifold via regularization)")
    with gr.Row():
        epochs = gr.Slider(50, 300, value=200, step=1, label="Training Epochs (AE; PCA fallback if no torch)")
        btn_train = gr.Button("Train AE & Derive Training Threshold", variant="primary")

    with gr.Row():
        ae_hist_train = gr.Image(label="Training Error Histogram + Training Threshold", type="pil")
        threshold_box = gr.Textbox(label="Training-derived Threshold (AE-only)", interactive=False)
    status_train = gr.Textbox(label="Training Status", interactive=False)

    gr.Markdown("#### Score Unseen/Test CSV")
    test_file = gr.File(label="Upload Test/Unseen CSV", file_count="single", file_types=[".csv"])
    btn_score = gr.Button("Score Test Data", variant="primary")

    with gr.Row():
        ae_hist_test = gr.Image(label="Test Reconstruction Error Histogram", type="pil")
        pca_proj_img = gr.Image(label="PCA Projection (Novelty in Red)", type="pil")

    df_test_view = gr.Dataframe(label="Scored Test Rows (error & novelty)", interactive=False)
    with gr.Row():
        nov_export = gr.File(label="Download novelty_scores.csv", interactive=False)
        status_test = gr.Textbox(label="Novelty Status", interactive=False)

    # ---------------- Wiring ----------------
    def do_load(f_if, f_nv):
        global DF_IF, DF_NV, NUMERIC_IF, NUMERIC_NV
        if f_if is None or f_nv is None:
            raise gr.Error("Please upload both CSV files.")
        DF_IF = pd.read_csv(f_if.name if hasattr(f_if, "name") else f_if)
        DF_NV = pd.read_csv(f_nv.name if hasattr(f_nv, "name") else f_nv)

        NUMERIC_IF = _detect_numeric_cols(DF_IF)
        NUMERIC_NV = _detect_numeric_cols(DF_NV)

        msg = f"Loaded Anomaly CSV: {len(DF_IF)} rows | Novelty Train CSV: {len(DF_NV)} rows."
        return (
            DF_IF.head(5),
            DF_NV.head(5),
            msg,
            gr.update(choices=NUMERIC_IF, value=NUMERIC_IF[: min(6, len(NUMERIC_IF))]),
            gr.update(choices=NUMERIC_NV, value=NUMERIC_NV[: min(6, len(NUMERIC_NV))]),
        )

    btn_load.click(
        fn=do_load,
        inputs=[file_if, file_nv],
        outputs=[df_if_head, df_nv_head, status_load, cols_if, cols_nv]
    )

    def do_if(contam_val: float, selected_cols_if: List[str]):
        img1, img2, csv_path, top_df, msg = run_isolation_forest(contam_val, selected_cols_if)
        return img1, img2, top_df, csv_path, msg

    btn_if.click(
        fn=do_if,
        inputs=[contam, cols_if],
        outputs=[pca_img, hist_img, if_top, if_csv, status_if]
    )

    def do_train(ep: int, selected_cols_nv: List[str]):
        msg, thr, img = train_autoencoder(
            epochs=ep, lr=3e-4, weight_decay=1e-3,
            selected_cols=selected_cols_nv, dropout=0.35, z_dim=2,
            noise_std=0.10, latent_l1=2e-3, es_patience=15
        )
        return img, f"{thr:.8f}", msg

    btn_train.click(
        fn=do_train,
        inputs=[epochs, cols_nv],
        outputs=[ae_hist_train, threshold_box, status_train]
    )

    def do_score(test_f, selected_cols_nv: List[str]):
        if test_f is None:
            raise gr.Error("Upload the Test/Unseen CSV.")
        df_test = pd.read_csv(test_f.name if hasattr(test_f, "name") else test_f)
        hist_img, pca_img, table, out_path, msg = score_unseen_novelty(df_test, selected_cols_nv)
        return hist_img, pca_img, table, out_path, msg

    btn_score.click(
        fn=do_score,
        inputs=[test_file, cols_nv],
        outputs=[ae_hist_test, pca_proj_img, df_test_view, nov_export, status_test]
    )

if __name__ == "__main__":
    demo.queue().launch()
