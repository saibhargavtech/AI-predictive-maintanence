import io
import warnings
from typing import Optional, List, Tuple
import tempfile
import os

import numpy as np
import pandas as pd
import streamlit as st
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

# =========================== Page Configuration ===========================
# Only set page config if running as standalone app
if __name__ == "__main__":
    st.set_page_config(
        page_title="ML Model Development - Predictive Maintenance",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# =========================== Session State Initialization ===========================
if 'df_if' not in st.session_state:
    st.session_state.df_if = None
if 'df_nv' not in st.session_state:
    st.session_state.df_nv = None
if 'numeric_if' not in st.session_state:
    st.session_state.numeric_if = []
if 'numeric_nv' not in st.session_state:
    st.session_state.numeric_nv = []
if 'scaler_if' not in st.session_state:
    st.session_state.scaler_if = None
if 'scaler_nv' not in st.session_state:
    st.session_state.scaler_nv = None
if 'if_model' not in st.session_state:
    st.session_state.if_model = None
if 'ae_model' not in st.session_state:
    st.session_state.ae_model = None
if 'ae_threshold' not in st.session_state:
    st.session_state.ae_threshold = None
if 'pca_vis' not in st.session_state:
    st.session_state.pca_vis = None

# Results storage for persistence
if 'if_results' not in st.session_state:
    st.session_state.if_results = None
if 'ae_results' not in st.session_state:
    st.session_state.ae_results = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

# =========================== Utilities =========================
def _detect_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _prep_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[StandardScaler, np.ndarray]:
    if not cols:
        st.error("Please select at least one numeric column.")
        return None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[cols].astype(float).values)
    return scaler, Xs

# ===================== Isolation Forest (Anomaly) =====================
def run_isolation_forest(contam: float, selected_cols: List[str]):
    """Run Isolation Forest anomaly detection - EXACTLY like Gradio version"""
    if st.session_state.df_if is None or len(st.session_state.df_if) == 0:
        st.error("Upload the Anomaly (success+failure) CSV first.")
        return None, None, None, None, None
    
    if not selected_cols:
        st.error("Select at least one numeric column for IF.")
        return None, None, None, None, None

    scaler, X = _prep_matrix(st.session_state.df_if, selected_cols)
    if scaler is None:
        return None, None, None, None, None
    
    st.session_state.scaler_if = scaler
    
    # Train Isolation Forest - EXACTLY like Gradio
    if_model = IsolationForest(
        n_estimators=300, contamination=float(contam),
        random_state=42, n_jobs=-1
    )
    if_model.fit(X)
    if_scores = -if_model.score_samples(X)
    
    st.session_state.if_model = if_model
    
    # Calculate threshold and flagged points
    thr = np.quantile(if_scores, 1.0 - float(contam))
    flagged = if_scores >= thr
    
    # PCA visualization - EXACTLY like Gradio
    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    fig1 = plt.figure(figsize=(6, 4.5))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=if_scores, s=18)
    plt.colorbar(sc, label="IF anomaly score (↑ = more anomalous)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Anomaly — PCA (colored by IF score)")
    plt.tight_layout()
    
    # Histogram - EXACTLY like Gradio
    fig2 = plt.figure(figsize=(6, 4.0))
    plt.hist(if_scores, bins=30)
    plt.axvline(thr, color="red", linestyle="--", 
                label=f"threshold @{1.0 - contam:.3f} quantile")
    plt.legend()
    plt.xlabel("IF score")
    plt.ylabel("Count")
    plt.title("IF Score Histogram")
    plt.tight_layout()
    
    # Prepare output data - EXACTLY like Gradio
    keep_cols = [c for c in ["datetime_stamp", "timestamp", "Machine_Id", "Plant_Id"] 
                 if c in st.session_state.df_if.columns]
    keep_cols += [c for c in selected_cols if c not in keep_cols]
    
    out = st.session_state.df_if.loc[:, keep_cols].copy()
    out["if_score"] = if_scores
    out["is_anomaly"] = flagged.astype(int)
    anomalies = out[out["is_anomaly"] == 1].sort_values("if_score", ascending=False)
    
    top_view = anomalies.head(10).loc[:, 
        [c for c in keep_cols if c in anomalies.columns] + ["if_score"]]
    
    status = f"IF trained (contam={contam:.3f}). Flagged {flagged.sum()} / {len(st.session_state.df_if)} rows."
    
    return fig1, fig2, top_view, anomalies, status

# =================== Autoencoder (Novelty) =====================
if TORCH_OK:
    class AE(nn.Module):
        """Compact AE with strong regularization to learn a tight 'normal' manifold - EXACTLY like Gradio"""
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
            z = self.enc_fc2(x)
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
    """Derive a STRICT threshold from TRAINING errors ONLY - EXACTLY like Gradio"""
    e = np.asarray(train_errors)
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    mad_cut = med + 1.2 * 1.4826 * (mad + 1e-12)
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
    latent_l1: float = 2e-3,
    es_patience: int = 15
) -> Tuple[str, float, object]:
    """Train AE on success-only tabular data with strong regularization - EXACTLY like Gradio"""
    if st.session_state.df_nv is None or len(st.session_state.df_nv) == 0:
        st.error("Upload the Novelty training (success-only) CSV first.")
        return None, None, None
    
    if not selected_cols:
        st.error("Select at least one numeric column for Novelty training.")
        return None, None, None

    scaler, X = _prep_matrix(st.session_state.df_nv, selected_cols)
    if scaler is None:
        return None, None, None
    
    st.session_state.scaler_nv = scaler
    d = X.shape[1]

    if TORCH_OK:
        device = torch.device("cpu")
        X_all = torch.tensor(X, dtype=torch.float32, device=device)

        model = AE(d, h=max(16, d // 4), z_dim=z_dim, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        patience = es_patience

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            x_noisy = X_all + noise_std * torch.randn_like(X_all)
            xhat, z = model(x_noisy)
            recon = mse(xhat, X_all)
            sparsity = latent_l1 * z.abs().mean()
            loss = recon + sparsity
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Early stopping
            cur = loss.item()
            if cur < best_loss - 1e-8:
                best_loss = cur
                best_state = model.state_dict()
                patience = es_patience
            else:
                patience -= 1
                if patience <= 0:
                    break
            
            # Update progress
            progress_bar.progress((ep + 1) / epochs)
            status_text.text(f'Epoch {ep + 1}/{epochs} - Loss: {cur:.6f}')

        progress_bar.empty()
        status_text.empty()

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            xhat_all, _ = model(X_all)
            ae_errors_train = ((xhat_all - X_all) ** 2).mean(dim=1).cpu().numpy()

        st.session_state.ae_model = model
        trainer_msg = (
            f"AE trained (epochs≈{epochs}, z={z_dim}, dropout={dropout}, "
            f"wd={weight_decay}, noise={noise_std}, latent_l1={latent_l1})."
        )
    else:
        # PCA fallback - EXACTLY like Gradio
        comps = max(1, min(min(8, d // 8), d))
        pca_rec = PCA(n_components=comps, svd_solver="randomized", random_state=42)
        Z = pca_rec.fit_transform(X)
        recon = pca_rec.inverse_transform(Z)
        ae_errors_train = ((recon - X) ** 2).mean(axis=1)
        st.session_state.ae_model = pca_rec
        trainer_msg = f"PCA reconstructor (strict) trained with n_components={comps}."

    # Strict threshold from TRAINING errors only
    ae_threshold = _strict_training_threshold(ae_errors_train)
    st.session_state.ae_threshold = ae_threshold

    # Visualization PCA on training space - EXACTLY like Gradio
    st.session_state.pca_vis = PCA(n_components=2, random_state=42).fit(X)

    # Plot training error histogram + threshold - EXACTLY like Gradio
    fig = plt.figure(figsize=(6.8, 4.2))
    plt.hist(ae_errors_train, bins=30, alpha=0.8)
    plt.axvline(ae_threshold, color="red", linestyle="--", 
                label=f"training threshold={ae_threshold:.6g}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("AE Training Reconstruction Errors (Strict, AE-regularization only)")
    plt.legend()
    plt.tight_layout()

    return trainer_msg, ae_threshold, fig

def score_unseen_novelty(test_df: pd.DataFrame, selected_cols: List[str]):
    """Score unseen tabular CSV using the TRAINING-DERIVED threshold - EXACTLY like Gradio"""
    if (st.session_state.scaler_nv is None or st.session_state.ae_model is None or 
        st.session_state.ae_threshold is None or st.session_state.pca_vis is None):
        st.error("Train the novelty model first.")
        return None, None, None, None, None

    for c in selected_cols:
        if c not in test_df.columns:
            st.error(f"Column '{c}' not found in Test CSV.")
            return None, None, None, None, None

    Xt = st.session_state.scaler_nv.transform(test_df[selected_cols].astype(float).values)

    if TORCH_OK and hasattr(st.session_state.ae_model, 'forward'):
        with torch.no_grad():
            x = torch.tensor(Xt, dtype=torch.float32)
            xhat, _ = st.session_state.ae_model(x)
            errs = ((xhat - x) ** 2).mean(dim=1).numpy()
    else:
        Zt = st.session_state.ae_model.transform(Xt)
        recon = st.session_state.ae_model.inverse_transform(Zt)
        errs = ((recon - Xt) ** 2).mean(axis=1)

    is_novel = (errs >= st.session_state.ae_threshold).astype(int)
    rate = is_novel.mean()

    # Histogram - EXACTLY like Gradio
    fig_h = plt.figure(figsize=(6.6, 4.2))
    plt.hist(errs, bins=30, alpha=0.85)
    plt.axvline(st.session_state.ae_threshold, color="red", linestyle="--", 
                label=f"threshold={st.session_state.ae_threshold:.6g}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.title("Test Reconstruction Errors")
    plt.legend()
    plt.tight_layout()

    # PCA projection - EXACTLY like Gradio
    Zp = st.session_state.pca_vis.transform(Xt)
    fig_p = plt.figure(figsize=(6.6, 4.6))
    mask = is_novel.astype(bool)
    plt.scatter(Zp[~mask, 0], Zp[~mask, 1], s=18, alpha=0.8, label="Normal")
    plt.scatter(Zp[mask, 0], Zp[mask, 1], s=28, alpha=0.95, c="red", label="Novelty")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Projection of Test Data (Novelty in Red)")
    plt.legend()
    plt.tight_layout()

    view = pd.DataFrame({"reconstruction_error": errs, "is_novelty": is_novel})
    scored_data = view.sort_values(["is_novelty", "reconstruction_error"], ascending=[False, False])

    status = (
        f"Scored {len(test_df)} rows. Flagged {int(is_novel.sum())} rows as novelty "
        f"({rate*100:.2f}% achieved) using training-only threshold={st.session_state.ae_threshold:.6g}."
    )
    
    return fig_h, fig_p, scored_data, view, status

# Original main UI content removed - now handled by render_ml_backend() function

# =========================== Main ML Backend Function ===========================
def render_ml_backend():
    """Render the ML backend interface"""
    # =========================== Main UI - EXACTLY like Gradio Layout =========================
    st.markdown("## Model Development for Predictive Maintenance (Anomaly & Novelty Detection)")

    # =========================== Upload & Preview Section =========================
    st.markdown("### 1) Upload Data")
    col1, col2 = st.columns(2)

    with col1:
        file_if = st.file_uploader(
            "Anomaly Detection Data (Success + Failure)", 
            type=["csv"], 
            help="Upload CSV with both success and failure cases"
        )

    with col2:
        file_nv = st.file_uploader(
            "Novelty Training Data (Success Only)", 
            type=["csv"], 
            help="Upload CSV with only success cases for training"
        )

    if st.button("Load & Preview", type="primary"):
        if file_if is not None and file_nv is not None:
            try:
                st.session_state.df_if = pd.read_csv(file_if)
                st.session_state.df_nv = pd.read_csv(file_nv)
                
                st.session_state.numeric_if = _detect_numeric_cols(st.session_state.df_if)
                st.session_state.numeric_nv = _detect_numeric_cols(st.session_state.df_nv)
                
                st.success(f"Loaded Anomaly CSV: {len(st.session_state.df_if)} rows | Novelty Train CSV: {len(st.session_state.df_nv)} rows.")
            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
        else:
            st.error("Please upload both CSV files")

    # Data preview - EXACTLY like Gradio
    if st.session_state.df_if is not None and st.session_state.df_nv is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Anomaly CSV — First 5 rows**")
            st.dataframe(st.session_state.df_if.head(5), use_container_width=True)
        with col2:
            st.markdown("**Novelty Train CSV — First 5 rows**")
            st.dataframe(st.session_state.df_nv.head(5), use_container_width=True)

    # Column selection - EXACTLY like Gradio
    if st.session_state.numeric_if and st.session_state.numeric_nv:
        col1, col2 = st.columns(2)
        with col1:
            selected_cols_if = st.multiselect(
                "Select numeric columns for Anomaly (IF)",
                st.session_state.numeric_if,
                default=st.session_state.numeric_if[:min(6, len(st.session_state.numeric_if))]
            )
        with col2:
            selected_cols_nv = st.multiselect(
                "Select numeric columns for Novelty (AE)",
                st.session_state.numeric_nv,
                default=st.session_state.numeric_nv[:min(6, len(st.session_state.numeric_nv))]
            )

    # =========================== Anomaly Detection Section - EXACTLY like Gradio =========================
    st.markdown("### 2) Anomaly Detection — Isolation Forest")

    # Controls row - EXACTLY like Gradio
    col1, col2 = st.columns([3, 1])

    with col1:
        contam = st.slider(
            "Contamination (expected anomaly %)", 
            min_value=0.001, 
            max_value=0.20, 
            value=0.02, 
            step=0.001
        )

    with col2:
        run_if = st.button("Run Isolation Forest", type="primary")

    # Status and Visualizations - EXACTLY like Gradio layout
    if run_if:
        if st.session_state.df_if is not None and st.session_state.numeric_if:
            if selected_cols_if:
                with st.spinner("Running Isolation Forest..."):
                    fig1, fig2, top_view, anomalies, status = run_isolation_forest(contam, selected_cols_if)
                    
                    if fig1 is not None:
                        # Store results for persistence
                        st.session_state.if_results = {
                            'fig1': fig1,
                            'fig2': fig2,
                            'top_view': top_view,
                            'anomalies': anomalies,
                            'status': status
                        }
                        st.success(status)
            else:
                st.error("Please select at least one numeric column for IF")

    # Display persistent IF results
    if st.session_state.if_results is not None:
        st.success(st.session_state.if_results['status'])
        
        # Visualizations - EXACTLY like Gradio layout with proper headings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PCA Scatter (colored by IF score)**")
            st.pyplot(st.session_state.if_results['fig1'])
        with col2:
            st.markdown("**IF Score Histogram**")
            st.pyplot(st.session_state.if_results['fig2'])
        
        # Results - EXACTLY like Gradio
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top anomalies (by IF score)**")
            st.dataframe(st.session_state.if_results['top_view'], use_container_width=True)
        with col2:
            st.markdown("**Download anomalies_if.csv**")
            csv_data = st.session_state.if_results['anomalies'].to_csv(index=False)
            st.download_button(
                label="Download anomalies_if.csv",
                data=csv_data,
                file_name="anomalies_if.csv",
                mime="text/csv"
            )

    # =========================== Novelty Detection Section - EXACTLY like Gradio =========================
    st.markdown("### 3) Novelty Detection — Autoencoder (tight manifold via regularization)")

    # Controls row - EXACTLY like Gradio
    col1, col2 = st.columns([3, 1])

    with col1:
        epochs = st.slider(
            "Training Epochs (AE; PCA fallback if no torch)", 
            min_value=50, 
            max_value=300, 
            value=200, 
            step=1
        )

    with col2:
        train_ae = st.button("Train AE & Derive Training Threshold", type="primary")

    # Status and Visualizations - EXACTLY like Gradio layout
    if train_ae:
        if st.session_state.df_nv is not None and st.session_state.numeric_nv:
            if selected_cols_nv:
                with st.spinner("Training Autoencoder..."):
                    trainer_msg, threshold, hist_img = train_autoencoder(
                        epochs=epochs,
                        selected_cols=selected_cols_nv
                    )
                    
                    if trainer_msg is not None:
                        # Store results for persistence
                        st.session_state.ae_results = {
                            'trainer_msg': trainer_msg,
                            'threshold': threshold,
                            'hist_img': hist_img
                        }
                        st.success(trainer_msg)
            else:
                st.error("Please select at least one numeric column for Novelty training")

    # Display persistent AE results
    if st.session_state.ae_results is not None:
        st.success(st.session_state.ae_results['trainer_msg'])
        
        # Visualizations - EXACTLY like Gradio layout with proper headings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Error Histogram + Training Threshold**")
            st.pyplot(st.session_state.ae_results['hist_img'])
        with col2:
            st.markdown("**Training-derived Threshold (AE-only)**")
            st.text(f"{st.session_state.ae_results['threshold']:.8f}")

    # =========================== Test Data Scoring - EXACTLY like Gradio =========================
    st.markdown("#### Score Unseen/Test CSV")
    test_file = st.file_uploader(
        "Upload Test/Unseen CSV", 
        type=["csv"], 
        help="Upload CSV with test data to score for novelty"
    )

    score_test = st.button("Score Test Data", type="primary")

    # Status and Visualizations - EXACTLY like Gradio layout
    if score_test:
        if test_file is not None and st.session_state.numeric_nv:
            try:
                test_df = pd.read_csv(test_file)
                
                with st.spinner("Scoring test data..."):
                    fig_h, fig_p, scored_data, view, status = score_unseen_novelty(
                        test_df, st.session_state.numeric_nv
                    )
                    
                    if fig_h is not None:
                        # Store results for persistence
                        st.session_state.test_results = {
                            'fig_h': fig_h,
                            'fig_p': fig_p,
                            'scored_data': scored_data,
                            'view': view,
                            'status': status
                        }
                        st.success(status)
            except Exception as e:
                st.error(f"Error processing test file: {str(e)}")
        else:
            st.error("Please upload test file and train the novelty model first")

    # Display persistent Test results
    if st.session_state.test_results is not None:
        st.success(st.session_state.test_results['status'])
        
        # Visualizations - EXACTLY like Gradio layout with proper headings
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Test Reconstruction Error Histogram**")
            st.pyplot(st.session_state.test_results['fig_h'])
        with col2:
            st.markdown("**PCA Projection (Novelty in Red)**")
            st.pyplot(st.session_state.test_results['fig_p'])
        
        # Results - EXACTLY like Gradio
        st.markdown("**Scored Test Rows (error & novelty)**")
        st.dataframe(st.session_state.test_results['scored_data'], use_container_width=True)
        
        # Download - EXACTLY like Gradio
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Download novelty_scores.csv**")
            csv_data = st.session_state.test_results['view'].to_csv(index=False)
            st.download_button(
                label="Download novelty_scores.csv",
                data=csv_data,
                file_name="novelty_scores.csv",
                mime="text/csv"
            )
        with col2:
            st.markdown("**Novelty Status**")
            st.text(st.session_state.test_results['status'])

# Run the interface if this file is executed directly
if __name__ == "__main__":
    render_ml_backend()