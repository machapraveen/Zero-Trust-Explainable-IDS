"""
Explainable Deep Learning Framework for Zero-Trust Intrusion Detection
Using Multi-Level Network Traffic Analytics
======================================================================
Implements the end-to-end 5-stage architecture:
  Stage 1: Multi-Level Network Traffic Input (Packet L1, Flow L2, Session/Zero-Trust/IoT L3)
  Stage 2: Leak-Free Preprocessing & Feature Extraction (Imputer + StandardScaler)
  Stage 3: 10-Model Deep Learning & ML Ensemble Classification
  Stage 4: SHAP Explainability & Multi-Level Attribution Layer
  Stage 5: Zero-Trust Decision Engine (Continuous Trust Scoring & Adaptive Response)
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1: Multi-Level Network Traffic Schema (24 Features across 3 Levels)
# ---------------------------------------------------------------------------

PACKET_LEVEL_FEATURES = [
    "pkt_size_mean",
    "pkt_size_std",
    "syn_flag_ratio",
    "ack_flag_ratio",
    "rst_flag_ratio",
    "psh_flag_ratio",
    "header_overhead_ratio",
    "payload_entropy",
]

FLOW_LEVEL_FEATURES = [
    "flow_duration",
    "bytes_sent",
    "bytes_recv",
    "packets_sent",
    "packets_recv",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "inter_arrival_ms",
]

SESSION_ZEROTRUST_FEATURES = [
    "port_scan_dispersion",
    "failed_auth_rate",
    "session_burst_score",
    "outbound_asymmetry",
    "lateral_movement_score",
    "device_posture_risk",
    "identity_privilege_risk",
    "iot_edge_telemetry_dev",
]

MULTILEVEL_FEATURES: List[str] = (
    PACKET_LEVEL_FEATURES + FLOW_LEVEL_FEATURES + SESSION_ZEROTRUST_FEATURES
)

FEATURE_LEVEL_MAP: Dict[str, str] = {
    **{f: "L1_Packet" for f in PACKET_LEVEL_FEATURES},
    **{f: "L2_Flow" for f in FLOW_LEVEL_FEATURES},
    **{f: "L3_Session_ZeroTrust" for f in SESSION_ZEROTRUST_FEATURES},
}

ATTACK_CLASSES: Dict[int, str] = {
    0: "Normal",
    1: "DDoS",
    2: "Port Scan",
    3: "Brute Force",
    4: "SQL Injection",
    5: "Malware C2",
    6: "Data Exfiltration",
    7: "Lateral Movement",
    8: "Zero-Day / IoT Botnet",
}

MITRE_MAP: Dict[str, Dict[str, str]] = {
    "Normal": {"id": "NONE", "tactic": "Authorized Baseline"},
    "DDoS": {"id": "T1498", "tactic": "Impact - Network Denial of Service"},
    "Port Scan": {"id": "T1046", "tactic": "Discovery - Network Service Discovery"},
    "Brute Force": {"id": "T1110", "tactic": "Credential Access - Brute Force"},
    "SQL Injection": {"id": "T1190", "tactic": "Initial Access - Exploit Public-Facing App"},
    "Malware C2": {"id": "T1071", "tactic": "Command and Control - Application Layer Protocol"},
    "Data Exfiltration": {"id": "T1048", "tactic": "Exfiltration - Exfiltration Over Alternative Protocol"},
    "Lateral Movement": {"id": "T1021", "tactic": "Lateral Movement - Remote Services"},
    "Zero-Day / IoT Botnet": {"id": "T1203", "tactic": "Execution - Zero-Day / Edge IoT Exploitation"},
}


# ---------------------------------------------------------------------------
# Stage 3: PyTorch Deep Learning Architectures (Models 1-6 of 10-Model Suite)
# ---------------------------------------------------------------------------

class MultiScaleCNN1D(nn.Module):
    """Model 1: 1D-CNN Spatial Multi-Level Feature Detector."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1)))


class BiLSTMAttentionNet(nn.Module):
    """Model 2: Bidirectional LSTM with Attention across Traffic Levels."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        # Reshape 24 features -> 3 multi-level steps x 8 features per level
        self.step_dim = 8
        self.seq_len = max(1, in_features // self.step_dim)
        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=48,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.attn = nn.Linear(96, 1)
        self.fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        seq = x[:, : self.seq_len * self.step_dim].view(b, self.seq_len, self.step_dim)
        out, _ = self.lstm(seq)
        weights = torch.softmax(self.attn(out), dim=1)
        ctx = torch.sum(weights * out, dim=1)
        return self.fc(ctx)


class TransformerTrafficNet(nn.Module):
    """Model 3: Self-Attention Transformer for Cross-Level Traffic Analytics."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.seq_len = 3
        self.step_dim = in_features // 3
        self.embed = nn.Linear(self.step_dim, 64)
        self.pos = nn.Parameter(torch.randn(1, self.seq_len, 64) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.embed(x.view(b, self.seq_len, self.step_dim)) + self.pos
        encoded = self.encoder(tokens)
        return self.fc(encoded.mean(dim=1))


class RelationalGNNNet(nn.Module):
    """Model 4: Graph Neural Message-Passing Network over Multi-Level Traffic Nodes."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_nodes = 3  # L1 Packet, L2 Flow, L3 Session/Zero-Trust
        self.node_dim = in_features // 3
        self.node_proj = nn.Linear(self.node_dim, 48)
        self.adj_logits = nn.Parameter(torch.ones(self.num_nodes, self.num_nodes))
        self.msg1 = nn.Linear(48, 48)
        self.msg2 = nn.Linear(48, 48)
        self.readout = nn.Sequential(
            nn.Linear(48 * self.num_nodes, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = torch.relu(self.node_proj(x.view(b, self.num_nodes, self.node_dim)))
        adj = torch.softmax(self.adj_logits, dim=-1)
        h = torch.relu(h + torch.matmul(adj, self.msg1(h)))
        h = torch.relu(h + torch.matmul(adj, self.msg2(h)))
        return self.readout(h.reshape(b, -1))


class DeepResidualMLP(nn.Module):
    """Model 5: Deep Residual Neural Classifier."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_features, 96), nn.BatchNorm1d(96), nn.GELU())
        self.block = nn.Sequential(
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
        )
        self.out = nn.Sequential(nn.GELU(), nn.Linear(96, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.block(h)
        return self.out(h)


class ZeroDayAutoencoder(nn.Module):
    """Model 6: Deep Unsupervised Autoencoder for Zero-Day Anomaly Reconstruction."""

    def __init__(self, in_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, 12),
            nn.GELU(),
            nn.Linear(12, 6),
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.GELU(),
            nn.Linear(12, 32),
            nn.GELU(),
            nn.Linear(32, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Synthetic Multi-Level Dataset Generator (Realistic Zero-Trust Telemetry)
# ---------------------------------------------------------------------------

def generate_multilevel_dataset(n_samples: int = 4500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic multi-level network traffic dataset across Packet (L1),
    Flow (L2), and Session/Zero-Trust/Edge-IoT (L3) layers.
    """
    rng = np.random.default_rng(seed)
    class_probs = [0.42, 0.10, 0.08, 0.08, 0.07, 0.06, 0.06, 0.06, 0.07]
    labels = rng.choice(len(ATTACK_CLASSES), size=n_samples, p=class_probs)
    rows = []

    for y in labels:
        # Baseline Normal (Class 0)
        pkt_mean = rng.normal(520, 140)
        pkt_std = rng.normal(130, 40)
        syn_r = rng.beta(1.5, 18)
        ack_r = rng.beta(12, 3)
        rst_r = rng.beta(1.2, 25)
        psh_r = rng.beta(3, 10)
        hdr_r = rng.uniform(0.04, 0.14)
        entropy = rng.normal(4.8, 0.5)

        dur = rng.exponential(2.5) + 0.05
        b_sent = rng.lognormal(7.8, 0.9)
        b_recv = rng.lognormal(8.5, 1.0)
        p_sent = max(2.0, b_sent / max(120.0, pkt_mean))
        p_recv = max(2.0, b_recv / max(140.0, pkt_mean))
        iat_ms = rng.uniform(15, 140)

        port_disp = rng.beta(1.2, 25)
        fail_auth = rng.beta(1.0, 40)
        burst = rng.beta(2, 12)
        lat_score = rng.beta(1.2, 20)
        dev_risk = rng.beta(1.5, 14)
        priv_risk = rng.beta(1.3, 16)
        iot_dev = rng.beta(1.4, 18)

        if y == 1:  # DDoS
            pkt_mean = rng.normal(96, 25)
            syn_r = rng.uniform(0.75, 0.99)
            ack_r = rng.uniform(0.01, 0.15)
            dur = rng.uniform(0.02, 0.45)
            b_sent = rng.uniform(150_000, 2_500_000)
            b_recv = rng.uniform(100, 3_000)
            p_sent = rng.uniform(2_500, 35_000)
            p_recv = rng.uniform(5, 80)
            iat_ms = rng.uniform(0.05, 1.8)
            burst = rng.uniform(0.82, 0.99)
            hdr_r = rng.uniform(0.45, 0.85)
        elif y == 2:  # Port Scan
            pkt_mean = rng.normal(64, 12)
            syn_r = rng.uniform(0.60, 0.95)
            rst_r = rng.uniform(0.45, 0.90)
            dur = rng.uniform(0.01, 0.30)
            b_sent = rng.uniform(120, 2_400)
            b_recv = rng.uniform(60, 900)
            p_sent = rng.uniform(15, 180)
            iat_ms = rng.uniform(0.5, 8.0)
            port_disp = rng.uniform(0.75, 0.99)
            burst = rng.uniform(0.55, 0.88)
        elif y == 3:  # Brute Force
            pkt_mean = rng.normal(190, 35)
            psh_r = rng.uniform(0.45, 0.82)
            dur = rng.uniform(0.4, 6.0)
            b_sent = rng.uniform(8_000, 65_000)
            b_recv = rng.uniform(4_000, 30_000)
            p_sent = rng.uniform(45, 420)
            fail_auth = rng.uniform(0.68, 0.99)
            priv_risk = rng.uniform(0.55, 0.92)
            burst = rng.uniform(0.50, 0.85)
        elif y == 4:  # SQL Injection
            pkt_mean = rng.normal(880, 180)
            pkt_std = rng.normal(390, 70)
            psh_r = rng.uniform(0.55, 0.92)
            entropy = rng.uniform(5.9, 7.1)
            b_sent = rng.uniform(4_500, 48_000)
            b_recv = rng.uniform(25_000, 420_000)
            priv_risk = rng.uniform(0.62, 0.96)
        elif y == 5:  # Malware C2
            pkt_mean = rng.normal(260, 40)
            entropy = rng.uniform(7.1, 7.95)
            dur = rng.uniform(25.0, 600.0)
            b_sent = rng.uniform(2_000, 35_000)
            b_recv = rng.uniform(2_000, 45_000)
            iat_ms = rng.uniform(450, 2500)
            dev_risk = rng.uniform(0.68, 0.98)
            iot_dev = rng.uniform(0.45, 0.85)
        elif y == 6:  # Data Exfiltration
            pkt_mean = rng.normal(1380, 80)
            entropy = rng.uniform(7.2, 7.98)
            dur = rng.uniform(8.0, 180.0)
            b_sent = rng.uniform(2_000_000, 45_000_000)
            b_recv = rng.uniform(2_000, 60_000)
            p_sent = b_sent / 1400.0
            priv_risk = rng.uniform(0.65, 0.97)
            dev_risk = rng.uniform(0.45, 0.88)
        elif y == 7:  # Lateral Movement
            pkt_mean = rng.normal(420, 90)
            psh_r = rng.uniform(0.42, 0.78)
            b_sent = rng.uniform(15_000, 180_000)
            b_recv = rng.uniform(12_000, 140_000)
            lat_score = rng.uniform(0.72, 0.99)
            priv_risk = rng.uniform(0.70, 0.98)
            fail_auth = rng.uniform(0.25, 0.65)
        elif y == 8:  # Zero-Day / IoT Botnet
            pkt_mean = rng.normal(310, 150)
            pkt_std = rng.normal(280, 80)
            entropy = rng.uniform(6.8, 7.95)
            syn_r = rng.uniform(0.35, 0.82)
            rst_r = rng.uniform(0.20, 0.65)
            b_sent = rng.uniform(45_000, 900_000)
            iat_ms = rng.uniform(1.2, 18.0)
            dev_risk = rng.uniform(0.75, 0.99)
            iot_dev = rng.uniform(0.78, 0.99)
            burst = rng.uniform(0.65, 0.96)

        dur = max(0.005, float(dur))
        b_sent = max(40.0, float(b_sent))
        b_recv = max(0.0, float(b_recv))
        p_sent = max(1.0, float(p_sent))
        p_recv = max(0.0, float(p_recv))
        bps = (b_sent + b_recv) / dur
        pps = (p_sent + p_recv) / dur
        out_asym = b_sent / max(1.0, b_sent + b_recv)

        rows.append(
            {
                "pkt_size_mean": max(40.0, pkt_mean),
                "pkt_size_std": max(1.0, pkt_std),
                "syn_flag_ratio": np.clip(syn_r, 0.0, 1.0),
                "ack_flag_ratio": np.clip(ack_r, 0.0, 1.0),
                "rst_flag_ratio": np.clip(rst_r, 0.0, 1.0),
                "psh_flag_ratio": np.clip(psh_r, 0.0, 1.0),
                "header_overhead_ratio": np.clip(hdr_r, 0.01, 0.95),
                "payload_entropy": np.clip(entropy, 1.0, 8.0),
                "flow_duration": dur,
                "bytes_sent": b_sent,
                "bytes_recv": b_recv,
                "packets_sent": p_sent,
                "packets_recv": p_recv,
                "flow_bytes_per_sec": bps,
                "flow_packets_per_sec": pps,
                "inter_arrival_ms": max(0.01, iat_ms),
                "port_scan_dispersion": np.clip(port_disp, 0.0, 1.0),
                "failed_auth_rate": np.clip(fail_auth, 0.0, 1.0),
                "session_burst_score": np.clip(burst, 0.0, 1.0),
                "outbound_asymmetry": np.clip(out_asym, 0.0, 1.0),
                "lateral_movement_score": np.clip(lat_score, 0.0, 1.0),
                "device_posture_risk": np.clip(dev_risk, 0.0, 1.0),
                "identity_privilege_risk": np.clip(priv_risk, 0.0, 1.0),
                "iot_edge_telemetry_dev": np.clip(iot_dev, 0.0, 1.0),
                "attack_class": int(y),
                "attack_name": ATTACK_CLASSES[int(y)],
                "is_attack": int(y != 0),
            }
        )

    return pd.DataFrame(rows)


def extract_multilevel_features(flow: Dict[str, Any]) -> Dict[str, float]:
    """
    Stage 1 & 2 Helper: Deterministically map raw network flow telemetry into
    the 24-feature Multi-Level Network Traffic representation (Packet + Flow + Zero-Trust/Session).
    """
    # If all 24 features are already explicitly supplied, use them directly
    if all(k in flow for k in MULTILEVEL_FEATURES):
        return {k: float(flow[k]) for k in MULTILEVEL_FEATURES}

    dur = max(0.005, float(flow.get("flow_duration", flow.get("duration", 1.0)) or 1.0))
    b_sent = max(0.0, float(flow.get("bytes_sent", flow.get("src_bytes", 512)) or 0.0))
    b_recv = max(0.0, float(flow.get("bytes_recv", flow.get("dst_bytes", 1024)) or 0.0))
    total_bytes = b_sent + b_recv

    p_sent = float(flow.get("packets_sent", 0) or max(1.0, round(b_sent / 520.0)))
    p_recv = float(flow.get("packets_recv", 0) or max(1.0, round(b_recv / 680.0)))
    total_pkts = max(1.0, p_sent + p_recv)

    dst_port = int(flow.get("dst_port", flow.get("destination_port", 443)) or 443)
    protocol = str(flow.get("protocol", "TCP") or "TCP").upper()
    src_ip = str(flow.get("src_ip", flow.get("source_ip", "192.168.1.100")) or "")
    dst_ip = str(flow.get("dst_ip", flow.get("destination_ip", "10.0.0.10")) or "")

    bps = float(flow.get("flow_bytes_per_sec", total_bytes / dur))
    pps = float(flow.get("flow_packets_per_sec", total_pkts / dur))
    pkt_mean = float(flow.get("pkt_size_mean", max(40.0, total_bytes / total_pkts)))
    pkt_std = float(flow.get("pkt_size_std", max(10.0, pkt_mean * 0.28)))
    iat_ms = float(flow.get("inter_arrival_ms", min(5000.0, (dur * 1000.0) / total_pkts)))
    out_asym = float(flow.get("outbound_asymmetry", b_sent / max(1.0, total_bytes)))

    # Infer Packet-Level (L1) indicators when raw packet capture counters aren't explicitly supplied
    is_flood_like = pps > 1500 or (b_sent > 120_000 and dur < 0.6 and b_recv < 5000)
    is_scan_like = total_bytes < 3000 and dur < 0.35 and dst_port in (21, 22, 23, 25, 445, 1433, 3306, 3389, 5432, 8080)
    is_c2_port = dst_port in (4444, 5555, 6666, 31337, 8443, 9001)
    is_internal_src = src_ip.startswith(("10.", "192.168.", "172.16."))
    is_internal_dst = dst_ip.startswith(("10.", "192.168.", "172.16."))
    is_east_west_priv = is_internal_src and is_internal_dst and dst_port in (22, 135, 139, 445, 3389, 5985)
    is_exfil_like = b_sent > 1_500_000 and out_asym > 0.85

    syn_r = float(flow.get("syn_flag_ratio", 0.88 if is_flood_like else (0.76 if is_scan_like else 0.08)))
    ack_r = float(flow.get("ack_flag_ratio", 0.08 if is_flood_like else 0.78))
    rst_r = float(flow.get("rst_flag_ratio", 0.68 if is_scan_like else 0.04))
    psh_r = float(flow.get("psh_flag_ratio", 0.68 if dst_port in (22, 80, 443, 3306, 5432) and b_sent > 3000 else 0.22))
    hdr_r = float(flow.get("header_overhead_ratio", min(0.85, max(0.04, 40.0 / max(48.0, pkt_mean)))))
    entropy = float(
        flow.get(
            "payload_entropy",
            7.55 if (is_exfil_like or is_c2_port) else (6.35 if protocol in ("HTTPS", "SSH") else 4.85),
        )
    )

    # Infer Session & Zero-Trust (L3) indicators
    port_disp = float(flow.get("port_scan_dispersion", 0.86 if is_scan_like else 0.05))
    fail_auth = float(
        flow.get(
            "failed_auth_rate",
            0.82 if (dst_port in (21, 22, 23, 3389) and dur < 3.0 and b_sent > 4000) else 0.03,
        )
    )
    burst = float(flow.get("session_burst_score", min(0.99, pps / 2500.0) if pps > 120 else 0.14))
    lat_score = float(flow.get("lateral_movement_score", 0.88 if is_east_west_priv else 0.06))
    dev_risk = float(
        flow.get(
            "device_posture_risk",
            0.85 if is_c2_port else (0.55 if not is_internal_src else 0.12),
        )
    )
    priv_risk = float(
        flow.get(
            "identity_privilege_risk",
            0.82 if (is_east_west_priv or is_exfil_like or dst_port in (3306, 5432, 1433)) else 0.10,
        )
    )
    iot_dev = float(flow.get("iot_edge_telemetry_dev", 0.84 if protocol in ("MQTT", "COAP") or is_c2_port else 0.08))

    return {
        "pkt_size_mean": pkt_mean,
        "pkt_size_std": pkt_std,
        "syn_flag_ratio": np.clip(syn_r, 0.0, 1.0),
        "ack_flag_ratio": np.clip(ack_r, 0.0, 1.0),
        "rst_flag_ratio": np.clip(rst_r, 0.0, 1.0),
        "psh_flag_ratio": np.clip(psh_r, 0.0, 1.0),
        "header_overhead_ratio": np.clip(hdr_r, 0.01, 0.95),
        "payload_entropy": np.clip(entropy, 1.0, 8.0),
        "flow_duration": dur,
        "bytes_sent": b_sent,
        "bytes_recv": b_recv,
        "packets_sent": p_sent,
        "packets_recv": p_recv,
        "flow_bytes_per_sec": bps,
        "flow_packets_per_sec": pps,
        "inter_arrival_ms": max(0.01, iat_ms),
        "port_scan_dispersion": np.clip(port_disp, 0.0, 1.0),
        "failed_auth_rate": np.clip(fail_auth, 0.0, 1.0),
        "session_burst_score": np.clip(burst, 0.0, 1.0),
        "outbound_asymmetry": np.clip(out_asym, 0.0, 1.0),
        "lateral_movement_score": np.clip(lat_score, 0.0, 1.0),
        "device_posture_risk": np.clip(dev_risk, 0.0, 1.0),
        "identity_privilege_risk": np.clip(priv_risk, 0.0, 1.0),
        "iot_edge_telemetry_dev": np.clip(iot_dev, 0.0, 1.0),
    }


# ---------------------------------------------------------------------------
# Main Unified 5-Stage Zero-Trust Explainable Deep Learning Engine
# ---------------------------------------------------------------------------

class ZeroTrustNIDSEngine:
    """
    Production 10-Model Deep Learning & ML Ensemble + SHAP Explainability +
    Zero-Trust Continuous Policy Decision Engine.
    """

    MODEL_CATALOG = [
        {
            "id": "deep_cnn",
            "name": "1D-CNN Deep Traffic Net",
            "family": "Deep Learning (PyTorch)",
            "icon": "🧬",
            "color": "#ec4899",
            "weight": 0.13,
            "role": "Spatial Packet/Flow Pattern Extraction",
        },
        {
            "id": "bilstm_attn",
            "name": "Bi-LSTM Temporal Attention",
            "family": "Deep Learning (PyTorch)",
            "icon": "🧠",
            "color": "#06b6d4",
            "weight": 0.13,
            "role": "Sequential Multi-Level Flow Dynamics",
        },
        {
            "id": "transformer_net",
            "name": "Transformer Self-Attention",
            "family": "Deep Learning (PyTorch)",
            "icon": "⚡",
            "color": "#8b5cf6",
            "weight": 0.14,
            "role": "Cross-Level Packet-Flow-ZeroTrust Attention",
        },
        {
            "id": "gnn_topology",
            "name": "Graph Neural Network (GNN)",
            "family": "Deep Learning (PyTorch)",
            "icon": "🔗",
            "color": "#3b82f6",
            "weight": 0.11,
            "role": "East-West Topology & Lateral Movement",
        },
        {
            "id": "deep_resmlp",
            "name": "Deep Residual MLP",
            "family": "Deep Learning (PyTorch)",
            "icon": "🔮",
            "color": "#a855f7",
            "weight": 0.09,
            "role": "Non-Linear High-Dimensional Representation",
        },
        {
            "id": "deep_autoencoder",
            "name": "Deep Reconstruction Autoencoder",
            "family": "Unsupervised Deep Learning",
            "icon": "🎯",
            "color": "#f97316",
            "weight": 0.08,
            "role": "Zero-Day & Novel Anomaly Detection",
        },
        {
            "id": "xgboost",
            "name": "XGBoost Gradient Booster",
            "family": "Gradient Boosted Trees",
            "icon": "🚀",
            "color": "#ef4444",
            "weight": 0.14,
            "role": "High-Precision Known Threat Classification",
        },
        {
            "id": "random_forest",
            "name": "Random Forest Classifier",
            "family": "Ensemble Trees",
            "icon": "🌲",
            "color": "#10b981",
            "weight": 0.08,
            "role": "Low-Variance Flow Classification",
        },
        {
            "id": "extra_trees",
            "name": "Extra-Trees Randomized Ensemble",
            "family": "Ensemble Trees",
            "icon": "🛡️",
            "color": "#14b8a6",
            "weight": 0.06,
            "role": "Volumetric & Noisy Traffic Resilience",
        },
        {
            "id": "isolation_forest",
            "name": "Isolation Forest Detector",
            "family": "Unsupervised Outlier ML",
            "icon": "🔍",
            "color": "#eab308",
            "weight": 0.04,
            "role": "Outlier Isolation & Edge IoT Drift",
        },
    ]

    def __init__(self, model_dir: Optional[str] = None, data_dir: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.model_dir = Path(model_dir) if model_dir else (base_dir / "models")
        self.data_dir = Path(data_dir) if data_dir else (base_dir / "data")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        self.feature_names = MULTILEVEL_FEATURES
        self.num_classes = len(ATTACK_CLASSES)

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None

        # 10 Models
        self.dl_models: Dict[str, nn.Module] = {}
        self.autoencoder: Optional[ZeroDayAutoencoder] = None
        self.ae_threshold: float = 1.0
        self.ml_models: Dict[str, Any] = {}
        self.iso_forest: Optional[IsolationForest] = None
        self.iso_bounds: Tuple[float, float] = (-0.2, 0.2)

        # SHAP Explainer (Stage 4)
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.shap_background: Optional[np.ndarray] = None
        self.xgb_binary_explainer_model: Optional[xgb.XGBClassifier] = None

        self.metrics: Dict[str, Any] = {}
        self.initialized = False
        self.predictions_count = 0
        self.initialized_at: Optional[str] = None
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Initialization / Persistence / Leak-Free Training
    # -----------------------------------------------------------------------

    def ensure_initialized(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Load saved 10-model bundle + SHAP explainer from disk, or train and persist if missing."""
        with self._lock:
            if self.initialized and not force_retrain:
                return self.metrics

            bundle_path = self.model_dir / "zero_trust_10model_bundle.pkl"
            torch_path = self.model_dir / "zero_trust_dl_weights.pt"

            if not force_retrain and bundle_path.exists() and torch_path.exists():
                try:
                    self._load_artifacts(bundle_path, torch_path)
                    self.initialized = True
                    logger.info("Loaded 10-Model Zero-Trust Ensemble + SHAP Explainer from disk.")
                    return self.metrics
                except Exception as exc:
                    logger.warning(f"Artifact load failed ({exc}); retraining clean 10-model suite...")

            return self.train_and_save()

    def _train_torch_classifier(
        self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 18, batch_size: int = 128
    ) -> nn.Module:
        model.to(self.device)
        model.train()
        ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                logits = model(bx)
                loss = crit(logits, by)
                loss.backward()
                opt.step()
        model.eval()
        return model

    def train_and_save(self) -> Dict[str, Any]:
        """
        Train all 10 Deep Learning + ML models and fit SHAP TreeExplainer following
        strict leak-free ML practices (train/val/test split BEFORE imputation/scaling).
        """
        t0 = time.time()
        torch.manual_seed(42)
        np.random.seed(42)

        csv_path = self.data_dir / "processed" / "zero_trust_multilevel_traffic.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = generate_multilevel_dataset(n_samples=4500, seed=42)
            df.to_csv(csv_path, index=False)

        # Clean infinities -> NaN before split
        X_raw = df[self.feature_names].replace([np.inf, -np.inf], np.nan).values
        y_multi = df["attack_class"].values.astype(int)

        # Strict Featurization Ordering: Split BEFORE fitting imputer & scaler!
        X_temp, X_test_raw, y_temp, y_test = train_test_split(
            X_raw, y_multi, test_size=0.15, random_state=42, stratify=y_multi
        )
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
        )

        self.imputer = SimpleImputer(strategy="median")
        X_train_imp = self.imputer.fit_transform(X_train_raw)
        X_val_imp = self.imputer.transform(X_val_raw)
        X_test_imp = self.imputer.transform(X_test_raw)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_imp)
        X_val = self.scaler.transform(X_val_imp)
        X_test = self.scaler.transform(X_test_imp)

        in_dim = len(self.feature_names)

        # 1-5: PyTorch Supervised Deep Learning Models
        dl_specs = {
            "deep_cnn": MultiScaleCNN1D(in_dim, self.num_classes),
            "bilstm_attn": BiLSTMAttentionNet(in_dim, self.num_classes),
            "transformer_net": TransformerTrafficNet(in_dim, self.num_classes),
            "gnn_topology": RelationalGNNNet(in_dim, self.num_classes),
            "deep_resmlp": DeepResidualMLP(in_dim, self.num_classes),
        }
        self.dl_models = {}
        for name, net in dl_specs.items():
            self.dl_models[name] = self._train_torch_classifier(net, X_train, y_train, epochs=18)

        # 6: Deep Unsupervised Autoencoder (trained strictly on benign X_train[y_train == 0])
        benign_train = X_train[y_train == 0]
        ae = ZeroDayAutoencoder(in_dim).to(self.device)
        ae.train()
        ae_loader = DataLoader(
            TensorDataset(torch.tensor(benign_train, dtype=torch.float32)),
            batch_size=128,
            shuffle=True,
        )
        ae_opt = optim.AdamW(ae.parameters(), lr=0.003, weight_decay=1e-4)
        mse = nn.MSELoss()
        for _ in range(22):
            for (bx,) in ae_loader:
                ae_opt.zero_grad()
                recon = ae(bx)
                loss = mse(recon, bx)
                loss.backward()
                ae_opt.step()
        ae.eval()
        with torch.no_grad():
            benign_val = torch.tensor(X_val[y_val == 0], dtype=torch.float32)
            val_err = torch.mean((ae(benign_val) - benign_val) ** 2, dim=1).numpy()
            self.ae_threshold = float(np.percentile(val_err, 95))
        self.autoencoder = ae

        # 7-9: Tree Ensemble Classifiers
        xgb_clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=self.num_classes,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        xgb_clf.fit(X_train, y_train)

        rf_clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=2)
        rf_clf.fit(X_train, y_train)

        et_clf = ExtraTreesClassifier(n_estimators=120, max_depth=14, random_state=42, n_jobs=2)
        et_clf.fit(X_train, y_train)

        self.ml_models = {
            "xgboost": xgb_clf,
            "random_forest": rf_clf,
            "extra_trees": et_clf,
        }

        # 10: Isolation Forest Unsupervised Outlier Detector (fit on benign training flows)
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=2)
        iso.fit(benign_train)
        iso_scores = -iso.decision_function(X_val)
        self.iso_bounds = (float(np.percentile(iso_scores, 5)), float(np.percentile(iso_scores, 95)))
        self.iso_forest = iso

        # Stage 4: Fit dedicated Binary Intrusion XGBoost + SHAP TreeExplainer for fast, exact SHAP values
        y_train_bin = (y_train != 0).astype(int)
        self.xgb_binary_explainer_model = xgb.XGBClassifier(
            n_estimators=90,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        self.xgb_binary_explainer_model.fit(X_train, y_train_bin)
        bg_idx = np.random.choice(len(X_train), size=min(120, len(X_train)), replace=False)
        self.shap_background = X_train[bg_idx]
        self.shap_explainer = shap.TreeExplainer(self.xgb_binary_explainer_model)

        # Evaluate individual models + 10-model ensemble on held-out Test Split
        y_test_bin = (y_test != 0).astype(int)
        per_model_metrics: Dict[str, Dict[str, float]] = {}

        for item in self.MODEL_CATALOG:
            mid = item["id"]
            atk_prob, cls_probs = self._single_model_probs_batch(mid, X_test)
            pred_bin = (atk_prob >= 0.5).astype(int)
            per_model_metrics[mid] = {
                "accuracy": round(float(accuracy_score(y_test_bin, pred_bin)), 4),
                "f1": round(float(f1_score(y_test_bin, pred_bin, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test_bin, atk_prob)), 4),
            }

        ens_atk_prob, ens_cls_probs = self._ensemble_probs_batch(X_test)
        ens_pred_bin = (ens_atk_prob >= 0.5).astype(int)
        ens_pred_cls = np.argmax(ens_cls_probs, axis=1)
        fp = int(np.sum((ens_pred_bin == 1) & (y_test_bin == 0)))
        tn = int(np.sum((ens_pred_bin == 0) & (y_test_bin == 0)))
        fpr = fp / max(1, fp + tn)

        self.initialized_at = datetime.utcnow().isoformat()
        self.metrics = {
            "framework": "Explainable Deep Learning Framework for Zero-Trust Intrusion Detection",
            "trained_at": self.initialized_at,
            "training_seconds": round(time.time() - t0, 2),
            "dataset_samples": int(len(df)),
            "n_features": in_dim,
            "n_models": 10,
            "shap_initialized": True,
            "ensemble": {
                "binary_accuracy": round(float(accuracy_score(y_test_bin, ens_pred_bin)), 4),
                "multiclass_accuracy": round(float(accuracy_score(y_test, ens_pred_cls)), 4),
                "precision": round(float(precision_score(y_test_bin, ens_pred_bin, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_bin, ens_pred_bin, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test_bin, ens_pred_bin, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test_bin, ens_atk_prob)), 4),
                "false_positive_rate": round(float(fpr), 4),
            },
            "per_model": per_model_metrics,
        }

        self._save_artifacts()
        self.initialized = True
        logger.info(
            f"10-Model Zero-Trust Ensemble trained in {self.metrics['training_seconds']}s | "
            f"Accuracy={self.metrics['ensemble']['binary_accuracy']:.2%}, "
            f"FPR={self.metrics['ensemble']['false_positive_rate']:.2%}"
        )
        return self.metrics

    def _save_artifacts(self) -> None:
        torch_state = {
            "dl_models": {k: m.state_dict() for k, m in self.dl_models.items()},
            "autoencoder": self.autoencoder.state_dict() if self.autoencoder else None,
            "ae_threshold": self.ae_threshold,
        }
        torch.save(torch_state, self.model_dir / "zero_trust_dl_weights.pt")

        bundle = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "ml_models": self.ml_models,
            "iso_forest": self.iso_forest,
            "iso_bounds": self.iso_bounds,
            "xgb_binary_explainer_model": self.xgb_binary_explainer_model,
            "shap_background": self.shap_background,
            "metrics": self.metrics,
            "initialized_at": self.initialized_at,
            "feature_names": self.feature_names,
        }
        with open(self.model_dir / "zero_trust_10model_bundle.pkl", "wb") as f:
            pickle.dump(bundle, f)

        # Also write legacy compatibility files expected by ModelPredictor / DetectionEngine
        with open(self.model_dir / "feature_config.json", "w") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "n_features": len(self.feature_names),
                    "saved_at": self.initialized_at,
                },
                f,
                indent=2,
            )
        with open(self.model_dir / "training_report.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _load_artifacts(self, bundle_path: Path, torch_path: Path) -> None:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.ml_models = bundle["ml_models"]
        self.iso_forest = bundle["iso_forest"]
        self.iso_bounds = bundle["iso_bounds"]
        self.xgb_binary_explainer_model = bundle["xgb_binary_explainer_model"]
        self.shap_background = bundle["shap_background"]
        self.metrics = bundle["metrics"]
        self.initialized_at = bundle.get("initialized_at") or datetime.utcnow().isoformat()

        in_dim = len(self.feature_names)
        torch_state = torch.load(torch_path, map_location=self.device, weights_only=False)
        dl_specs = {
            "deep_cnn": MultiScaleCNN1D(in_dim, self.num_classes),
            "bilstm_attn": BiLSTMAttentionNet(in_dim, self.num_classes),
            "transformer_net": TransformerTrafficNet(in_dim, self.num_classes),
            "gnn_topology": RelationalGNNNet(in_dim, self.num_classes),
            "deep_resmlp": DeepResidualMLP(in_dim, self.num_classes),
        }
        self.dl_models = {}
        for name, net in dl_specs.items():
            net.load_state_dict(torch_state["dl_models"][name])
            net.to(self.device).eval()
            self.dl_models[name] = net

        ae = ZeroDayAutoencoder(in_dim).to(self.device)
        ae.load_state_dict(torch_state["autoencoder"])
        ae.eval()
        self.autoencoder = ae
        self.ae_threshold = float(torch_state["ae_threshold"])

        self.shap_explainer = shap.TreeExplainer(self.xgb_binary_explainer_model)

    # -----------------------------------------------------------------------
    # Inference Helpers (All 10 Models)
    # -----------------------------------------------------------------------

    def _single_model_probs_batch(self, model_id: str, X_scaled: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if model_id in self.dl_models:
            net = self.dl_models[model_id]
            with torch.no_grad():
                logits = net(torch.tensor(X_scaled, dtype=torch.float32, device=self.device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return 1.0 - probs[:, 0], probs

        if model_id in self.ml_models:
            probs = self.ml_models[model_id].predict_proba(X_scaled)
            return 1.0 - probs[:, 0], probs

        if model_id == "deep_autoencoder" and self.autoencoder is not None:
            with torch.no_grad():
                xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
                err = torch.mean((self.autoencoder(xt) - xt) ** 2, dim=1).cpu().numpy()
            # Smooth logistic calibration around 95th-percentile benign threshold
            ratio = err / max(1e-6, self.ae_threshold)
            atk_prob = 1.0 / (1.0 + np.exp(-3.0 * (ratio - 1.05)))
            return np.clip(atk_prob, 0.01, 0.99), None

        if model_id == "isolation_forest" and self.iso_forest is not None:
            raw = -self.iso_forest.decision_function(X_scaled)
            lo, hi = self.iso_bounds
            norm = (raw - lo) / max(1e-6, (hi - lo))
            return np.clip(norm, 0.01, 0.99), None

        raise ValueError(f"Unknown model_id: {model_id}")

    def _ensemble_probs_batch(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X_scaled)
        weighted_atk = np.zeros(n, dtype=np.float64)
        weighted_cls = np.zeros((n, self.num_classes), dtype=np.float64)
        cls_weight_sum = 0.0
        total_weight = 0.0

        for item in self.MODEL_CATALOG:
            mid = item["id"]
            w = item["weight"]
            atk_p, cls_p = self._single_model_probs_batch(mid, X_scaled)
            weighted_atk += w * atk_p
            total_weight += w
            if cls_p is not None:
                weighted_cls += w * cls_p
                cls_weight_sum += w

        ens_atk = weighted_atk / max(1e-8, total_weight)
        ens_cls = weighted_cls / max(1e-8, cls_weight_sum)
        return ens_atk, ens_cls

    # -----------------------------------------------------------------------
    # Stage 4 (SHAP) + Stage 5 (Zero-Trust Policy Engine) Full Flow Analysis
    # -----------------------------------------------------------------------

    def analyze_flow(self, flow: Dict[str, Any], include_shap: bool = True) -> Dict[str, Any]:
        """
        Execute all 5 stages on a single network flow and return rich, transparent results.
        """
        self.ensure_initialized()
        t0 = time.perf_counter()

        # Stage 1 & 2: Multi-Level Feature Extraction + Leak-Free Scaling
        extracted = extract_multilevel_features(flow)
        x_raw = np.array([[extracted[f] for f in self.feature_names]], dtype=np.float64)
        x_imp = self.imputer.transform(x_raw)
        x_scaled = self.scaler.transform(x_imp)

        # Stage 3: 10-Model Deep Learning & ML Ensemble Vote
        model_votes: List[Dict[str, Any]] = []
        weighted_atk = 0.0
        weighted_cls = np.zeros(self.num_classes, dtype=np.float64)
        cls_w_sum = 0.0

        for item in self.MODEL_CATALOG:
            mid = item["id"]
            w = item["weight"]
            atk_p_arr, cls_p_arr = self._single_model_probs_batch(mid, x_scaled)
            atk_p = float(atk_p_arr[0])
            weighted_atk += w * atk_p
            pred_cls_name = "Anomaly" if atk_p >= 0.5 else "Normal"
            if cls_p_arr is not None:
                weighted_cls += w * cls_p_arr[0]
                cls_w_sum += w
                pred_cls_idx = int(np.argmax(cls_p_arr[0]))
                pred_cls_name = ATTACK_CLASSES[pred_cls_idx]

            model_votes.append(
                {
                    "id": mid,
                    "name": item["name"],
                    "family": item["family"],
                    "icon": item["icon"],
                    "weight": w,
                    "threat_probability": round(atk_p, 4),
                    "verdict": "THREAT" if atk_p >= 0.5 else "BENIGN",
                    "predicted_class": pred_cls_name,
                }
            )

        ens_threat_prob = float(np.clip(weighted_atk, 0.0, 1.0))
        ens_cls_probs = weighted_cls / max(1e-8, cls_w_sum)

        # Check Deep Autoencoder reconstruction anomaly for Zero-Day boost
        ae_prob = next(m["threat_probability"] for m in model_votes if m["id"] == "deep_autoencoder")
        is_threat = bool(ens_threat_prob >= 0.50 or ae_prob >= 0.85)

        if is_threat:
            # Pick highest-probability non-benign attack class
            attack_probs_only = ens_cls_probs.copy()
            attack_probs_only[0] = -1.0
            best_cls_idx = int(np.argmax(attack_probs_only))
            attack_type = ATTACK_CLASSES[best_cls_idx]
            confidence = round(float(max(ens_threat_prob, ens_cls_probs[best_cls_idx])), 4)
        else:
            attack_type = "Normal"
            confidence = round(float(1.0 - ens_threat_prob), 4)

        # Stage 4: Real SHAP Interpretability Layer
        shap_payload: Dict[str, Any] = {}
        if include_shap and self.shap_explainer is not None:
            raw_shap = self.shap_explainer.shap_values(x_scaled)
            if isinstance(raw_shap, list):
                sv = np.array(raw_shap[1] if len(raw_shap) > 1 else raw_shap[0])[0]
            else:
                arr = np.array(raw_shap)
                sv = arr[0, :, 1] if arr.ndim == 3 else arr[0]

            base_val = self.shap_explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
            else:
                base_val = float(base_val)

            # Normalize SHAP contributions to probability scale for intuitive SOC display
            abs_total = float(np.sum(np.abs(sv))) or 1.0
            scaled_sv = sv / abs_total * max(0.15, abs(ens_threat_prob - 0.25))
            shap_dict = {f: round(float(v), 4) for f, v in zip(self.feature_names, scaled_sv)}

            sorted_pairs = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
            top_pos = [
                {"feature": f, "level": FEATURE_LEVEL_MAP[f], "value": round(extracted[f], 4), "shap": v}
                for f, v in sorted_pairs
                if v > 0
            ][:5]
            top_neg = [
                {"feature": f, "level": FEATURE_LEVEL_MAP[f], "value": round(extracted[f], 4), "shap": v}
                for f, v in sorted_pairs
                if v < 0
            ][:4]

            # Multi-Level Attribution Breakdown (Packet vs Flow vs Session/Zero-Trust)
            l1_abs = sum(abs(shap_dict[f]) for f in PACKET_LEVEL_FEATURES)
            l2_abs = sum(abs(shap_dict[f]) for f in FLOW_LEVEL_FEATURES)
            l3_abs = sum(abs(shap_dict[f]) for f in SESSION_ZEROTRUST_FEATURES)
            lvl_sum = max(1e-8, l1_abs + l2_abs + l3_abs)
            level_attribution = {
                "Packet_Level_L1_pct": round(100.0 * l1_abs / lvl_sum, 1),
                "Flow_Level_L2_pct": round(100.0 * l2_abs / lvl_sum, 1),
                "Session_ZeroTrust_L3_pct": round(100.0 * l3_abs / lvl_sum, 1),
            }

            top_driver_txt = ", ".join(
                f"{item['feature']} ({'+' if item['shap'] > 0 else ''}{item['shap']:.3f})"
                for item in top_pos[:3]
            ) or "baseline flow profile"

            analyst_summary = (
                f"Classified as {attack_type} ({confidence * 100:.1f}% confidence) by 10-Model Deep Ensemble. "
                f"Primary risk drivers: {top_driver_txt}. "
                f"Traffic analytics attribution: L1 Packet {level_attribution['Packet_Level_L1_pct']}%, "
                f"L2 Flow {level_attribution['Flow_Level_L2_pct']}%, "
                f"L3 Zero-Trust/Session {level_attribution['Session_ZeroTrust_L3_pct']}%."
            )

            # Top 8 features dict for clean rendering in alert_detail.html
            top8_dict = dict(sorted_pairs[:8])

            shap_payload = {
                "top_features": top8_dict,
                "all_shap_values": shap_dict,
                "base_value": round(base_val, 4),
                "top_positive_contributors": top_pos,
                "top_negative_contributors": top_neg,
                "multilevel_attribution": level_attribution,
                "analyst_summary": analyst_summary,
            }

        # Stage 5: Zero-Trust Decision Engine ("Never Trust, Always Verify")
        zt_decision = self._evaluate_zero_trust_policy(
            flow=flow,
            extracted=extracted,
            is_threat=is_threat,
            attack_type=attack_type,
            threat_prob=ens_threat_prob,
            ae_anomaly=ae_prob,
        )

        severity = zt_decision["severity"]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.predictions_count += 1

        return {
            "is_threat": is_threat,
            "attack_type": attack_type,
            "severity": severity,
            "confidence": confidence,
            "threat_probability": round(ens_threat_prob, 4),
            "model_used": "10-Model Zero-Trust Deep Ensemble",
            "latency_ms": latency_ms,
            "source_ip": flow.get("src_ip", flow.get("source_ip", "0.0.0.0")),
            "destination_ip": flow.get("dst_ip", flow.get("destination_ip", "0.0.0.0")),
            "description": shap_payload.get(
                "analyst_summary",
                f"{attack_type} detected by 10-Model Deep Ensemble",
            ),
            "multilevel_features": {
                "L1_Packet": {k: round(extracted[k], 4) for k in PACKET_LEVEL_FEATURES},
                "L2_Flow": {k: round(extracted[k], 4) for k in FLOW_LEVEL_FEATURES},
                "L3_Session_ZeroTrust": {k: round(extracted[k], 4) for k in SESSION_ZEROTRUST_FEATURES},
            },
            "ensemble_votes": model_votes,
            "shap_explanation": shap_payload,
            "zero_trust_decision": zt_decision,
        }

    def _evaluate_zero_trust_policy(
        self,
        flow: Dict[str, Any],
        extracted: Dict[str, float],
        is_threat: bool,
        attack_type: str,
        threat_prob: float,
        ae_anomaly: float,
    ) -> Dict[str, Any]:
        """Stage 5: Zero-Trust Continuous Verification & Adaptive Response Engine."""
        posture_penalty = (
            0.35 * extracted["device_posture_risk"]
            + 0.35 * extracted["identity_privilege_risk"]
            + 0.30 * extracted["lateral_movement_score"]
        )
        combined_risk = float(np.clip(0.68 * threat_prob + 0.17 * ae_anomaly + 0.15 * posture_penalty, 0.0, 1.0))
        trust_score = round((1.0 - combined_risk) * 100.0, 1)

        src_ip = flow.get("src_ip", flow.get("source_ip", "unknown"))
        dst_ip = flow.get("dst_ip", flow.get("destination_ip", "unknown"))
        dst_port = flow.get("dst_port", flow.get("destination_port", "*"))

        if not is_threat and trust_score >= 78.0:
            action = "ALLOW_CONTINUOUS_VERIFY"
            severity = "info"
            enforcement = f"PERMIT session {src_ip} -> {dst_ip}:{dst_port} with continuous L1-L3 telemetry verification"
        elif not is_threat or trust_score >= 58.0:
            action = "STEP_UP_MFA_AND_INSPECT"
            severity = "low" if not is_threat else "medium"
            enforcement = f"REQUIRE step-up identity verification & deep packet inspection for {src_ip}"
        elif trust_score >= 38.0:
            action = "RATE_LIMIT_AND_DPI"
            severity = "medium"
            enforcement = f"THROTTLE flow rate for {src_ip} -> {dst_ip}:{dst_port} and mirror to sandbox"
        elif trust_score >= 20.0:
            action = "MICRO_SEGMENT_QUARANTINE"
            severity = "high"
            enforcement = f"QUARANTINE {src_ip} into restricted Zero-Trust VLAN; block east-west lateral ports"
        else:
            action = "ZERO_TRUST_BLOCK_AND_ISOLATE"
            severity = "critical"
            enforcement = f"REVOKE session token & BLOCK {src_ip} at perimeter + micro-segment firewall immediately"

        # Escalate critical attack classes regardless
        if is_threat and attack_type in ("DDoS", "SQL Injection", "Data Exfiltration", "Malware C2", "Zero-Day / IoT Botnet"):
            if threat_prob >= 0.78:
                severity = "critical"
                action = "ZERO_TRUST_BLOCK_AND_ISOLATE"
                enforcement = f"REVOKE session token & BLOCK {src_ip} at perimeter + micro-segment firewall immediately"
            elif severity in ("info", "low"):
                severity = "high"

        mitre = MITRE_MAP.get(attack_type, {"id": "T1071", "tactic": "Suspicious Network Activity"})
        return {
            "principle": "Never Trust, Always Verify",
            "entity_trust_score": trust_score,
            "combined_risk_score": round(combined_risk, 4),
            "policy_action": action,
            "severity": severity,
            "enforcement_rule": enforcement,
            "mitre_attack_id": mitre["id"],
            "mitre_tactic": mitre["tactic"],
            "edge_iot_adaptive_mode": bool(extracted["iot_edge_telemetry_dev"] > 0.5),
        }

    def get_status_summary(self) -> Dict[str, Any]:
        self.ensure_initialized()
        return {
            "initialized": self.initialized,
            "initialized_at": self.initialized_at,
            "n_models": len(self.MODEL_CATALOG),
            "shap_explainer_active": self.shap_explainer is not None,
            "zero_trust_engine_active": True,
            "n_multilevel_features": len(self.feature_names),
            "predictions_count": self.predictions_count,
            "ensemble_metrics": self.metrics.get("ensemble", {}),
            "per_model_metrics": self.metrics.get("per_model", {}),
            "models": self.MODEL_CATALOG,
        }


# Singleton Engine Instance
_ENGINE_INSTANCE: Optional[ZeroTrustNIDSEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_zero_trust_engine() -> ZeroTrustNIDSEngine:
    """Return process-wide initialized ZeroTrustNIDSEngine singleton."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        with _ENGINE_LOCK:
            if _ENGINE_INSTANCE is None:
                _ENGINE_INSTANCE = ZeroTrustNIDSEngine()
                _ENGINE_INSTANCE.ensure_initialized()
    return _ENGINE_INSTANCE

