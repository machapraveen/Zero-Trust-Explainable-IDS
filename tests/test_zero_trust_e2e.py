"""
End-to-End Verification Suite for the 5-Stage Explainable Deep Learning
Zero-Trust Intrusion Detection System (10-Model Ensemble + SHAP + SOC UI/API).
"""

import json
import pytest
from app import create_app, db
from app.models.database import User, Alert
from ml.zero_trust_pipeline import get_zero_trust_engine


@pytest.fixture
def zt_client():
    app = create_app("testing")
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            u = User(username="admin", email="admin@ztxai.local", role="admin")
            u.set_password("admin123")
            db.session.add(u)
            db.session.commit()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_five_stage_zero_trust_engine_and_shap():
    engine = get_zero_trust_engine()
    summary = engine.get_status_summary()

    assert summary["initialized"] is True
    assert summary["n_models"] == 10
    assert summary["shap_explainer_active"] is True
    assert summary["n_multilevel_features"] == 24

    # Benign enterprise HTTPS flow
    benign = engine.analyze_flow(
        {
            "src_ip": "192.168.1.25",
            "dst_ip": "10.0.0.10",
            "dst_port": 443,
            "protocol": "HTTPS",
            "duration": 1.8,
            "bytes_sent": 4200,
            "bytes_recv": 18600,
            "syn_flag_ratio": 0.08,
            "ack_flag_ratio": 0.86,
            "payload_entropy": 5.9,
            "lateral_movement_score": 0.04,
            "device_posture_risk": 0.08,
            "identity_privilege_risk": 0.06,
            "iot_edge_telemetry_dev": 0.05,
        }
    )
    assert benign["is_threat"] is False
    assert len(benign["ensemble_votes"]) == 10
    assert benign["zero_trust_decision"]["entity_trust_score"] > 60

    # Volumetric DDoS flow
    ddos = engine.analyze_flow(
        {
            "src_ip": "185.220.101.42",
            "dst_ip": "10.0.0.15",
            "dst_port": 80,
            "protocol": "TCP",
            "duration": 0.25,
            "bytes_sent": 820000,
            "bytes_recv": 640,
            "packets_sent": 9500,
            "packets_recv": 12,
            "syn_flag_ratio": 0.95,
            "ack_flag_ratio": 0.05,
            "payload_entropy": 2.6,
            "session_burst_score": 0.97,
            "device_posture_risk": 0.78,
        }
    )
    assert ddos["is_threat"] is True
    assert ddos["attack_type"] == "DDoS"
    assert len(ddos["shap_explanation"]["top_features"]) == 8
    assert "multilevel_attribution" in ddos["shap_explanation"]
    assert ddos["zero_trust_decision"]["policy_action"] in (
        "ZERO_TRUST_BLOCK_AND_ISOLATE",
        "MICRO_SEGMENT_QUARANTINE",
    )


def test_soc_routes_and_zero_trust_apis(zt_client):
    # Public health & status
    assert zt_client.get("/api/v1/health").status_code == 200
    st = zt_client.get("/api/v1/status").get_json()
    assert st["models_loaded"] is True
    assert st["n_models"] == 10
    assert st["shap_explainer_active"] is True

    # Live 5-stage analyze API creating an alert
    res = zt_client.post(
        "/api/v1/zero-trust/analyze",
        json={
            "record_alert": True,
            "flow": {
                "src_ip": "198.96.155.3",
                "dst_ip": "172.16.0.20",
                "dst_port": 4444,
                "protocol": "MQTT",
                "duration": 0.65,
                "bytes_sent": 145000,
                "bytes_recv": 4800,
                "syn_flag_ratio": 0.62,
                "payload_entropy": 7.72,
                "session_burst_score": 0.89,
                "lateral_movement_score": 0.76,
                "device_posture_risk": 0.92,
                "identity_privilege_risk": 0.79,
                "iot_edge_telemetry_dev": 0.96,
            },
        },
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert body["analysis"]["is_threat"] is True
    alert_id = body["analysis"]["alert_id"]

    # Authenticate as admin and verify all SOC Dashboard + Zero-Trust Lab pages
    login_res = zt_client.post(
        "/auth/login",
        data={"username": "admin", "password": "admin123"},
        follow_redirects=True,
    )
    assert login_res.status_code == 200

    for url in [
        "/dashboard",
        "/zero-trust",
        "/ai-models",
        "/alerts/",
        f"/alerts/{alert_id}",
        "/analytics/",
        "/analytics/traffic",
        "/analytics/threats",
        "/analytics/reports",
        "/showcase",
        "/auth/users",
        "/auth/settings",
        "/api/v1/",
        "/auth/register",
    ]:
        page = zt_client.get(url, follow_redirects=True)
        assert page.status_code == 200, f"Failed on {url}"

    # Verify AI model selector endpoint
    sel = zt_client.post(
        "/api/ai-models/select",
        json={"attack_type": "DDoS", "severity": "critical", "speed_priority": False},
    )
    assert sel.status_code == 200
    assert sel.get_json()["success"] is True
