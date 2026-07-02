import pytest
from ingestion.http_client import get_ssl_verify


def test_get_ssl_verify_defaults_to_true(monkeypatch):
    monkeypatch.delenv("HTTPX_SSL_VERIFY", raising=False)
    assert get_ssl_verify() is True


def test_get_ssl_verify_returns_ca_bundle_path(monkeypatch):
    monkeypatch.setenv("HTTPX_SSL_VERIFY", "/etc/ssl/corporate-ca.pem")
    assert get_ssl_verify() == "/etc/ssl/corporate-ca.pem"


def test_get_ssl_verify_false_disables_and_warns(monkeypatch):
    monkeypatch.setenv("HTTPX_SSL_VERIFY", "false")
    with pytest.warns(UserWarning):
        assert get_ssl_verify() is False
