"""Check the actual CPU preflight's pinned data and private Hub write probe."""

import pytest

from scripts.cloud.checkpoint_preflight import find_index
from workflow_distillation.epoch_data import sha256_file


def test_preflight_finds_new_kaggle_mount_and_checks_expected_index(tmp_path):
    mounted = tmp_path / "datasets/owner/epoch/index.json"
    mounted.parent.mkdir(parents=True)
    mounted.write_text('{"data":[]}')
    expected = sha256_file(mounted)
    assert find_index(tmp_path / "epoch/index.json", expected, root=tmp_path) == mounted
    with pytest.raises(ValueError, match="checksum"):
        find_index(mounted, "0" * 64, root=tmp_path)


def test_preflight_rejects_ambiguous_dataset_mount(tmp_path):
    for owner in ("one", "two"):
        path = tmp_path / owner / "epoch/index.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    with pytest.raises(ValueError, match="unique"):
        find_index(tmp_path / "epoch/index.json", "0" * 64, root=tmp_path)


def test_secret_diagnostics_expose_status_without_response_or_token():
    from urllib.error import HTTPError

    from scripts.cloud.checkpoint_preflight import secret_error_details

    cause = HTTPError("https://example.invalid", 403, "secret hf_DO_NOT_LOG", {}, None)
    error = ConnectionError("secret hf_DO_NOT_LOG")
    error.__cause__ = cause
    details = secret_error_details(error)
    assert details == {"error_type": "ConnectionError", "http_status": 403,
                       "secret_access_hint": "not_authorized_for_session"}
    assert "hf_" not in str(details)


def test_secret_diagnostics_keep_unknown_network_failure_distinct():
    from scripts.cloud.checkpoint_preflight import secret_error_details

    assert secret_error_details(ConnectionError("private message")) == {
        "error_type": "ConnectionError", "secret_access_hint": "network_or_service_error",
    }


def test_secret_diagnostics_classify_missing_attachment_without_logging_response():
    import io
    from urllib.error import HTTPError

    from scripts.cloud.checkpoint_preflight import secret_error_details

    body = b'{"error":"No user secrets exist for kernel 123 label HF_TOKEN hf_DO_NOT_LOG"}'
    error = HTTPError("https://example.invalid", 400, "bad request", {}, io.BytesIO(body))
    details = secret_error_details(error)
    assert details["http_status"] == 400
    assert details["secret_access_hint"] == "secret_not_attached_to_session"
    assert "hf_" not in str(details)


def test_secret_service_diagnostics_only_emit_fixed_clues():
    import io
    from urllib.error import HTTPError

    from scripts.cloud.checkpoint_preflight import secret_error_details

    body = b'<html>400 Bad Request. Malformed request, hf_SUPER_SECRET_VALUE.</html>'
    error = HTTPError("https://example.invalid", 400, "bad request", {}, io.BytesIO(body))
    details = secret_error_details(error)
    assert details["response_format"] == "html"
    assert details["service_response_clues"] == ["malformed", "request"]
    assert "SECRET" not in str(details)
