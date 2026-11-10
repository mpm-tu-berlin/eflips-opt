import warnings

import pytest

from eflips.opt.util import ors_base_url, ors_client_from_env


@pytest.fixture
def clean_env(monkeypatch):
    """Neither ORS variable is set."""
    monkeypatch.delenv("OPENROUTESERVICE_BASE_URL", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    return monkeypatch


class TestOrsBaseUrl:
    def test_new_variable(self, clean_env):
        clean_env.setenv("OPENROUTESERVICE_BASE_URL", "http://new.invalid/ors/")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ors_base_url() == "http://new.invalid/ors/"

    def test_new_variable_wins_without_warning(self, clean_env):
        clean_env.setenv("OPENROUTESERVICE_BASE_URL", "http://new.invalid/ors/")
        clean_env.setenv("BASE_URL", "http://old.invalid/ors/")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ors_base_url() == "http://new.invalid/ors/"

    def test_legacy_variable_warns(self, clean_env):
        clean_env.setenv("BASE_URL", "http://old.invalid/ors/")
        with pytest.warns(PendingDeprecationWarning, match="OPENROUTESERVICE_BASE_URL"):
            assert ors_base_url() == "http://old.invalid/ors/"

    def test_neither_variable(self, clean_env):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert ors_base_url() is None

    def test_client_needs_a_variable(self, clean_env):
        with pytest.raises(ValueError, match="OPENROUTESERVICE_BASE_URL"):
            ors_client_from_env()
