"""Tests for src.spark, with a stand-in for SparkSession.builder (no cluster needed)."""

import pytest

import src.spark
from src.config import Settings

CONFIG = {"spark": {"app_name": "test-app", "timezone": "UTC"}}
TIMEZONE = ("config", "spark.sql.session.timeZone", "UTC")


class FakeBuilder:
    """Records the builder calls get_spark makes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
            return self

        return record


@pytest.fixture(name="builder")
def fake_builder(monkeypatch):
    """Replace SparkSession with a class whose builder records calls."""
    builder = FakeBuilder()
    monkeypatch.setattr(src.spark, "SparkSession", type("FakeSession", (), {"builder": builder}))
    return builder


def settings(remote=None, package=None):
    """Settings with only the Spark values that matter here."""
    return Settings(panel_path=None, spark_master="local[2]", spark_remote=remote, graphframes_package=package)


def test_remote_connects_with_name_and_timezone(builder):
    src.spark.get_spark(settings(remote="sc://example:15002", package="ignored"), CONFIG)
    assert ("remote", "sc://example:15002") in builder.calls
    assert ("appName", "test-app") in builder.calls
    assert TIMEZONE in builder.calls
    assert not any(call[0] == "master" or "spark.jars.packages" in call for call in builder.calls)
    assert builder.calls[-1] == ("getOrCreate",)


def test_local_adds_graphframes_package_and_timezone(builder):
    src.spark.get_spark(settings(package="io.graphframes:pkg"), CONFIG)
    assert ("master", "local[2]") in builder.calls
    assert ("config", "spark.jars.packages", "io.graphframes:pkg") in builder.calls
    assert TIMEZONE in builder.calls


def test_local_without_package_adds_no_jars(builder):
    src.spark.get_spark(settings(), CONFIG)
    assert not any("spark.jars.packages" in call for call in builder.calls)
