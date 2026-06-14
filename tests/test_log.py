"""Tests for observability/logging module."""

import io
import json
import pytest
from obs.log import Logger, LogEntry, init_logger, log, get_logger


class TestLogEntry:
    def test_basic_entry(self):
        entry = LogEntry(event="test_event", data={"key": "value"})
        d = entry.to_dict()
        assert d["event"] == "test_event"
        assert d["key"] == "value"
        assert "timestamp" in d

    def test_empty_data(self):
        entry = LogEntry(event="bare")
        d = entry.to_dict()
        assert d["event"] == "bare"


class TestLogger:
    def test_human_format(self):
        buf = io.StringIO()
        logger = Logger(output=None, format="human")
        logger._file = buf
        logger.log("test", x=1, y=2)
        output = buf.getvalue()
        assert "test" in output
        assert "x=1" in output
        assert "y=2" in output

    def test_json_format(self):
        buf = io.StringIO()
        logger = Logger(output=None, format="json")
        logger._file = buf
        logger.log("test", x=1)
        output = buf.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["event"] == "test"
        assert parsed["x"] == 1

    def test_multiple_entries(self):
        buf = io.StringIO()
        logger = Logger(output=None, format="human")
        logger._file = buf
        logger.log("first")
        logger.log("second")
        entries = logger.get_entries()
        assert len(entries) == 2
        assert entries[0]["event"] == "first"
        assert entries[1]["event"] == "second"

    def test_elapsed_time(self):
        logger = Logger(output=None, format="human")
        logger._file = io.StringIO()
        logger.log("e1")
        logger.log("e2")
        entries = logger.get_entries()
        assert entries[1]["elapsed_s"] >= entries[0]["elapsed_s"]


class TestGlobalLogger:
    def test_init_and_log(self):
        buf = io.StringIO()
        init_logger(output=None, format="human")
        get_logger()._file = buf
        log("global_test", msg="hello")
        output = buf.getvalue()
        assert "global_test" in output
        assert "msg=hello" in output

    def test_singleton(self):
        l1 = get_logger()
        l2 = get_logger()
        assert l1 is l2
