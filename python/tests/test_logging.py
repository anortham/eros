"""Tests for logging setup and ErosConfig.logs_dir."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from eros.config import ErosConfig


class TestLogsDir:
    """ErosConfig.logs_dir property."""

    def test_logs_dir_under_data_dir(self):
        config = ErosConfig(eros_data_dir=Path("/tmp/test-eros"))
        assert config.logs_dir == Path("/tmp/test-eros/logs")

    def test_logs_dir_default(self):
        config = ErosConfig()
        assert config.logs_dir == Path(".eros/logs")


@pytest.fixture()
def clean_eros_logger():
    """Save and restore eros logger state between tests."""
    logger = logging.getLogger("eros")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    yield logger
    logger.handlers = original_handlers
    logger.level = original_level


class TestSetupLogging:
    """_setup_logging() configures console + file handlers."""

    def test_creates_log_directory(self, tmp_path, clean_eros_logger):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        assert (tmp_path / "eros-data" / "logs").is_dir()

    def test_adds_file_handler(self, tmp_path, clean_eros_logger):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        file_handlers = [
            h for h in clean_eros_logger.handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_file_handler_is_debug(self, tmp_path, clean_eros_logger):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        file_handler = next(
            h for h in clean_eros_logger.handlers
            if isinstance(h, RotatingFileHandler)
        )
        assert file_handler.level == logging.DEBUG

    def test_console_handler_is_info(self, tmp_path, clean_eros_logger):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        stream_handlers = [
            h for h in clean_eros_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.INFO

    def test_log_message_written_to_file(self, tmp_path, clean_eros_logger):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        clean_eros_logger.info("test sentinel message")

        log_file = tmp_path / "eros-data" / "logs" / "eros.log"
        assert log_file.exists()
        contents = log_file.read_text()
        assert "test sentinel message" in contents

    def test_debug_only_in_file_not_console(self, tmp_path, clean_eros_logger, capsys):
        from eros.server import _setup_logging

        config = ErosConfig(eros_data_dir=tmp_path / "eros-data")
        _setup_logging(config)
        clean_eros_logger.debug("debug-only-sentinel")

        # Should be in log file
        log_file = tmp_path / "eros-data" / "logs" / "eros.log"
        assert "debug-only-sentinel" in log_file.read_text()
