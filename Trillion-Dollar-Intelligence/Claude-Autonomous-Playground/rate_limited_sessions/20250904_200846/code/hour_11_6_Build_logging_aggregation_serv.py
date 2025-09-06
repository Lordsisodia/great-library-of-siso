#!/usr/bin/env python3
"""
Build logging aggregation service
Production-ready Python implementation
Generated Hour 11 - Project 6
Created: 2025-09-04T20:56:55.361194
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AI-Generated Implementation:
**Logging Aggregation Service**

This implementation provides a logging aggregation service using Python. It consists of a logger class that aggregates logs from multiple sources and a configuration manager to handle configuration changes.

### Directory Structure

The code is organized in the following directory structure:

```bash
logging_aggregator/
|____ config/
|       |____ config.py
|____ src/
|       |____ __init__.py
|       |____ logger.py
|       |____ config_manager.py
|____ tests/
|       |____ test_logger.py
|____ main.py
|____ requirements.txt
|____ setup.py
```

### `config/config.py`

```python
# config/config.py

from typing import Dict

class Config:
    def __init__(self):
        self.logging_config = {
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": "logs/aggregated.log"
        }
        self.sources = [
            {"name": "source1", "log_level": "INFO", "log_file": "logs/source1.log"},
            {"name": "source2", "log_level": "DEBUG", "log_file": "logs/source2.log"}
        ]

    def update(self, config: Dict):
        self.logging_config = config.get("logging_config", self.logging_config)
        self.sources = config.get("sources", self.sources)

    def get_config(self):
        return self.logging_config, self.sources
```

### `src/logger.py`

```python
# src/logger.py

import logging
import logging.config
from typing import Dict
from config import config

class Logger:
    def __init__(self, name: str, log_level: str, log_file: str):
        self.name = name
        self.log_level = log_level
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(config.logging_config["log_format"])
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log(self, message: str, level: str = "INFO"):
        level = getattr(logging, level.upper())
        if level >= self.log_level:
            self.logger.log(level, message)

    def remove_handler(self):
        self.logger.removeHandler(self.logger.handlers[0])
```

### `src/config_manager.py`

```python
# src/config_manager.py

from typing import Dict
import json
from config import Config

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = Config()

    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                self.config.update(json.load(f))
        except FileNotFoundError:
            self.config.update({})

    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config.get_config(), f)

    def update_config(self, config: Dict):
        self.config.update(config)
        self.save_config()
```

### `main.py`

```python
# main.py

from src.logger import Logger
from src.config_manager import ConfigManager
import json

if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.load_config()

    logger1 = Logger("source1", config_manager.config.sources[0]["log_level"], config_manager.config.sources[0]["log_file"])
    logger2 = Logger("source2", config_manager.config.sources[1]["log_level"], config_manager.config.sources[1]["log_file"])

    logger1.log("This is a log message from source 1.")
    logger2.log("This is a log message from source 2.")

    config_manager.save_config()
```

### `tests/test_logger.py`

```python
# tests/test_logger.py

import unittest
from src.logger import Logger

class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger = Logger("test_logger", "INFO", "test.log")
        logger.log("This is a log message.")
        with open("test.log", "r") as f:
            log_message = f.read()
            self.assertIn("test_logger", log_message)
            self.assertIn("INFO", log_message)
            self.assertIn("This is a log message.", log_message)

if __name__ == "__main__":
    unittest.main()
```

### `setup.py`

```python
# setup.py

from setuptools import setup

setup(
    name="logging_aggregator",
    version="1.0",
    packages=["logging_aggregator"]
)
```

### `requirements.txt`

```bash
loguru
```

This implementation provides a basic logging aggregation service using Python. The service consists of a logger class that aggregates logs from multiple sources and a configuration manager to handle configuration changes.

**Example Use Cases:**

*   The logging aggregation service can be used to aggregate logs from multiple applications or services, providing a unified view of system activity.
*   The service can be configured to log messages from different sources at different levels (e.g., INFO, DEBUG, WARNING).
*   The service can be used to monitor system activity and detect potential issues before they become major problems.

**Advantages:**

*   The service provides a centralized logging mechanism, making it easier to manage and monitor system activity.
*   The service can be configured to log messages from different sources, providing a unified view of system activity.
*   The service can be used to detect potential issues before they become major problems.

**Disadvantages:**

*   The service requires configuration changes to be made to the logging properties of the source applications or services.
*   The service may require additional infrastructure to support large-scale logging operations.

if __name__ == "__main__":
    print(f"🚀 Build logging aggregation service")
    print(f"📊 Generated: 2025-09-04T20:56:55.361207")
    logger.info(f"Starting Build logging aggregation service...")
