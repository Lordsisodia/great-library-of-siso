#!/usr/bin/env python3
"""
Build logging aggregation service
Production-ready Python implementation
Generated Hour 1 - Project 2
Created: 2025-09-04T20:10:06.259947
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
=============================

### Overview

This Python module provides a logging aggregation service that collects logs from multiple sources and stores them in a centralized database. The service uses a client-server architecture, where the client is responsible for sending logs to the server, and the server is responsible for storing and processing the logs.

### Requirements

* Python 3.8+
* `sqlalchemy` for database interactions
* `logging` for log handling
* `configparser` for configuration management

### Implementation

#### **config.py**
```python
import configparser
from typing import Dict

class Config:
    def __init__(self, config_file: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_database_url(self) -> str:
        """Returns the database URL from the configuration file."""
        return self.config.get('database', 'url')

    def get_log_dir(self) -> str:
        """Returns the log directory path from the configuration file."""
        return self.config.get('logging', 'dir')

    def get_log_level(self) -> str:
        """Returns the log level from the configuration file."""
        return self.config.get('logging', 'level')
```

#### **log_entry.py**
```python
from typing import Dict
from datetime import datetime

class LogEntry:
    def __init__(self, message: str, level: str, timestamp: datetime):
        self.message = message
        self.level = level
        self.timestamp = timestamp

    @classmethod
    def from_dict(cls, data: Dict):
        """Creates a LogEntry instance from a dictionary."""
        return cls(data['message'], data['level'], datetime.fromisoformat(data['timestamp']))

    def to_dict(self) -> Dict:
        """Returns a dictionary representation of the LogEntry instance."""
        return {'message': self.message, 'level': self.level, 'timestamp': self.timestamp.isoformat()}
```

#### **logger.py**
```python
import logging
from typing import Dict
from log_entry import LogEntry

class Logger:
    def __init__(self, log_level: str, log_dir: str):
        self.log_level = log_level
        self.log_dir = log_dir
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.getLevelName(log_level))
        self.handler = logging.FileHandler(log_dir)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)

    def error(self, message: str):
        """Logs an error message."""
        self.logger.error(message)
```

#### **log_server.py**
```python
import os
import json
from typing import Dict
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from log_entry import LogEntry
from config import Config

class Log(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    message = Column(String)
    level = Column(String)
    timestamp = Column(DateTime)

class LogServer:
    def __init__(self, config: Config):
        self.config = config
        self.engine = create_engine(config.get_database_url())
        self.Base = declarative_base()
        self.Base.metadata.create_all(self.engine)

    def add_log(self, log_entry: LogEntry):
        """Adds a log entry to the database."""
        new_log = Log(message=log_entry.message, level=log_entry.level, timestamp=log_entry.timestamp)
        self.engine.execute(self.Base.metadata.insert(self.Base.metadata.tables['logs']), new_log)

    def get_logs(self) -> Dict:
        """Returns all logs from the database."""
        logs = self.engine.execute(self.Base.metadata.tables['logs']).fetchall()
        return [log_entry.to_dict() for log_entry in logs]
```

#### **client.py**
```python
import json
from typing import Dict
from config import Config
from log_server import LogServer
from logger import Logger

class Client:
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config.get_log_level(), config.get_log_dir())
        self.log_server = LogServer(config)

    def send_log(self, message: str, level: str):
        """Sends a log message to the server."""
        log_entry = LogEntry(message, level, datetime.now())
        self.logger.info(f'Sending log: {log_entry.message}')
        self.log_server.add_log(log_entry)
        self.logger.info(f'Log sent successfully')
```

#### **main.py**
```python
import logging
from config import Config
from client import Client

def main():
    config = Config('config.ini')
    client = Client(config)
    client.send_log('This is a test log message', 'INFO')

if __name__ == '__main__':
    main()
```

### Configuration

Create a `config.ini` file with the following content:
```ini
[database]
url = postgresql://user:password@host:port/dbname

[logging]
dir = /path/to/log/directory
level = INFO
```

### Usage

1. Run the `main.py` script to send a log message to the server.
2. Run the `log_server.py` script to start the log server.
3. Use the `client.py` script to send log messages to the server.

### Documentation

* `config.py`: Provides a `Config` class that loads configuration from a file.
* `log_entry.py`: Provides a `LogEntry` class that represents a log entry.
* `logger.py`: Provides a `Logger` class that logs messages to a file.
* `log_server.py`: Provides a `LogServer` class that stores logs in a database.
* `client.py`: Provides a `Client` class that sends log messages to the server.
* `main.py`: The main script that demonstrates how to use the logging aggregation service.

### Testing

* Create a test database and modify the `config.ini` file to point to the test database.
* Run the `log_server.py` script to start the test log server.
* Run the `client.py` script to send log messages to the test server.
* Use a tool like `psql` to verify that the logs are stored in the test database.

### Error Handling

* Use try-except blocks to catch and handle exceptions in the `client.py` and `log_server.py` scripts.
* Log errors using the `logger.py` script.
* Use the `logging` module to configure logging levels and handlers.

if __name__ == "__main__":
    print(f"🚀 Build logging aggregation service")
    print(f"📊 Generated: 2025-09-04T20:10:06.259972")
    logger.info(f"Starting Build logging aggregation service...")
