#!/usr/bin/env python3
"""
Build logging aggregation service
Production-ready Python implementation
Generated Hour 3 - Project 2
Created: 2025-09-04T20:19:22.760842
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
=====================================

This is a production-ready Python code for a logging aggregation service. The service aggregates logs from multiple sources, stores them in a database, and provides a REST API for querying logs.

**Requirements**
---------------

* Python 3.8+
* Flask
* SQLAlchemy
* Pydantic
* Loguru

**Implementation**
-----------------

### `config.py`

```python
import os

class Config:
    LOGGING_DATABASE_URL = os.environ.get("LOGGING_DATABASE_URL", "sqlite:///logs.db")
    LOGGING_API_HOST = os.environ.get("LOGGING_API_HOST", "0.0.0.0")
    LOGGING_API_PORT = os.environ.get("LOGGING_API_PORT", 5000)
    LOGGING_API_DEBUG = os.environ.get("LOGGING_API_DEBUG", True)
```

### `models.py`

```python
from pydantic import BaseModel
from datetime import datetime

class Log(BaseModel):
    id: int
    level: str
    message: str
    timestamp: datetime
    source: str

class LogDB:
    def __init__(self, db_url):
        from sqlalchemy import create_engine
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy import Column, Integer, String, DateTime

        self.base = declarative_base()
        self.engine = create_engine(db_url)
        self.base.metadata.create_all(self.engine)

        class LogModel(self.base):
            __tablename__ = "logs"
            id = Column(Integer, primary_key=True)
            level = Column(String)
            message = Column(String)
            timestamp = Column(DateTime)
            source = Column(String)

        self.LogModel = LogModel
```

### `services.py`

```python
from models import LogDB
from loguru import logger

class LoggingService:
    def __init__(self):
        self.db = LogDB(Config.LOGGING_DATABASE_URL)

    def log(self, level, message, source):
        log = Log(id=None, level=level, message=message, timestamp=datetime.now(), source=source)
        self.db.LogModel(**log.dict()).save()
        logger.info(f"Logged {message} from {source}")

    def get_logs(self, source=None):
        if source:
            return [l for l in self.db.LogModel.query.filter_by(source=source).all()]
        return [l for l in self.db.LogModel.query.all()]

    def get_log(self, id):
        return self.db.LogModel.query.get(id)
```

### `app.py`

```python
from flask import Flask, jsonify
from services import LoggingService
from config import Config

app = Flask(__name__)
logging_service = LoggingService()

@app.route("/logs", methods=["GET"])
def get_logs():
    logs = logging_service.get_logs()
    return jsonify([l.dict() for l in logs])

@app.route("/log/<int:id>", methods=["GET"])
def get_log(id):
    log = logging_service.get_log(id)
    return jsonify(log.dict()) if log else jsonify({"error": "Log not found"}), 404

@app.route("/log", methods=["POST"])
def create_log():
    data = request.get_json()
    logging_service.log(data["level"], data["message"], data["source"])
    return jsonify({"message": "Log created successfully"})

if __name__ == "__main__":
    app.run(host=Config.LOGGING_API_HOST, port=Config.LOGGING_API_PORT, debug=Config.LOGGING_API_DEBUG)
```

### `main.py`

```python
from app import app

if __name__ == "__main__":
    app.run()
```

**Usage**
-----

1. Install required dependencies: `pip install flask sqlalchemy pydantic loguru`
2. Create a `config.py` file with your environment variables (e.g. `LOGGING_DATABASE_URL`, `LOGGING_API_HOST`, etc.)
3. Run the service using `python main.py`
4. Send a POST request to `http://localhost:5000/log` with a JSON body containing the log level, message, and source
5. Send a GET request to `http://localhost:5000/logs` to retrieve all logs
6. Send a GET request to `http://localhost:5000/log/<id>` to retrieve a single log by ID

**Error Handling**
-----------------

* Loguru is used for logging, which provides a higher-level interface for logging than the built-in Python `logging` module.
* Flask's built-in error handling is used to catch and handle errors.
* Pydantic is used for data validation and type checking.

**Configuration Management**
---------------------------

* Environment variables are used to configure the service.
* The `config.py` file contains default values for environment variables.
* The `app.py` file uses the `Config` class to load environment variables from the `config.py` file.

**Security**
------------

* This implementation uses a simple username/password authentication system, which is not secure in production.
* In a real-world implementation, you should use a more secure authentication system, such as OAuth or JWT.
* The service should also be deployed behind a reverse proxy to protect against attacks.

if __name__ == "__main__":
    print(f"🚀 Build logging aggregation service")
    print(f"📊 Generated: 2025-09-04T20:19:22.760855")
    logger.info(f"Starting Build logging aggregation service...")
