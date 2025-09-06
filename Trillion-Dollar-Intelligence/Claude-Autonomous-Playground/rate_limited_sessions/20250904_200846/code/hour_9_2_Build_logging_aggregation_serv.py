#!/usr/bin/env python3
"""
Build logging aggregation service
Production-ready Python implementation
Generated Hour 9 - Project 2
Created: 2025-09-04T20:47:06.579609
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
**Logging Aggregation Service in Python**
=====================================

**Overview**
------------

This logging aggregation service is designed to collect and store logs from various sources in a centralized location. The service uses a message queue (RabbitMQ) to receive log messages, and a database (PostgreSQL) to store the aggregated logs.

**Requirements**
---------------

* Python 3.8+
* RabbitMQ
* PostgreSQL
* `pika` for RabbitMQ interaction
* `psycopg2` for PostgreSQL interaction

**Implementation**
-----------------

### `config.py`

```python
from typing import Dict

class Config:
    def __init__(self):
        self.rabbitmq = {
            'host': 'localhost',
            'port': 5672,
            'username': 'guest',
            'password': 'guest',
            'queue_name': 'logs',
        }
        self.postgresql = {
            'host': 'localhost',
            'database': 'logs',
            'username': 'postgres',
            'password': 'postgres',
        }
        self.log_level = 'INFO'
```

### `logging_service.py`

```python
import logging
import pika
from typing import Optional
from config import Config

class LoggingService:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.getLevelName(self.config.log_level))
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

        self.rabbitmq_connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=self.config.rabbitmq['host'],
            port=self.config.rabbitmq['port'],
            credentials=pika.PlainCredentials(self.config.rabbitmq['username'], self.config.rabbitmq['password']),
        ))
        self.rabbitmq_channel = self.rabbitmq_connection.channel()
        self.rabbitmq_channel.queue_declare(queue=self.config.rabbitmq['queue_name'])

    def start(self):
        self.rabbitmq_channel.basic_consume(
            on_message_callback=self.process_message,
            queue=self.config.rabbitmq['queue_name'],
            no_ack=True,
        )
        self.logger.info('Logging service started. Waiting for messages...')
        self.rabbitmq_channel.start_consuming()

    def process_message(self, ch, method, properties, body):
        try:
            self.logger.info(f'Received message: {body.decode()}')
            self.save_log(body.decode())
        except Exception as e:
            self.logger.error(f'Error processing message: {str(e)}')
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)

    def save_log(self, log_message: str):
        try:
            import psycopg2
            db = psycopg2.connect(
                host=self.config.postgresql['host'],
                database=self.config.postgresql['database'],
                user=self.config.postgresql['username'],
                password=self.config.postgresql['password'],
            )
            cur = db.cursor()
            cur.execute("INSERT INTO logs (message) VALUES (%s)", (log_message,))
            db.commit()
            cur.close()
            db.close()
            self.logger.info(f'Log saved: {log_message}')
        except Exception as e:
            self.logger.error(f'Error saving log: {str(e)}')
```

### `main.py`

```python
from logging_service import LoggingService
from config import Config

if __name__ == '__main__':
    config = Config()
    logging_service = LoggingService(config)
    logging_service.start()
```

**Error Handling**
------------------

*   The logging service uses a try-except block to catch any exceptions that may occur while processing messages or saving logs.
*   If an exception occurs, the service logs the error message and continues processing the next message.
*   If an error occurs while saving a log, the service logs the error message and continues processing the next message.

**Configuration Management**
---------------------------

*   The logging service uses a configuration file (`config.py`) to store the connection details for RabbitMQ and PostgreSQL.
*   The configuration file can be modified to change the connection details or other settings.
*   The logging service uses the configuration file to connect to RabbitMQ and PostgreSQL.

**Logging**
----------

*   The logging service uses the Python `logging` module to log messages at different levels (e.g., INFO, WARNING, ERROR).
*   The service logs messages at the INFO level by default, but this can be changed by modifying the `log_level` setting in the configuration file.
*   The service logs messages to the console and saves them to the logs database.

**Database Schema**
-------------------

*   The logs database table has a single column, `message`, which stores the log message.

```sql
CREATE TABLE logs (
    message TEXT
);
```

**Example Use Case**
--------------------

To use the logging service, you can create a Python script that sends log messages to the RabbitMQ queue using the `pika` library. Then, start the logging service using the `main.py` script.

```python
import pika

# Create a connection to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='logs')

# Send a log message
channel.basic_publish(exchange='',
                      routing_key='logs',
                      body='This is a log message.')

connection.close()
```

When the logging service receives the log message, it will save it to the logs database and log a message to the console.

if __name__ == "__main__":
    print(f"🚀 Build logging aggregation service")
    print(f"📊 Generated: 2025-09-04T20:47:06.579624")
    logger.info(f"Starting Build logging aggregation service...")
