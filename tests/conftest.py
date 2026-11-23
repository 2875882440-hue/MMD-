"""Isolate the app's import-time recovery before test modules are collected.

tmp_path fixtures run after collection, so patching DATA inside an API fixture
is too late to protect live jobs while motionbridge.app is first imported.
"""
import logging
import os
from pathlib import Path
import uuid

PROJECT=Path(__file__).resolve().parents[1]
IMPORT_STORE=PROJECT/'.test-runtime'/('app-import-'+uuid.uuid4().hex)
os.environ['MOTIONBRIDGE_DATA']=str(IMPORT_STORE)
# Avoid mixing synthetic API messages with the running application's log.
logging.getLogger('motionbridge').addHandler(logging.NullHandler())
