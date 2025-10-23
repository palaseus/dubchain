"""
REST API Module

FastAPI-based REST API for DubChain blockchain operations.
"""

import logging

logger = logging.getLogger(__name__)
from .app import app

__all__ = ["app"]
