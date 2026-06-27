"""
Tanaka & Takii (2016) exact B&B for CRP-D (restricted + duplicate priorities).

Public symbol
-------------
Tanaka2016BBDuplicate : platform BaseAlgorithm subclass (entry point).
"""

from .algorithm import Tanaka2016BBDuplicate

__all__ = ["Tanaka2016BBDuplicate"]
