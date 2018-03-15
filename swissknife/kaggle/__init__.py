"""
Module focused on dealing with Kaggle-specific format of data representation to
simplify files loading, processing and preparing for submission.
"""

from .datasets import KaggleClassifiedImagesSource


__all__ = ['KaggleClassifiedImagesSource']