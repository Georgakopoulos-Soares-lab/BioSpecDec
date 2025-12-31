"""Lightweight experiment pipeline for speculative decoding.

This package provides:
- unified generation runner that outputs JSONL records
- likelihood scoring that operates on saved token IDs
- sweep runner for grid experiments

It intentionally depends only on the existing environment (torch/transformers/etc.).
"""
