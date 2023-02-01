"""
GAM datasets
"""
# -*- coding: utf-8 -*-

from pygam.datasets.load_datasets import (
    cake,
    chicago,
    coal,
    default,
    faithful,
    head_circumference,
    hepatitis,
    mcycle,
    toy_classification,
    toy_interaction,
    trees,
    wage,
)

__all__ = [
    "mcycle",
    "coal",
    "faithful",
    "trees",
    "wage",
    "default",
    "cake",
    "hepatitis",
    "toy_classification",
    "head_circumference",
    "chicago",
    "toy_interaction",
]

if __name__ == "__main__":
    DATASETS = [
        cake,
        chicago,
        coal,
        default,
        faithful,
        head_circumference,
        hepatitis,
        mcycle,
        toy_classification,
        toy_interaction,
        trees,
        wage,
    ]

    for data_loader in DATASETS:
        X, y = data_loader()
        print(data_loader, X.shape)
