from timeit import default_timer

import numpy as np
import pandas as pd

from ..models import Biclustering

# Warnings are usually ignored and just printed, however some metrics rely on Warnings
np.seterr(divide='warn', invalid='warn')


def asr(biclustering: Biclustering, data: pd.DataFrame) -> dict:
    start = default_timer()
    results = []
    for bicluster in biclustering.biclusters:
        b = data.iloc[bicluster.rows, bicluster.cols]
        rows, cols = b.shape
        row_corr = b.transpose().corr(method='spearman').to_numpy()
        col_corr = b.corr(method='spearman').to_numpy()

        rho_gene = (np.nan_to_num(row_corr).sum() - np.nan_to_num(row_corr.diagonal()).sum()) / (
            max(1, rows * (rows - 1)))
        rho_condition = (np.nan_to_num(col_corr).sum() - np.nan_to_num(col_corr.diagonal()).sum()) / (
            max(1, cols * (cols - 1)))

        results.append(max(rho_gene, rho_condition))
    end = default_timer()
    if len(results) == 0:
        results = [-1]
    return {'time': end - start,
            'avg': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)}


def _standardize(b: np.array) -> np.array:
    row_means = np.mean(b, axis=1).reshape(-1, 1)
    row_stds = np.std(b, axis=1).reshape(-1, 1)
    if np.any(row_stds == 0):
        raise Warning
    # row_stds = np.array([s if s != 0 else 1 for s in row_stds]).reshape(-1, 1)
    return (b - row_means) / row_stds


def ve(biclustering: Biclustering, data: pd.DataFrame) -> dict:
    start = default_timer()
    results = []
    for bicluster in biclustering.biclusters:
        b = data.iloc[bicluster.rows, bicluster.cols].to_numpy()
        try:
            bs = _standardize(b)
        except Warning:
            results.append(1)
        else:
            virtual_gene = np.mean(bs, axis=0)
            results.append(np.sum(np.abs(bs - virtual_gene), axis=None) / (b.shape[0] * b.shape[1]))
    end = default_timer()
    if len(results) == 0:
        results = [-1]
    return {'time': end - start,
            'avg': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)}


def vet(biclustering: Biclustering, data: pd.DataFrame) -> dict:
    start = default_timer()
    results = []
    for bicluster in biclustering.biclusters:
        b = data.iloc[bicluster.rows, bicluster.cols].to_numpy()
        try:
            bs = _standardize(b)
        except Warning:
            results.append(1)
        else:
            virtual_condition = np.mean(bs, axis=1).reshape(-1, 1)
            results.append(np.sum(np.abs(bs - virtual_condition), axis=None) / (b.shape[0] * b.shape[1]))
    end = default_timer()
    if len(results) == 0:
        results = [-1]
    return {'time': end - start,
            'avg': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)}


def msr(biclustering: Biclustering, data: pd.DataFrame) -> dict:
    start = default_timer()
    results = []
    for bicluster in biclustering.biclusters:
        b = data.iloc[bicluster.rows, bicluster.cols].to_numpy()
        row_means = np.mean(b, axis=1).reshape(-1, 1)
        col_means = np.mean(b, axis=0)
        all_mean = np.mean(b, axis=None)
        results.append(np.sum((b - row_means - col_means + all_mean) ** 2) / (b.shape[0] * b.shape[1]))
    end = default_timer()
    if len(results) == 0:
        results = [-1]
    return {'time': end - start,
            'avg': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)}


def smsr(biclustering: Biclustering, data: pd.DataFrame) -> dict:
    start = default_timer()
    results = []
    for bicluster in biclustering.biclusters:
        b = data.iloc[bicluster.rows, bicluster.cols].to_numpy()
        row_means = np.mean(b, axis=1).reshape(-1, 1)
        row_means_2 = row_means ** 2
        col_means = np.mean(b, axis=0)
        col_means_2 = col_means ** 2
        all_mean = np.mean(b, axis=None)
        div = row_means_2 * col_means_2
        div[div == 0] = 1
        results.append(
            np.sum(((row_means * col_means - b * all_mean) ** 2 / div)) / (b.shape[0] * b.shape[1]))
    end = default_timer()
    if len(results) == 0:
        results = [-1]
    return {'time': end - start,
            'avg': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results)}


METRICS = {'ASR': asr,
           'MSR': msr,
           'SMSR': smsr,
           'VE': ve,
           'VEt': vet}


def calculate_metric(metric: str, biclustering: Biclustering, data: pd.DataFrame) -> dict:
    if metric not in METRICS:
        raise ValueError(f'Metric \'{metric}\' is not defined')
    return METRICS[metric](biclustering, data)
