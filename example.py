import multiprocessing

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from biclustlib.algorithms import *
from biclustlib.algorithms.wrappers import *
from biclustlib.benchmark import GeneExpressionBenchmark, Algorithm
from biclustlib.benchmark.data import load_tavazoie, load_prelic, load_jaskowiak


def discretize_data(raw_data: pd.DataFrame, n_bins: int = 2) -> pd.DataFrame:
    return pd.DataFrame(KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans').fit_transform(raw_data),
                        index=raw_data.index).astype(int if n_bins > 2 else bool)


if __name__ == '__main__':
    data = load_tavazoie()
    n_biclusters = 10
    reduction_level = 10
    discretion_level = 50

    data_dis = discretize_data(data, discretion_level)
    data_bin = discretize_data(data)

    setup = [
        Algorithm('CCA', ChengChurchAlgorithm(n_biclusters), data),
        Algorithm('xMotifs', RConservedGeneExpressionMotifs(n_biclusters), data_dis),
        Algorithm('BiBit', BitPatternBiclusteringAlgorithm(), data_bin),
        Algorithm('Bimax', RBinaryInclusionMaximalBiclusteringAlgorithm(n_biclusters), data_bin),
        Algorithm('LAS', LargeAverageSubmatrices(n_biclusters), data),
        Algorithm('Plaid', RPlaid(n_biclusters), data),
        Algorithm('Spectral', Spectral(n_clusters=data.shape[1] // 2), data + abs(data.min().min()) + 1),
    ]

    with multiprocessing.Pool() as pool:
        tavazoie_benchmark = GeneExpressionBenchmark(algorithms=setup,
                                                     raw_data=data,
                                                     reduction_level=reduction_level).run(pool)
    tavazoie_benchmark.generate_report()

    tavazoie_benchmark.perform_goea()
    tavazoie_benchmark.generate_goea_report()
    print(tavazoie_benchmark.overall_execution_time)
    print(tavazoie_benchmark.goea_time)
