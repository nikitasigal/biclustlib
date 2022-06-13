# biclustlib
The package is an extension of [biclustlib](https://github.com/padilha/biclustlib) Python library by Victor Alexandre Padilha.  
It is highly recommended to see the original repository first.  
The goal of this package is to create a unified biclustering framework for performing research on gene expression data and comparing different biclustering algorithms and measures.   

Distributed under GPLv3 license.
## Installation
```pip install biclustlib```    
You must also install [R](https://www.r-project.org/) and the following R packages:
+ [biclust](https://cran.r-project.org/web/packages/biclust/index.html)
+ [isa2](https://cran.r-project.org/web/packages/isa2/index.html)

## Benchmarking example
```python
import multiprocessing

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from biclustlib.algorithms import *
from biclustlib.algorithms.wrappers import *
from biclustlib.benchmark import GeneExpressionBenchmark, Algorithm
from biclustlib.benchmark.util import Converter
from biclustlib.benchmark.data import load_tavazoie, load_prelic


def discretize_data(raw_data: pd.DataFrame, n_bins: int = 2) -> pd.DataFrame:
    return pd.DataFrame(KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans').fit_transform(raw_data),
                        index=data.index).astype(int if n_bins > 2 else bool)


if __name__ == '__main__':
    pool = multiprocessing.Pool()

    data = load_tavazoie()
    n_biclusters = 5
    discretion_level = 30
    reduction_level = 15
    significance_cutoff = .05

    data_dis = discretize_data(data, discretion_level)
    data_bin = discretize_data(data)

    setup = [
        Algorithm('CCA', ChengChurchAlgorithm(n_biclusters), data),
        Algorithm('xMotifs', RConservedGeneExpressionMotifs(n_biclusters), data_dis),
        Algorithm('BiBit', BitPatternBiclusteringAlgorithm(), data_bin),
        Algorithm('Bimax', RBinaryInclusionMaximalBiclusteringAlgorithm(n_biclusters), data_bin),
        Algorithm('LAS', LargeAverageSubmatrices(n_biclusters), data),
        Algorithm('Plaid', RPlaid(n_biclusters), data),
        Algorithm('Spectral', Spectral(n_clusters=data.shape[1] // 2), data + 2),
        Algorithm('QUBIC', RConservedGeneExpressionMotifs(n_biclusters), data_bin),
    ]

    tavazoie_benchmark = GeneExpressionBenchmark(algorithms=setup,
                                                 raw_data=data,
                                                 n_biclusters=n_biclusters,
                                                 reduction_level=reduction_level).run(pool)
    tavazoie_benchmark.perform_goea()
    tavazoie_benchmark.generate_report()

    pool.close()

```
