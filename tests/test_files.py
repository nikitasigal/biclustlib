import multiprocessing
import os

from biclustlib.algorithms import *
from biclustlib.benchmark import GeneExpressionBenchmark, Algorithm
from biclustlib.benchmark.data import load_tavazoie


def test_files():
    data = load_tavazoie()
    n_biclusters = 10
    reduction_level = 10
    discretion_level = 50

    setup = [
        Algorithm('CCA', ChengChurchAlgorithm(n_biclusters), data),
    ]

    with multiprocessing.Pool() as pool:
        tavazoie_benchmark = GeneExpressionBenchmark(algorithms=setup,
                                                     raw_data=data,
                                                     reduction_level=reduction_level).run(pool)
    tavazoie_benchmark.generate_report('custom_report')

    tavazoie_benchmark.perform_goea()
    tavazoie_benchmark.generate_goea_report('custom_report')

    assert os.path.isdir('custom_report'), 'Report directory was not created'

    expected = {'detailed_metrics', 'metrics', 'found.png', 'goea.png', 'goea_report.csv', 'report.csv', 'times.png'}
    actual = set(os.listdir('custom_report/'))
    assert actual == expected, 'Report directory has incorrect structure'

    expected = {'ASR.csv', 'MSR.csv', 'SMSR.csv', 'VE.csv', 'VEt.csv'}
    actual = set(os.listdir('custom_report/detailed_metrics/'))
    assert actual == expected, 'Detailed metrics directory has incorrect structure'

    expected = {'ASR.png', 'MSR.png', 'SMSR.png', 'VE.png', 'VEt.png'}
    actual = set(os.listdir('custom_report/metrics/'))
    assert actual == expected, 'Metrics plots directory has incorrect structure'


