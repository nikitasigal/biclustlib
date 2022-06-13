import multiprocessing
import os.path
from os.path import join, dirname

import matplotlib.pyplot as plt
import pandas as pd
from goatools.anno.gaf_reader import GafReader
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag

from ..algorithms._base import BaseBiclusteringAlgorithm
from ..evaluation.goea import enrich_biclustering, enrichment_rate
from ..evaluation.internal import *
from ..models import Biclustering


class Algorithm:
    """
    This class a convenience wrapper for running biclustering algorithms in a benchmark.

    Parameters
    ----------
    name : str
        The name of the algorithm to be used in plots and reports.

    instance : BaseBiclusteringAlgorithm
        An instantiated instance of the algorithm.

    data : pd.DataFrame
        Data to be passed to the algorithm.
    """

    def __init__(self,
                 name: str,
                 instance: BaseBiclusteringAlgorithm,
                 data: pd.DataFrame):
        self.name = name
        self.instance = instance
        self.data = data

    def run(self) -> dict:
        """
        Run the algorithm and measure execution time

        :return: A dictionary with execution time, resulting biclustering and number of biclusters found
        """
        start = default_timer()
        result = self.instance.run(self.data)
        end = default_timer()
        return {'execution_time': end - start,
                'biclustering': result,
                'n_found': len(result.biclusters)}


class GeneExpressionBenchmark:
    def __init__(self,
                 algorithms: list[Algorithm],
                 raw_data: pd.DataFrame,
                 internal_metrics: list[str] = METRICS,
                 reduction_level: int = -1):
        """
        This class represents an instance of a biclustering benchmark for gene expression data

        :param algorithms: A list of algorithms to be used in the benchmark.
        :param raw_data: The original data from the dataset.
        :param internal_metrics: A list of internal metrics to be calculated.
            For available metrics see biclustlib.evaluation.internal.METRICS
        :param reduction_level: Biclusterings, that are larger than this amount, will be reduced to it by taking a random sample.
            No reduction is done if set to -1.
        """

        self.algorithms = algorithms
        self.raw_data = raw_data

        self.internal_metrics = internal_metrics

        self.reduction_level = reduction_level

        self.report = {algorithm.name: None for algorithm in self.algorithms}

    def __iter__(self):
        return self.report.__iter__()

    def __getitem__(self, item):
        return self.report[item]

    def __setitem__(self, key, value):
        self.report[key] = value

    def __len__(self):
        return len(self.report)

    @staticmethod
    def _run_in_parallel(gene_benchmark,
                         algorithm: Algorithm
                         ) -> dict:
        """
        Run the algorithm and calculate all metrics in a separate process.

        :param gene_benchmark: Benchmark, for which the algorithm is run.
        :param algorithm: An algorithm to run.
        :return: Dictionary with the resulting biclustering and metrics statistics
        """
        report = algorithm.run()

        if gene_benchmark.reduction_level != -1:
            report['biclustering'] = Biclustering(np.random.choice(
                report['biclustering'].biclusters,
                min(gene_benchmark.reduction_level, len(report['biclustering'].biclusters)),
                replace=False
            ))

        for metric in gene_benchmark.internal_metrics:
            report[metric] = calculate_metric(metric, report['biclustering'], gene_benchmark.raw_data)

        return report

    def run(self, multiprocessing_pool: multiprocessing.Pool = None) -> any:
        """
        Run the benchmark and calculate all specified metrics.

        :param multiprocessing_pool: If provided, all algorithms will be run in parallel.
            Can decrease overall execution time, but might affect per algorithm performance depending on CPU configuration.
        :return: self
        """

        # Start the timer to measure overall computation time
        experiment_start = default_timer()

        # If a multiprocessing pool is provided, run in parallel
        if multiprocessing_pool is not None:
            multiprocessing_results = multiprocessing_pool.starmap(GeneExpressionBenchmark._run_in_parallel,
                                                                   [(self, algorithm)
                                                                    for algorithm in self.algorithms])
            for i in range(len(self)):
                self[self.algorithms[i].name] = multiprocessing_results[i]
        else:
            # Run all models and extract resulting Biclustering instances
            for algorithm in self.algorithms:
                self[algorithm.name] = algorithm.run()

            # Reduce biclusterings by randomly choosing only the amount of biclusters originally intended
            # Otherwise may !significantly! increase computation times
            if self.reduction_level != -1:
                for algorithm in self:
                    self[algorithm]['biclustering'] = Biclustering(np.random.choice(
                        self[algorithm]['biclustering'].biclusters,
                        min(self.reduction_level, len(self[algorithm]['biclustering'].biclusters)),
                        replace=False
                    ))

            # Calculate all requested metrics
            # If a multiprocessing pool is provided, run in parallel
            for metric in self.internal_metrics:
                for algorithm in self:
                    self[algorithm][metric] = calculate_metric(metric, self[algorithm]['biclustering'], self.raw_data)

        # Save computation time
        experiment_end = default_timer()
        self.overall_execution_time = experiment_end - experiment_start

        return self

    def perform_goea(self,
                     pop_path: str = join(dirname(__file__), 'util', 'sgd.pop'),
                     gaf_path: str = join(dirname(__file__), 'util', 'sgd.gaf'),
                     obo_path: str = join(dirname(__file__), 'util', 'go-basic.obo'),
                     alphas: list[float] = (.05, .01, .005, .0001, .00001)
                     ) -> any:
        """
        Perform GO Enrichment Analysis for each bicluster in each algorithm.

        :param pop_path: A path to a file, containing all genes in the genome. Defaults to Yeast
        :param gaf_path: A path to a .gaf associations file. Defaults to Yeast.
        :param obo_path: A path to a .obo GO Basic or GO Slim annotations file. Defaults to go-basic.obo
        :param alphas: A list of significance level. For each significance level an enrichment rate will be calculated.
        :return: self
        """

        try:
            with open(pop_path) as f:
                population = [line.split()[0] for line in f]
        except OSError:
            raise Exception(f'File {pop_path} does not exist')

        goea_study = GOEnrichmentStudy(pop=population,
                                       assoc=GafReader(gaf_path, prt=None).read_gaf(),
                                       obo_dag=GODag(obo_path, prt=None),
                                       methods=['fdr_bh'])

        goea_start = default_timer()
        for algorithm in self:
            self[algorithm].update(enrich_biclustering(goea_study, self.raw_data.index.to_list(),
                                                       self[algorithm]['biclustering']))
            for alpha in alphas:
                self[algorithm][f'a = {alpha}'] = enrichment_rate(
                    self[algorithm]['enriched_biclustering'], alpha)

        goea_end = default_timer()
        self.goea_time = goea_end - goea_start
        self.alphas = alphas

        return self

    def generate_report(self, report_dir: str = 'report') -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Generate a report on benchmark results - draw plots and tables.

        :param report_dir: Specify a directory, where the report results will be saved.
        :return: A tuple: general report and a dictionary of detailed reports for each metric
        """

        if not hasattr(self, 'overall_execution_time'):
            print('Benchmark was never executed')
            print('See .run() method')
            return

        GeneExpressionBenchmark._check_dir(report_dir)
        GeneExpressionBenchmark._check_dir(join(report_dir, 'detailed_metrics'))

        self.plot_execution_times(report_dir)
        self.plot_metrics(report_dir)
        self.plot_found(report_dir)

        report = pd.DataFrame(index=self.report)
        detailed_reports = {}
        report['Execution time'] = [self[algorithm]['execution_time'] for algorithm in self]
        report['Biclusters found'] = [self[algorithm]['n_found'] for algorithm in self]
        for metric in self.internal_metrics:
            report[metric] = [self[algorithm][metric]['avg'] for algorithm in self]
            detailed_reports[metric] = pd.DataFrame(index=self.report)
            for detail in 'time', 'avg', 'median', 'std', 'min', 'max':
                detailed_reports[metric][detail] = [self[algorithm][metric][detail] for algorithm in self]
            detailed_reports[metric].to_csv(join(report_dir, 'detailed_metrics', f'{metric}.csv'))

        report.to_csv(join(report_dir, 'report.csv'))

        return report, detailed_reports

    def generate_goea_report(self, report_dir: str = 'report') -> pd.DataFrame:
        """
        Generate a report on GO Enrichment Analysis results - draw plots and tables.

        :param report_dir: Specify a directory, where the report results will be saved.
        :return: A dataframe with enrichment results for each algorithm.
        """

        if not hasattr(self, 'goea_time'):
            print('GO Enrichment Analysis has not been performed for this benchmark')
            print('See GeneExpressionBenchmark.perform_goea() method')
            return

        GeneExpressionBenchmark._check_dir(report_dir)
        self.plot_enrichment_rates(report_dir)

        report = pd.DataFrame(index=self.report)
        report['GO Enrichment time'] = [self[algorithm]['enrichment_time'] for algorithm in self]
        for alpha in self.alphas:
            report[f'a = {alpha}'] = [self[algorithm][f'a = {alpha}'] for algorithm in self]
        best_biclusters = [min(self[algorithm]['enriched_biclustering'], key=lambda r: r.p_uncorrected)
                           for algorithm in self]
        report['Most enriched term'] = [term.name for term in best_biclusters]
        report['p-value'] = [term.p_uncorrected for term in best_biclusters]
        report['Adjusted p-value'] = [term.p_fdr_bh for term in best_biclusters]

        report.to_csv(join(report_dir, 'goea_report.csv'))

        return report

    def plot_metrics(self, report_dir: str = 'report') -> None:
        """
        Draw and save plots for each metric and each algorithm

        :param report_dir: Specify a directory, where the report results will be saved.
        """

        GeneExpressionBenchmark._check_dir(join(report_dir, 'metrics'))
        if not hasattr(self, 'overall_execution_time'):
            print('Benchmark was never executed')
            print('See .run() method')
            return

        for i, metric in enumerate(self.internal_metrics):
            plt.bar(self, [self[algorithm][metric]['avg'] for algorithm in self],
                    yerr=[self[algorithm][metric]['std'] for algorithm in self])
            plt.title(metric)
            plt.ylim(0)

            plt.tight_layout()
            plt.savefig(join(report_dir, 'metrics', f'{metric}.png'))

    def plot_execution_times(self, report_dir: str = 'report') -> None:
        """
        Draw and save the plot of each algorithm's execution time

        :param report_dir: Specify a directory, where the report results will be saved.
        """

        if not hasattr(self, 'overall_execution_time'):
            print('Benchmark was never executed')
            print('See .run() method')
            return
        GeneExpressionBenchmark._check_dir(report_dir)

        plt.bar(self, [self[algorithm]['execution_time'] for algorithm in self])
        plt.title('Execution time')
        plt.ylabel('Time (sec)')

        plt.tight_layout()
        plt.savefig(join(report_dir, 'times.png'))

    def plot_enrichment_rates(self, report_dir: str = 'report') -> None:
        """
        Draw and save the plot of enrichment rates for each significance level

        :param report_dir: Specify a directory, where the report results will be saved.
        """

        GeneExpressionBenchmark._check_dir(report_dir)
        if not hasattr(self, 'alphas'):
            print('GO Enrichment Analysis has not been performed for this benchmark')
            print('See GeneExpressionBenchmark.perform_goea() method')
            return

        x = np.arange(len(self))
        bar_width = 1 / (len(self.alphas) + 1)
        for i in range(len(self.alphas)):
            y = [self[algorithm][f'a = {self.alphas[i]}'] for algorithm in self]
            plt.bar(x + bar_width * i, y, width=bar_width, label=f'a = {self.alphas[i]}')

        x_ticks = x + (bar_width * (len(self.alphas) - 1) / 2)
        plt.xticks(x_ticks, self)
        plt.title('GO Enrichment Analysis of resulting biclusters')
        plt.ylabel('Proportion of enriched biclusters for given significance level (a)')
        plt.ylim((0, 1.05))

        plt.legend()
        plt.savefig(join(report_dir, 'goea.png'))

    def plot_found(self, report_dir: str = 'report') -> None:
        """
        Draw and save the plot for number of biclusters found be each algorithm

        :param report_dir: Specify a directory, where the report results will be saved.
        """

        GeneExpressionBenchmark._check_dir(report_dir)
        if not hasattr(self, 'overall_execution_time'):
            print('Benchmark was never executed')
            print('See GeneExpressionBenchmark.run() method')
            return

        plt.bar(self, [self[algorithm]['n_found'] for algorithm in self])
        plt.title('Number of biclusters found by each algorithm')

        plt.tight_layout()
        plt.savefig(join(report_dir, 'found.png'))

    @staticmethod
    def _check_dir(dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
