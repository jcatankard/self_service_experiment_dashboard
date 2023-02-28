import samples

from typing import List, Tuple, Dict
import plotly.figure_factory as ff
from numpy.typing import NDArray
import polars as pl
import numpy as np


class ExperimentAnalysis:
    """class to help us analysis A/B/n experiment data with non-parametric analysis"""

    def __init__(self,
                 data: pl.DataFrame,
                 partition_column: str,
                 control_cell: str,
                 visitor_column: str,
                 numerator_column: str,
                 denominator_column: str = None,
                 confidence_level: float = 90,
                 n_tail_test: int = 2,
                 which_tail: str = 'upper'
                 ):
        """parameters -
        :param data: the data set that contains visitor level data, experiment partition and metrics to test for
        :param partition_column: which column identifies the experimentation partition to which a visitor belongs
        :param control_cell: which is the control cell within hte partition_column
        :param visitor_column: the column in the data set that is used to identify visitor by ID
        :param numerator_column: such as "orders" for conversion rate or "revenue" for revenue per order
        :param denominator_column: such as "order" for revenue per order or None if for a visitor metric such as CR%
        :param confidence_level: the statistical level of confidence we are testing for
        :param n_tail_test: number of tails: 1 or 2
        :param which_tail: upper or lower bound for a one tail test
        """
        self.data = data
        self.partition_column = partition_column
        self.control_cell = control_cell
        self.visitor_column = visitor_column
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self._check_denominator()
        self._groupby_visitor()
        self.confidence_level = self._check_confidence_level(confidence_level)
        self.n_tail_test = self._check_n_tails(n_tail_test)
        self.which_tail = self._check_which(which_tail)
        self.percentiles = self._create_significance_percentiles()

        self.test_cells: List[str] = []
        self.control_values: Tuple[NDArray[float], NDArray[float]] = (np.array([], dtype=np.float64),
                                                                      np.array([], dtype=np.float64))
        self.test_values: Dict[str, Tuple[NDArray[float], NDArray[float]]] = {}
        self.hopper_count: int = 0
        self.control_samples: NDArray[float] = np.array([], dtype=np.float64)
        self.test_samples: Dict[str, NDArray[float]] = {}
        self.results_distribution: Dict[str, NDArray[float]] = {}

    @staticmethod
    def _check_n_tails(n: int):
        if n not in [1, 2]:
            raise ValueError('n_tails should be either 1 for a one tail test or 2 for a two tail test')
        return n

    @staticmethod
    def _check_confidence_level(cl: float):
        if (cl <= 50) | (cl >= 100):
            raise ValueError('confidence intervals should be between 50 and 100 such as 90., 95., 97.5 or 99.')
        return cl

    def _check_which(self, which_tail: str):
        if (self.n_tail_test == 1) & (which_tail not in ['upper', 'lower']):
            raise ValueError('which_tail should be one of "upper" or "lower" in a one tail test')
        return which_tail

    def _check_denominator(self):
        """if no denominator_col, create a col == 1 to allow us to divide by something"""
        if self.denominator_column is None:
            self.denominator_column = 'denominator'
            self.data = self.data.with_columns([pl.lit(1).alias(self.denominator_column)])

    def _create_significance_percentiles(self) -> List[float]:
        if self.n_tail_test == 2:
            p_lower = (100 - self.confidence_level) / 2
            percentiles = [p_lower, 50, 100 - p_lower]
        else:
            percentiles = [self.confidence_level] if self.which_tail == 'upper' else [100 - self.confidence_level]
            percentiles.append(50)
        return sorted(percentiles)

    def _groupby_visitor(self):
        """group by visitors as they are our unit of randomization"""
        self.data = (self.data
                     .groupby([self.visitor_column, self.partition_column])
                     .sum()
                     )

    def remove_hoppers(self):
        self.data = (self.data
                     .with_columns([pl.col(self.partition_column).unique_counts().over(self.visitor_column)
                                   .alias('n_partitions_per_visitor')])
                     )

        hoppers = (self.data
                   .filter(pl.col('n_partitions_per_visitor') > 1)
                   .select(self.visitor_column)
                   .unique(subset=self.visitor_column)
                   )
        self.hopper_count = hoppers.to_numpy().size

        self.data = (self.data
                     .filter(pl.col('n_partitions_per_visitor') == 1)
                     .drop('n_partitions_per_visitor')
                     )

    def get_values(self):
        self.control_values = self._get_values_by_partition(self.control_cell)
        self._get_test_cells()
        self.test_values = {}
        for t in self.test_cells:
            values = self._get_values_by_partition(t)
            self.test_values[t] = (values[0], values[1])

    def _get_values_by_partition(self, partition: str) -> Tuple[NDArray[float], NDArray[float]]:
        values = (self.data
                  .filter(pl.col(self.partition_column) == partition)
                  .select(self.numerator_column, self.denominator_column)
                  )
        values = values.to_numpy().T.astype(dtype=np.float64)
        return values[0], values[1]

    def _get_test_cells(self):
        test_cells = (self.data
                      .select(self.partition_column)
                      .filter(pl.col(self.partition_column) != self.control_cell)
                      .unique()
                      )
        self.test_cells = test_cells.to_numpy()[:, 0]

    def create_samples(self, n: int = 1000):
        self.control_samples = samples.create_randomized(self.control_values[0], self.control_values[1], n)
        self.test_samples = {t: samples.create_randomized(v[0], v[1], n) for t, v in self.test_values.items()}

    def create_results_distribution(self):
        check_values = self.control_samples != 0
        self.results_distribution = {}
        for t, v in self.test_samples.items():
            if check_values.size > 0:
                self.results_distribution[t] = v[check_values] / self.control_samples[check_values] - 1
            else:
                self.results_distribution[t] = np.array([], dtype=np.float64)

    def summarize_results(self):
        results = {}
        for t in self.test_cells:
            # results[str(t)] = {p: np.percentile(self.results_distribution[t], p).round(4) for p in self.percentiles}
            results[str(t)] = np.percentile(self.results_distribution[t], self.percentiles).round(4).tolist()

        results = (pl.DataFrame(results)
                   .to_pandas()
                   )
        results.index = self.percentiles
        results = results.T
        results['significant?'] = results.apply(lambda z: z[self.percentiles[0]] / z[self.percentiles[0]] > 0,
                                                         axis=1)
        results[self.percentiles] = results[self.percentiles].applymap(lambda x: format(x, '.2%'))
        return results

    def create_bar_chart_data(self, metric):
        bar_data = (self.data
                    .groupby([self.partition_column])
                    .sum()
                    .with_columns([(pl.col(self.numerator_column) / pl.col(self.denominator_column)).alias(metric)])
                    .select(self.partition_column, metric)
                    )
        return bar_data.to_pandas()

    def plot_values_histogram(self, metric):
        c_mask = self.control_values[1] != 0
        c_values = self.control_values[0][c_mask] / self.control_values[1][c_mask]
        min_val, max_val = np.min(c_values), np.max(c_values)
        hist_data = [c_values]
        group_labels = [self.control_cell]
        for t in self.test_cells:
            t_mask = self.test_values[t][1] != 0
            t_values = self.test_values[t][0][t_mask] / self.test_values[t][1][t_mask]
            min_val = min_val if np.min(t_values) > min_val else np.min(t_values)
            max_val = max_val if np.max(t_values) < max_val else np.max(t_values)
            hist_data.append(t_values)
            group_labels.append(t)

        bin_size = (max_val - min_val) / 50
        fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size, show_rug=False)
        fig.update_layout(title=f'Distribution of {metric}')
        fig.update_xaxes({'title': metric})
        return fig

    def plot_results_histogram(self, metric):
        min_val, max_val = 0, 0
        hist_data = []
        group_labels = []
        for t in self.test_cells:
            t_values = self.results_distribution[t]
            min_val = min_val if np.min(t_values) > min_val else np.min(t_values)
            max_val = max_val if np.max(t_values) < max_val else np.max(t_values)
            hist_data.append(t_values)
            group_labels.append(t)

        bin_size = (max_val - min_val) / 200
        fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size, show_rug=False)
        fig.update_layout(title=f'Distribution of means: {metric}', xaxis_tickformat='.1%')
        fig.update_xaxes({'title': f'change vs {self.control_cell}'})
        return fig
