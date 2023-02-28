from numpy.typing import NDArray
from typing import List, Dict
import polars as pl
import numpy as np


def create_orders(conversion_rate: float, total_n_visitors: int) -> NDArray[int]:
    return np.random.binomial(1, conversion_rate, total_n_visitors)


def create_revenue(revenue_per_order: float, total_n_visitors: int) -> NDArray[float]:
    return np.random.chisquare(revenue_per_order, total_n_visitors)


def combine_orders_revenue(orders: NDArray[int], revenue: NDArray[float]) -> NDArray[float]:
    return orders * revenue


def create_partition_mapping(n_partitions: int) -> dict:
    mapping = {0: 'C'}
    n_test_cells = n_partitions - 1
    for p in range(n_test_cells):
        mapping[p + 1] = 'T' + str(p + 1)
    return mapping


def create_partitions(n_partitions: int, total_n_visitors: int, partition_mapping: Dict[int, str]) -> NDArray[str]:
    randomize = np.random.randint(0, n_partitions, total_n_visitors)
    map_to_partition = list(map(lambda x: partition_mapping[x], randomize))
    return np.array(map_to_partition)


def create_dataframe(total_n_visitors: int, orders: NDArray[int], revenue: NDArray[float],
                     partitions: NDArray[str]) -> pl.DataFrame:
    df = pl.DataFrame({'visitor_id': np.arange(total_n_visitors),
                       'order': orders, 'revenue': revenue, 'test_cell': partitions
                       })
    df = df.with_columns([pl.concat_str([pl.lit('A'), pl.col('visitor_id')]).alias('visitor_id')])
    return df


def create_dummy_data(total_n_visitors: int, conversion_rates: List[float],
                      revenue_per_order: List[float]) -> pl.DataFrame:

    n_partitions = len(conversion_rates)
    partition_mapping = create_partition_mapping(n_partitions)
    partitions = create_partitions(n_partitions, total_n_visitors, partition_mapping)

    # set orders and revenue arrays
    orders = np.zeros((total_n_visitors,), dtype=np.float64)
    revenue = np.zeros((total_n_visitors,), dtype=np.float64)

    # add orders and revenue for each partition
    for p in range(n_partitions):
        o = create_orders(conversion_rates[p], total_n_visitors)
        r = create_revenue(revenue_per_order[p], total_n_visitors)
        r = combine_orders_revenue(o, r)

        orders = np.where(partitions == partition_mapping[p], o, orders)
        revenue = np.where(partitions == partition_mapping[p], r, revenue)

    return create_dataframe(total_n_visitors, orders, revenue, partitions)


def print_summary(x: pl.DataFrame):
    summary = (x
               .groupby('test_cell')
               .mean()
               .with_columns([(pl.col('revenue') / pl.col('order')).alias('revenue_per_order')])
               )
    print(summary)
