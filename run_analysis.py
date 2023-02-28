from experiment_analysis import ExperimentAnalysis
import create_dummy_data

# DATA ###################################
df = create_dummy_data.create_dummy_data(total_n_visitors=360_000,
                                         conversion_rates=[0.05, 0.05, 0.06],
                                         revenue_per_order=[25.0, 25.0, 20.0]
                                         )
create_dummy_data.print_summary(df)

# VARIABLES ##############################
partition_col = 'test_cell'  # can choose from drop down
control_cell = 'C'  # can choose from drop down
visitor_col = 'visitor_id'  # can choose from drop down
confidence_level = 90  # check > 50 & < 100 from drop down
n_tail_test = 2  # check 1 or 2 from drop down
which_tail = 'upper'  # check if upper or lower for one tail test or None for two-tail from dropdown
numerator_col = 'revenue'  # can choose from drop down
denominator_col = None  # None 'order' can choose from drop down

ea = ExperimentAnalysis(data=df,
                        partition_column=partition_col,
                        control_cell=control_cell,
                        visitor_column=visitor_col,
                        numerator_column=numerator_col,
                        denominator_column=denominator_col,
                        confidence_level=confidence_level,
                        n_tail_test=n_tail_test,
                        which_tail=which_tail
                        )
ea.remove_hoppers()
ea.get_values()
ea.create_samples(n=10 ** 4)
ea.create_results_distribution()
ea.print_results()
