from create_dummy_data import create_dummy_data
from experiment_analysis import ExperimentAnalysis
# https://docs.streamlit.io/library/cheatsheet
import streamlit as st
import polars as pl
import numpy as np
import logging


@st.cache_resource
def load_dummy_data(total_n_visitors, conversion_rates_input, revenue_per_order_input):
    data = create_dummy_data(total_n_visitors=total_n_visitors,
                             conversion_rates=conversion_rates_input.tolist(),
                             revenue_per_order=revenue_per_order_input.tolist()
                             )
    return data.select('test_cell', 'visitor_id', 'revenue', 'order')


@st.cache_resource
def load_data_from_file(file_object, file_type_input):
    try:
        data = pl.read_csv(file_object) if file_type_input == 'csv' else pl.read_parquet(file_object)
        st.success(f'{file_type_input} uploaded successfully')
        return data
    except TypeError:
        st.warning(f'Select a {file_type_input} file type')
        st.stop()
        return None


st.set_page_config(page_icon='👾', page_title='Experiment analysis')
st.title('Experiment analysis: self-service')
st.write('This app makes it easy to run self-service analysis for your a/b/n experiment using bootstrap resampling.')

with st.sidebar:
    data_source = st.selectbox('Pick where your data is from:', ['use your own data', 'use dummy data'])
    if data_source == 'use dummy data':
        st.markdown('This option will create randomized data based on the parameters below.')
        n_visitors = st.number_input('Pick the total number of visitors', 1_000, 1_000_000)

        n_cells = st.slider('Pick the number of test cells (inc. control)', 2, 5)
        conversion_rates = np.zeros(n_cells, dtype=np.float64)
        r_per_o = np.zeros(n_cells, dtype=np.float64)

        for c in range(n_cells):
            name = 'Control' if c == 0 else 'Test' + str(c)
            st.markdown(f'<h3>{name} cell</h3>', unsafe_allow_html=True)

            cr = st.slider(f'Choose average conversion rate percentage for {name}:', 0.0, 100.0, 5.5)
            conversion_rates[c] = cr / 100

            rpo = st.slider(f'Choose average revenue per order for {name}:', 0.0, 250.0, 25.0)
            r_per_o[c] = rpo

        logging.info(f'conversion rates: {conversion_rates}')
        logging.info(f'revenue per order:{r_per_o}')

        st.markdown("Revenue per order follows a <a href='https://en.wikipedia.org/wiki/Chi-squared_distribution' "
                    "target='_blank'>chi-squared distribution</a>.", unsafe_allow_html=True)

        df = load_dummy_data(n_visitors, conversion_rates, r_per_o)
        df.write_csv('test.csv')
        df.write_parquet('test.parquet')

    else:
        file_type = st.selectbox('Select file type:', ['csv', 'parquet'])
        st.markdown(f'You will need columns for "visitor id" or similar, "experiment tag" and metrics of interest')
        file = st.file_uploader(f'Upload a {file_type} file', type=[file_type])
        df = load_data_from_file(file, file_type)

    if 'df' not in locals():
        logging.info('stopping due to no dataframe found')
        st.stop()
    else:
        st.dataframe(df.to_pandas())

col1, col2 = st.columns(2)
selected_cols = []
with col1:
    with st.expander('SELECT EXPERIMENT PARTITIONS AND VISITORS'):
        partition_col = st.selectbox('Pick experiment partition column', df.columns)
        selected_cols.append(partition_col)
        logging.info(f'partition column: {partition_col}')

        control_cell = st.selectbox('Pick control partition', np.unique(df[partition_col].to_numpy()))
        logging.info(f'control partition: {control_cell}')

        visitor_col = st.selectbox('Pick visitor id column', [c for c in df.columns if c not in selected_cols])
        selected_cols.append(visitor_col)
        logging.info(f'visitor id column: {visitor_col}')

with col2:
    with st.expander('SELECT EXPERIMENT PARAMETERS'):
        confidence_level = st.number_input(f'Choose a confidence level:', 50.0, 99.9, 90.0)
        logging.info(f'confidence level: {confidence_level}')

        n_tail_test = st.radio('number of tails to test for:', [1, 2], index=1)
        logging.info(f'number of tails: {n_tail_test}')

        which_tail = None
        if n_tail_test == 1:
            which_tail = st.radio('which tail?:', ['upper', 'lower'], index=0)
            logging.info(f'which tail: {which_tail}')

with st.expander("SELECT METRIC FOR ANALYSIS"):
    st.markdown('Choose from what values your metric is calculated.')
    st.markdown('For example, if it is conversion rate: choose orders and visitors.')
    st.markdown('If it is revenue per visitor: choose revenue and visitors.')
    st.markdown('If it is average order value: choose revenue and orders.')

    numerator_col = st.selectbox('Pick numerator ', [c for c in df.columns if c not in selected_cols])
    selected_cols.append(numerator_col)
    logging.info(f'numerator column: {numerator_col}')

    d = st.selectbox('Pick denominator', ['visitor'] + [c for c in df.columns if c not in selected_cols])
    denominator_col = None if d == 'visitor' else d
    logging.info(f'denominator column: {denominator_col}')

    metric = f'{numerator_col} per {d}'
    st.markdown(f'Metric for analysis: <b>{metric}</b>', unsafe_allow_html=True)

# CREATE OBJECT FOR EXPERIMENT ANALYSIS
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

# PLOT BAR CHART
st.markdown(f'<h3>{metric}</h3>', unsafe_allow_html=True)
bar_data = ea.create_bar_chart_data(metric)
st.bar_chart(bar_data, x=ea.partition_column)

# PLOT VALUES DISTRIBUTION
fig = ea.plot_values_histogram(metric)
fig.update_layout(title=f'Distribution of {metric}')
fig.update_xaxes({'title': metric})
st.plotly_chart(fig, use_container_width=True)

st.markdown('<h3>Experiment analysis</h3>', unsafe_allow_html=True)
n_samples = st.selectbox('Choose number of bootstrap resamples', (10 ** p for p in range(3, 7)))
if st.button('Run analysis'):

    with st.spinner('Computing results...'):
        ea.create_samples(n=n_samples)
        ea.create_results_distribution()
        results = ea.summarize_results()
        st.markdown('<h3>Results by percentile</h3>', unsafe_allow_html=True)
        st.table(results)

        fig = ea.plot_results_histogram(metric)
        st.plotly_chart(fig, use_container_width=True)
