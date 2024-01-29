import pandas as pd
import streamlit as st
import numpy as np
import json

@st.cache_resource
def load_raw(filename='../Simplified Dataset.xlsx'):
    data = pd.ExcelFile(filename)
    return data

@st.cache_data
def load_data(cases_sheet, filename='../Simplified Dataset.xlsx'):
    with open(filename, 'rb') as f:
        data = pd.read_excel(f, sheet_name=cases_sheet)
        data = data.fillna('None')
    return data


raw = load_raw()

st.header('Cases sheet')
cases = st.selectbox('Cases Sheet', raw.sheet_names, key='cases_sheet')
#variables = st.selectbox('Variables', data.sheet_names, key='variable_sheet')

data = load_data(st.session_state.cases_sheet)
for col in data.columns:
    if f'{col}_keep' not in st.session_state:
        st.session_state[f'{col}_keep'] = True
    if f'{col}_type' not in st.session_state:
        st.session_state[f'{col}_type'] = 'Categorical'
    if f'{col}_grouped' not in st.session_state:
        st.session_state[f'{col}_grouped'] = False
    if f'{col}_num_groups' not in st.session_state:
        st.session_state[f'{col}_num_groups'] = 1


st.header('View Data Sample')
st.selectbox('Column', data.columns, key='selected_columns')
col_data = data[st.session_state['selected_columns']]
with st.expander('Data view...'):
    st.write(col_data)
    # for x in list(set(map(str, col_data.values))):
    #     st.write(x, list(map(str, col_data.values)).count(x))

# st.write(st.session_state)

# st.header('Choose Columns')
# use_column_checkboxes = {}
# with st.expander('Columns'):
#     for col in data.columns:
#         use_column_checkboxes[col] = st.checkbox(col, key=f'{col}_keep')

st.header('Columns')
for col in data.columns:
    st.toggle(f'Keep `{col}`?', value=True, key=f'{col}_keep')
    if st.session_state[f'{col}_keep']:
        st.subheader(col)
        # def drop_col():
        #     st.session_state[f'{col}_keep'] = False
        #     use_column_checkboxes[col] = False
        # st.button("Don't use column", on_click=drop_col, key=f'drop-{col}')
        col_data = data[col]
        distinct = list(set(map(str, col_data.values)))
        if st.session_state[f'{col}_keep']:
            with st.expander('raw', expanded=False):
                for x in list(set(map(str, col_data.values)))[:5]:
                    st.write(x, list(map(str, col_data.values)).count(x))
                # st.write(col_data[:5])
            with st.expander(col, expanded=True):
                st.radio('Type?', ['Categorical', 'Binned'], key=f'{col}_type')
                if st.session_state[f'{col}_type'] == 'Categorical':
                    st.multiselect('Classes', distinct, default=distinct, key=f'{col}_classes')
                    st.multiselect('Empty', distinct, key=f'{col}_empty')
                    st.selectbox('What to do with unmatched values?', ['warn', 'keep', 'drop', 'random', 'major'], key=f'{col}_unmatched')
                if st.session_state[f'{col}_type'] == 'Binned':
                    st.toggle('Default order', value=True, key=f'{col}-logical_order')
                    if not st.session_state[f'{col}-logical_order']:
                        for i, v in enumerate(distinct):
                            st.number_input(v, value=i+1, min_value=1, max_value=len(distinct), step=1, key=f'{col}-order-{v}')
            with st.expander('Extra Options'):
                st.toggle('Use for training?', value=True, key=f'{col}_train_with')
                st.toggle('Ask?', value=False, key=f'{col}_ask')
                st.toggle('Try to predict?', value=False, key=f'{col}_target')
                st.checkbox('Allow grouping?', value=False, key=f'{col}_grouped')
                if f'{col}_grouped' in st.session_state and st.session_state[f'{col}_grouped']:
                    def increment(col):
                        def f():
                            n = st.session_state[f'{col}_num_groups']
                            st.session_state[f'{col}_num_groups'] = n + 1
                        return f
                    st.button('add group', on_click=increment(col), key=f'{col}-add-group')
                    for i in range(st.session_state[f'{col}_num_groups']):
                        st.multiselect(f'Group {i}', distinct, key=f'{col}-group{i}')
        st.divider()

with open('config.json', 'w') as f:
    json.dump(st.session_state.to_dict(), f, indent=4)
