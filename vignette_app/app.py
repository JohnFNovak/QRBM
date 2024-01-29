import streamlit as st
import pandas as pd
import numpy as np
import time

DEBUG = False

def config():
    data_file = st.file_uploader("Upload and Excel file for this session", type='xlsx')

    while True:
        if data_file is not None:
            data = pd.read_excel(data_file, sheet_name='Cases-Use')
            data = data.fillna('None')
            variables = pd.read_excel(data_file, sheet_name='Variables')
            st.session_state.data = data
            st.session_state.variables = variables
            break


@st.cache_data
def load_data(filename='../Simplified Dataset.xlsx'):
    with open(filename, 'rb') as f:
        data = pd.read_excel(f, sheet_name='Cases-Use')
        data = data.fillna('None')
        variables = pd.read_excel(f, sheet_name='Variables')
    # st.session_state.data = data
    # st.session_state.variables = variables
    return data, variables

@st.cache_data
def vignette_text_prep(variables):
    vignette_strings = {}
    vignette_order = []
    for i in range(len(variables)):
        r = variables.xs(i)
        if isinstance(r['Format'], str):
            vignette_order.append(r['Variable'])
            format_string = r['Format'].replace('{value}', '{'+r['Variable']+'}').replace('{variable}', r['Variable'])
            if isinstance(r['Condition'], str):
                condition = r['Condition'].replace('{value}', '{'+r['Variable']+'}')
            else:
                condition = 'True'
            vignette_strings[r['Variable']] = (r['Vignette'], format_string, condition)

    # st.session_state.vignette_strings = vignette_strings
    # st.session_state.vignette_order = vignette_order
    return vignette_strings, vignette_order


def show_vignette(indx, data, vignette_strings, vignette_order):
    entry = data.xs(indx)
    entry_dict = entry.to_dict()
    # st.write(entry_dict)

    # TODO: in the future, these scores should come from a model
    # TODO: `vignette_order` is the wrong thing to be looping over.
    # Some columns are skipped because they are combined with others (eg:`Sex`)
    scores = {v: np.random.randint(0, 10) for v in vignette_order}
    score_map = {9: '#33691E',
                8: '#558B2F',
                7: '#689F38',
                6: '#7CB342',
                5: '#8BC34A',
                4: '#9CCC65',
                3: '#AED581',
                2: '#C5E1A5',
                1: '#DCEDC8',
                0: '#FFFFFF'}

    options = {}
    for k, v in vignette_strings.items():
        if v[0] == 'ask':
            options[k] = list(set(data[k].values))

    text = ''
    for v in vignette_order:
        config = vignette_strings[v]
        # st.write(config[1])
        # st.write(config[2])
        # st.write(config[2].format(**entry_dict))
        if config[0] == 'show':
            # NOTE: we are not doing *any* sterilization. THIS IS DANGEROUS
            cond = eval(config[2].format(**entry_dict))
            if cond:
                text_string = config[1]
                if st.session_state.ai_assistance:
                    # For if we hightlight just the values
                    if v in scores:
                        text_string = text_string.replace('{'+v+'}', '<span style="background-color: ' + score_map[scores[v]] +'">{' + v + "}</span>")
                    text_string = text_string.format(**entry_dict)#.replace('>', '\>').replace('<', '\<')

                    # For if we highlight the whole string:
                    # text_string = text_string.format(**entry_dict)#.replace('>', '\>').replace('<', '\<')
                    # if v in scores:
                    #     text_string = f'<span style="background-color: {score_map[scores[v]]}">' + text_string + "</span>"
                else:
                    text_string = text_string.format(**entry_dict)
                text += text_string + ' '

    # st.write(text)
    start = time.clock_gettime(0)
    st.components.v1.html(text, height=275, scrolling=True)

    with st.form("diagnosis_form"):
        st.write("Diagnosis and next steps")
        questions = {}
        def callback():
            st.session_state['stage'] = 'feedback'
            st.session_state['duration'] = time.clock_gettime(0) - start
        for v in vignette_order:
                config = vignette_strings[v]
                if config[0] == 'ask':
                    # TODO: options should be populated from the excel file
                    question = st.selectbox(config[1].format(**entry_dict), options[v], key=v)
                    questions[v] = question
        button = st.form_submit_button('Submit my picks', on_click=callback)


def feedback(indx):
    def callback():
        st.session_state['stage'] = 'diagnose'
    with st.form("feedback_form"):
        st.write("Feedback")
        confidence = st.radio('Confidence', ['Not confident at all', 'Less than confident', 'Neutral', 'Somewhat confident', 'Very confident'], key='confidence', index=2)
        ai_feedback = st.radio('Do you think the AI assistance was helpful?', ['Actively not helpful', 'Not very helpful', 'Neutral', 'Somewhat helpful', 'Very helpful'], key='ai_feedback', index=2)
        st.form_submit_button('Submit feedback', on_click=callback)


def login():
    def callback():
        st.session_state['stage'] = 'diagnose'
        # NOTE: this is a hack because session state, tied to widgets, gets removed with the widget (╯°□°）╯︵ ┻━┻
        st.session_state['name'] = st.session_state['_name']
        st.session_state['ai_assistance'] = st.session_state['_ai_assistance']
    with st.form("login_form"):
        st.write("Login")
        st.text_input('Your Name', key='_name')
        st.toggle('AI assistance enabled', value=True, key='_ai_assistance')
        st.form_submit_button('Start', on_click=callback)


# TODO: Figure out how to let the admin select the data when the app starts
# config()
data, variables = load_data()
vignette_strings, vignette_order = vignette_text_prep(variables)


if 'stage' not in st.session_state:
    st.session_state['stage'] = 'login'

if st.session_state['stage'] == 'login':
    login()
elif st.session_state['stage'] == 'diagnose':
    indx = np.random.randint(0, len(data))
    st.session_state['indx'] = indx
    if DEBUG:
        st.write(st.session_state)
    show_vignette(indx, data, vignette_strings, vignette_order)
elif st.session_state['stage'] == 'feedback':
    if DEBUG:
        st.write(st.session_state)
    feedback(st.session_state['indx'])

if 'name' in st.session_state:
    st.write(f'Currently logged in as: {st.session_state.name}')