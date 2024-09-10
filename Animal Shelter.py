import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

title_container = st.container(border=True)
# st.title('Animal Shelter')
# st.text('-Siddharth Ranganatha & Vismay BS')
# title_container.title('Animal Shelter')
title_container.title(':blue[_Animal Shelter_]   :dog2:')
title_container.text('-Siddharth Ranganatha & Vismay BS')

engine = create_engine(
    '''postgresql+psycopg2://postgres:postgres@localhost:5432/Animal Shelter''')


def read_data(cmd, con=engine):
    data = pd.read_sql(cmd, con)
    con.dispose()
    return data

def check_data(df1, df2):
    return True if df1.equals(df2) else False

st.sidebar.markdown('##### :blue[Choose to view or update data:]')
on = st.sidebar.toggle('Update')
st.sidebar.divider()

if not on:
    st.sidebar.markdown("# Filters")
    database = st.sidebar.selectbox(
        'Database Selection',
        ('Animals', 'Adoptions', 'Adopters', 'Donations', 'Staff', 'Volunteers'))
    if database == 'Animals':
        cmd = "SELECT * from public.animals"
        data = read_data(cmd)
        selection = list(set(data.species.values))
        selection.sort()
        animal_selection = st.sidebar.selectbox(
            'Animal Selection',
            selection)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Filtered data:')
        filter_container.write(data.loc[data.species == animal_selection])
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    elif database == 'Adoptions':
        cmd = "SELECT * from public.adoptions"
        data = read_data(cmd)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    elif database == 'Adopters':
        cmd = "SELECT * from public.adopters"
        data = read_data(cmd)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    elif database == 'Donations':
        cmd = "SELECT * from public.donations"
        data = read_data(cmd)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    elif database == 'Staff':
        cmd = "SELECT * from public.staff"
        data = read_data(cmd)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    elif database == 'Volunteers':
        cmd = "SELECT * from public.volunteers"
        data = read_data(cmd)
        filter_container = st.container(border=True)
        filter_container.empty()
        filter_container.markdown('#### Complete Database:')
        filter_container.write(data)

    else:
        st.write('Error')
else:
    cmd = "SELECT * from public.animals"
    data = read_data(cmd)
    df1 = data.copy()
    data_container = st.container(border=True)
    df = data_container.data_editor(data, num_rows='dynamic')
    button_container = st.container()
    with button_container:
        submit = st.button('Submit')
        if submit:
            if check_data(df, df1) == True:
                # st.error('No updates!', icon="⚠️")
                st.toast('No updates!', icon="⚠️")
            else:
                st.balloons()
                # st.success('Data updated successfully!', icon="✅")
                st.toast('Data updated successfully!', icon='😍')
           
