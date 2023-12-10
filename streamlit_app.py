import streamlit as st

st.set_page_config(
    page_title="ALGO",
)

st.write("# ALGO")

st.sidebar.success("Select an app above.")

left_co, right_co = st.columns(2)
with left_co:
    st.image('algo.png', caption='ALGO', width=300)
with right_co:
    st.markdown( """ 
        ## Meet ALGO
"""
)