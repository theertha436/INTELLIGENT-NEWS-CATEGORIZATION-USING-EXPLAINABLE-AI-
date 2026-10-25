import streamlit as st

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Streamlit Test")
st.success("Streamlit UI is working correctly!")

st.write("If you can see this page, Streamlit itself is working.")
st.write("The problem is inside app.py.")