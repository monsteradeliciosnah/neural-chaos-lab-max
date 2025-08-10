import streamlit as st

from neural_chaos_lab_max.plotting import plot_attractor
from neural_chaos_lab_max.systems import lorenz, rossler

st.set_page_config(page_title="Neural Chaos Lab Max", layout="wide")
st.title("Neural Chaos Lab Max")

system = st.selectbox("System", ["lorenz", "rossler"])
n = st.slider("n", 1000, 10000, 5000, 500)
if st.button("Generate"):
    series = lorenz(n) if system == "lorenz" else rossler(n)
    fig = plot_attractor(series, f"{system} ({n})")
    st.plotly_chart(fig, use_container_width=True)
