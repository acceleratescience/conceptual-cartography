import streamlit as st
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

model_name = 'BAAI_bge-base-en-v1.5'

path = PROJECT_ROOT / 'output' / model_name /'window_None' / 'theory'

metrics_path = path / 'metrics/metrics_layer-10.pt'
landscape_path = path / 'landscapes/landscape_layer-10.pt'
contexts_path = path / 'contexts.txt'

# Load metrics
metrics = torch.load(metrics_path, weights_only=False)
landscape = torch.load(landscape_path, weights_only=False)
with open(contexts_path, 'r', encoding='utf-8') as f:
    context = [line.strip() for line in f if line.strip()]



st.title(f"Visualizing Landscapes for {model_name}")
st.subheader("Metrics")
st.write(metrics)

st.subheader("Landscape")
# plotting the landscape
st.plotly_chart(landscape.create_plotly_landscape(context, 'theory', width=50))