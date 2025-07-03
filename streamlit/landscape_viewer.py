"""Streamlit app for viewing conceptual landscapes."""

import streamlit as st
import torch
from pathlib import Path
import sys
import re

# THis is probably wrong, but avoids warnings for some reason...
# TODO: Why?
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis import Landscape
from src.visualization import LandscapeVisualizer


def get_available_layers(path):
    """Find all available layer numbers from the metrics and landscapes directories."""
    metrics_dir = path / 'metrics'
    landscapes_dir = path / 'landscapes'
    
    layers = set()
    
    # Check metrics directory
    if metrics_dir.exists():
        for file in metrics_dir.glob('metrics_layer-*.pt'):
            match = re.search(r'layer-(\d+)\.pt$', file.name)
            if match:
                layers.add(int(match.group(1)))
    
    # Check landscapes directory  
    if landscapes_dir.exists():
        for file in landscapes_dir.glob('landscape_layer-*.pt'):
            match = re.search(r'layer-(\d+)\.pt$', file.name)
            if match:
                layers.add(int(match.group(1)))
    
    return sorted(list(layers))


def main():
    """Main Streamlit app for visualizing landscapes and metrics."""
    # Get path and target word from command line arguments
    if len(sys.argv) != 3:
        st.error("Usage: streamlit run landscape_viewer.py <path_to_directory> <target_word>")
        st.stop()
    
    path_str = sys.argv[1]
    target_word = sys.argv[2]
    path = Path(path_str)
    
    if not path.exists():
        st.error(f"Path does not exist: {path}")
        st.stop()
    
    if not path.is_dir():
        st.error(f"Path is not a directory: {path}")
        st.stop()

    st.title(f"Conceptual Landscapes: {path.name}")
    st.markdown(f"**Target word:** `{target_word}`")
    
    # Get available layers
    available_layers = get_available_layers(path)
    
    if not available_layers:
        st.error("No layer files found in the specified directory")
        st.stop()
    
    # Initialize session state for current layer
    if 'current_layer' not in st.session_state:
        st.session_state.current_layer = available_layers[0]
    
    # Create navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous Layer", disabled=(st.session_state.current_layer == available_layers[0])):
            current_idx = available_layers.index(st.session_state.current_layer)
            if current_idx > 0:
                st.session_state.current_layer = available_layers[current_idx - 1]
                st.rerun()
    
    with col2:
        st.markdown(f"**Layer {st.session_state.current_layer}** ({available_layers.index(st.session_state.current_layer) + 1} of {len(available_layers)})")
    
    with col3:
        if st.button("Next Layer →", disabled=(st.session_state.current_layer == available_layers[-1])):
            current_idx = available_layers.index(st.session_state.current_layer)
            if current_idx < len(available_layers) - 1:
                st.session_state.current_layer = available_layers[current_idx + 1]
                st.rerun()
    
    # Optional: Add a selectbox for direct layer selection
    with st.expander("Jump to specific layer"):
        selected_layer = st.selectbox(
            "Select layer:", 
            available_layers, 
            index=available_layers.index(st.session_state.current_layer),
            key="layer_selector"
        )
        if selected_layer != st.session_state.current_layer:
            st.session_state.current_layer = selected_layer
            st.rerun()

    # Load data for current layer
    current_layer = st.session_state.current_layer
    metrics_path = path / f'metrics/metrics_layer-{current_layer}.pt'
    landscape_path = path / f'landscapes/landscape_layer-{current_layer}.pt'
    contexts_path = path / 'contexts.txt'

    try:
        # Load contexts (assuming this is the same for all layers)
        with open(contexts_path, 'r', encoding='utf-8') as f:
            context = [line.strip() for line in f if line.strip()]
        
        # Load metrics if available
        metrics = None
        if metrics_path.exists():
            metrics = torch.load(metrics_path, weights_only=False)
        
        # Load landscape if available
        landscape = None
        if landscape_path.exists():
            landscape = torch.load(landscape_path, weights_only=False)
            
    except Exception as e:
        st.error(f"Error loading data for layer {current_layer}: {e}")
        st.stop()

    # Display landscape
    if landscape is not None:
        st.subheader(f"Landscape - Layer {current_layer}")
        try:
            fig = LandscapeVisualizer.create_plotly_figure(landscape, context, target_word, width=50)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating landscape plot: {e}")
    else:
        st.warning(f"No landscape data found for layer {current_layer}")

    # Display metrics
    if metrics is not None:
        st.subheader(f"Metrics - Layer {current_layer}")
        
        # Display metrics in a nice format
        if hasattr(metrics, 'mev'):
            # New MetricsResult format
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("MEV", f"{metrics.mev:.4f}")
                st.metric("Average Similarity", f"{metrics.average_similarity:.4f}")
                st.metric("Similarity Std", f"{metrics.similarity_std:.4f}")
            
            with col2:
                if metrics.intra_similarity is not None:
                    st.metric("Intra-cluster Similarity", f"{metrics.intra_similarity:.4f}")
                if metrics.inter_similarity is not None:
                    st.metric("Inter-cluster Similarity", f"{metrics.inter_similarity:.4f}")
        else:
            # Legacy dict format
            st.write(metrics)
    else:
        st.warning(f"No metrics data found for layer {current_layer}")
    
    # Show available files info
    with st.expander("Available files info"):
        st.write(f"**Available layers:** {', '.join(map(str, available_layers))}")
        st.write(f"**Metrics file:** {'✓' if metrics_path.exists() else '✗'} {metrics_path.name}")
        st.write(f"**Landscape file:** {'✓' if landscape_path.exists() else '✗'} {landscape_path.name}")
        st.write(f"**Contexts file:** {'✓' if contexts_path.exists() else '✗'} {contexts_path.name}")
        st.write(f"**Target word:** {target_word}")


if __name__ == "__main__":
    main() 