"""Landscape visualization components."""

import re
import numpy as np
import plotly.graph_objects as go
from typing import List
from src.analysis.landscapes import Landscape

class MetricsVisualizer:
    """Visualizer for metrics trends across layers."""
    
    @staticmethod
    def create_metrics_trend_plots(layer_metrics, current_layer):
        """Create three trend plots for metrics across layers."""
        layers = sorted(layer_metrics.keys())
        mev_values = []
        avg_sim_values = []
        intra_sim_values = []
        inter_sim_values = []
        
        for layer in layers:
            data = layer_metrics[layer]
            if hasattr(data, 'mev'):
                mev_values.append(data.mev)
                avg_sim_values.append(data.average_similarity)
                intra_sim_values.append(data.intra_similarity if data.intra_similarity is not None else None)
                inter_sim_values.append(data.inter_similarity if data.inter_similarity is not None else None)
            else:
                mev_values.append(None)
                avg_sim_values.append(None)
                intra_sim_values.append(None)
                inter_sim_values.append(None)
        
        # MEV plot
        fig_mev = go.Figure()
        fig_mev.add_trace(go.Scatter(x=layers, y=mev_values, mode='lines+markers', name='MEV'))
        fig_mev.add_vline(x=current_layer, line_dash="dash", line_color="red", annotation_text="Current")
        fig_mev.update_layout(title="MEV", height=200, margin=dict(l=20, r=20, t=40, b=20))
        
        # Average Similarity plot
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(x=layers, y=avg_sim_values, mode='lines+markers', name='Avg Similarity'))
        fig_avg.add_vline(x=current_layer, line_dash="dash", line_color="red", annotation_text="Current")
        fig_avg.update_layout(title="Average Similarity", height=200, margin=dict(l=20, r=20, t=40, b=20))
        
        # Intra/Inter Similarity plot
        fig_cluster = go.Figure()
        if any(v is not None for v in intra_sim_values):
            fig_cluster.add_trace(go.Scatter(x=layers, y=intra_sim_values, mode='lines+markers', name='Intra-cluster'))
        if any(v is not None for v in inter_sim_values):
            fig_cluster.add_trace(go.Scatter(x=layers, y=inter_sim_values, mode='lines+markers', name='Inter-cluster'))
        fig_cluster.add_vline(x=current_layer, line_dash="dash", line_color="red", annotation_text="Current")
        fig_cluster.update_layout(
            title="Cluster Similarities", 
            height=200, 
            margin=dict(l=20, r=20, t=40, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig_mev, fig_avg, fig_cluster

class LandscapeVisualizer:
    """Create interactive visualizations of conceptual landscapes."""
    
    @staticmethod
    def create_plotly_figure(landscape: Landscape, contexts: List[str], 
                           target_word: str, width: int = 50) -> go.Figure:
        """Create an interactive Plotly figure of the landscape.
        
        Args:
            landscape: Landscape data structure
            contexts: List of context strings for hover text
            target_word: Target word to highlight in contexts
            width: Text wrapping width for hover text
            
        Returns:
            Plotly figure object
        """
        # Create log-transformed density surface to avoid issues with zeros
        log_landscape = np.log10(landscape.density_surface + 1e-10)
        
        # Prepare hover text with highlighted target words
        hover_texts = []
        for i, context in enumerate(contexts):
            cluster = landscape.cluster_labels[i]
            hover_texts.append(wrap_text_with_highlight(context, target_word, None, width))

        # Create figure with clean layout
        fig = go.Figure()
        
        # Add contour plot for density surface
        fig.add_trace(go.Contour(
            x=landscape.grid_x[0, :],
            y=landscape.grid_y[:, 0],
            z=log_landscape,
            colorscale='hot',
            opacity=0.6,
            showscale=False,
            contours=dict(
                start=log_landscape.min(),
                end=log_landscape.max(),
                size=(log_landscape.max() - log_landscape.min()) / 40,
            ),
            hoverinfo='skip'
        ))

        # Add scatter plot for data points  
        fig.add_trace(go.Scatter(
            x=landscape.pca_embeddings[:, 0],
            y=landscape.pca_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=landscape.cluster_labels,
                colorscale='turbo',
                opacity=0.9,
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='gray',
                font_size=14,
                font_family="Arial",
                font_color='black'
            ),
            showlegend=False
        ))

        # Clean, modern layout
        fig.update_layout(
            width=600,
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=12, color='black')
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=12, color='black')
            ),
            hovermode='closest'
        )
        
        return fig


def wrap_text_with_highlight(text: str, keyword: str, color: str = None, width: int = 50) -> str:
    """Highlight keyword and wrap text with simple HTML.
    
    Args:
        text: Input text to process
        keyword: Keyword to highlight
        color: Color for highlighting (unused, kept for compatibility)
        width: Maximum line width for wrapping
        
    Returns:
        HTML-formatted text with highlighting and line breaks
    """
    lines = []
    current_line = ''
    
    words = text.split(' ')
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += ' ' + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # highlighting target
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    highlighted_lines = []
    for line in lines:
        highlighted_line = pattern.sub(f'<b><span style="color: #000080;">\\g<0></span></b>', line)
        highlighted_lines.append(highlighted_line)
        
    return '<br>'.join(highlighted_lines)