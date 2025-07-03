"""Landscape visualization components."""

import re
import numpy as np
import plotly.graph_objects as go
from typing import List
from src.analysis.landscapes import Landscape


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
        fig = go.Figure()
        
        # Create log-transformed density surface to avoid issues with zeros
        log_landscape = np.log10(landscape.density_surface + 1e-10)
        
        # Add contour plot for density surface
        fig.add_trace(go.Contour(
            x=landscape.grid_x[0, :],
            y=landscape.grid_y[:, 0],
            z=log_landscape,
            colorscale='hot',
            opacity=0.7,
            showscale=False,
            contours=dict(
                start=log_landscape.min(),
                end=log_landscape.max(),
                size=(log_landscape.max() - log_landscape.min()) / 50,
            ),
        ))

        # Prepare hover text with highlighted target words
        hover_texts = []
        for i, context in enumerate(contexts):
            cluster = landscape.cluster_labels[i]
            color = f'rgb(0, {80 + int(175 * (cluster/8))}, {180 + int(75 * (cluster/8))})'
            hover_texts.append(wrap_text_with_highlight(context, target_word, color, width))

        # Add scatter plot for data points
        fig.add_trace(go.Scatter(
            x=landscape.pca_embeddings[:, 0],
            y=landscape.pca_embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=landscape.cluster_labels,
                colorscale='viridis',
                showscale=False,
            ),
            text=hover_texts,
            hoverinfo='text',
            hoverlabel=dict(
                font_size=16,
                font_family="Arial",
                align="left"
            ),
            showlegend=False
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{target_word}',
                font=dict(size=20, color='rgb(49,51,63)') 
            ),
            width=600,
            height=600,
            paper_bgcolor='rgba(255,255,255,0.8)',
            plot_bgcolor='rgba(255,255,255,0.8)',
            xaxis=dict(
                tickfont=dict(size=16),
                zeroline=False,
                gridcolor='rgb(240,242,246)'
            ),
            yaxis=dict(
                tickfont=dict(size=16),
                zeroline=False,
                gridcolor='rgb(240,242,246)'
            ),
            hovermode='closest'
        )
        
        return fig


def wrap_text_with_highlight(text: str, keyword: str, color: str, width: int = 50) -> str:
    """Highlight keyword and wrap text with simple HTML.
    
    Args:
        text: Input text to process
        keyword: Keyword to highlight
        color: Color for highlighting (currently unused)
        width: Maximum line width for wrapping
        
    Returns:
        HTML-formatted text with highlighting and line breaks
    """
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    colored_text = pattern.sub(f'<b>\\g<0></b></span>', text)
    
    lines = []
    current_line = ''
    
    words = colored_text.split(' ')
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += ' ' + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
        
    return '<br>'.join(lines) 