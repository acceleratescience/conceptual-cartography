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
                font_family="Arial"
            ),
            showlegend=False
        ))

        # Clean, modern layout
        fig.update_layout(
            width=800,
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
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    colored_text = pattern.sub(f'<b>\\g<0></b>', text)
    
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