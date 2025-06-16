import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
import tqdm
from dataclasses import dataclass

import re

import numpy as np
import plotly.graph_objects as go

@dataclass
class Landscape:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    X_pca: np.ndarray
    consensus_labels: np.ndarray
    ari_scores: list


    def create_plotly_landscape(self, contexts, target_word, width=50):
        fig = go.Figure()
        
        log_landscape = np.log10(self.Z + 1e-10)  # Avoid log(0) issues
        
        fig.add_trace(go.Contour(
            x=self.X[0, :],
            y=self.Y[:, 0],
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

        hover_texts = []
        for i, context in enumerate(contexts):
            cluster = self.consensus_labels[i]

            color = f'rgb(0, {80 + int(175 * (cluster/8))}, {180 + int(75 * (cluster/8))})'
            hover_texts.append(wrap_text_with_highlight(context, target_word, color, width))
        

        fig.add_trace(go.Scatter(
            x=self.X_pca[:, 0],
            y=self.X_pca[:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=self.consensus_labels,
                colorscale='viridis',
                showscale=False,
            ),
            text=hover_texts,
            hoverinfo='text',
            hoverlabel=dict(
                # bgcolor='rgb(240,242,246)',  # Light gray background
                # bordercolor='rgb(230,232,236)',  # Slightly darker border
                font_size=16,
                font_family="Arial",
                align="left"
            ),
            showlegend=False
        ))
        

        fig.update_layout(
            title=dict(
                text=f'{target_word}',
                font=dict(size=20, color='rgb(49,51,63)') 
            ),
            width=800,
            height=800,
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
    
def wrap_text_with_highlight(text, keyword, color, width=50):
    """Highlight keyword and wrap text with simple HTML"""
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

def optimize_clustering(embeddings, pca_components_range, n_clusters_range, 
                      covariance_types=['full', 'tied', 'diag', 'spherical']):
    
    results = []
    best_score = -1
    best_params = None
    best_pca = None
    best_gmm = None

    param_grid = {
        'n_components': pca_components_range,
        'n_clusters': n_clusters_range,
        'covariance_type': covariance_types
    }
    
    for params in ParameterGrid(param_grid):
        pca = PCA(n_components=int(params['n_components']))
        reduced_data = pca.fit_transform(embeddings)
        
        gmm = GaussianMixture(
            n_components=int(params['n_clusters']),
            covariance_type=params['covariance_type']
        )
        
        labels = gmm.fit_predict(reduced_data)
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(reduced_data, labels)
            
            results.append({
                'n_pca_components': int(params['n_components']),
                'n_clusters': int(params['n_clusters']),
                'covariance_type': params['covariance_type'],
                'silhouette_score': float(score),
                'explained_variance_ratio': np.sum(pca.explained_variance_ratio_)
            })
            
            print(f"Params: {params}, Silhouette Score: {score:.4f}, Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_pca = pca
                best_gmm = gmm
    
    best_score = float(best_score)
    best_params['n_components'] = int(best_params['n_components'])
    best_params['n_clusters'] = int(best_params['n_clusters'])

    results_df = pd.DataFrame(results)

    best = {
        'best_params': best_params,
        'best_score': best_score,
        'best_pca': best_pca,
        'best_gmm': best_gmm
    }
    
    return best, results_df


def get_consensus_labels_and_ARI(embeddings,
                                 optimal_params,
                                 forced_components=False,
                                 labels=True):
    
    if forced_components:
        n_components = forced_components
    else:
        n_components = optimal_params['n_components']
    

    n_clusters = optimal_params['n_clusters']

    X = embeddings

    X_pca = PCA(n_components=n_components).fit_transform(X)

    n_init = 100 

    cluster_assignments = []
    Zs = []

    for _ in range(n_init):
        gmm = GaussianMixture(n_components=n_clusters, n_init=1, random_state=np.random.randint(100000))
        gmm.fit(X_pca)
        labels = gmm.predict(X=X_pca)
        cluster_assignments.append(labels)


        x_min = X_pca[:, 0].min() - 1
        x_max = X_pca[:, 0].max() + 1
        y_min = X_pca[:, 1].min() - 1
        y_max = X_pca[:, 1].max() + 1

        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        XX, Y = np.meshgrid(x, y)
        XXX = np.array([XX.ravel(), Y.ravel()]).T
        Z = -gmm.score_samples(XXX)
        Z = Z.reshape(XX.shape)
        Zs.append(Z)

    ari_scores = []

    for i in range(n_init):
        for j in range(i + 1, n_init):
            ari = adjusted_rand_score(cluster_assignments[i], cluster_assignments[j])
            ari_scores.append(ari)

    consensus_matrix = np.zeros((X_pca.shape[0], X_pca.shape[0]))

    for labels in cluster_assignments:
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
                    if i != j:
                        consensus_matrix[j, i] += 1

    consensus_matrix /= n_init

    link = linkage(consensus_matrix, method='average')
    consensus_labels = fcluster(link, t=n_clusters, criterion='maxclust')

    x_min = X_pca[:, 0].min() - 1
    x_max = X_pca[:, 0].max() + 1
    y_min = X_pca[:, 1].min() - 1
    y_max = X_pca[:, 1].max() + 1

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    mean_landscape = np.mean(Zs, axis=0)

    landscape = {
        'X': X,
        'Y': Y,
        'Z': mean_landscape,
        'X_pca': X_pca,
        'consensus_labels': consensus_labels,
        'ari_scores': ari_scores
    }

    return landscape