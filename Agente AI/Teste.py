import os, re, itertools
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

A = np.array([[0, 1, 3, 1],
        [1, 1, 2, 1],
        [2, 1, 1, 3],
        [2, 3, 0, 1]])
n = A.shape[0]

def _dematel(A) -> None:
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    k = min(1/row_sums.max(), 1/col_sums.max())
    M = A * k
    T = M @ np.linalg.inv(np.eye(n) - M)
    R = T.sum(axis=1)            # influência exercida
    C = T.sum(axis=0)            # influência recebida
    rc_sum = R + C
    rc_diff= R - C
    
    return T, R, C, rc_sum, rc_diff



T, R, C, rc_sum, rc_diff = _dematel(A)

fatores = ["Experiência da equipe", "Diâmetro da câmara de combustão",
               "Disponibilidade de recursos", "Pressão de operação"]



def ensure_dag(G: nx.DiGraph) -> nx.DiGraph:
    """
    Remove arestas até que o grafo fique acíclico (DAG).
    Estratégia: enquanto houver ciclo, encontra um deles e
    remove a aresta de MENOR peso dentro do ciclo.
    """
    G = G.copy()
    try:
        # networkx 3.x
        find_cycle = nx.find_cycle
    except AttributeError:
        from networkx.algorithms.cycles import find_cycle  # retro‑compat

    while True:
        try:
            cycle_edges = find_cycle(G, orientation="original")
        except nx.exception.NetworkXNoCycle:
            break  # Já é DAG
        # Menor peso na volta
        min_edge = min(
            cycle_edges,
            key=lambda e: G.get_edge_data(e[0], e[1]).get("weight", 1)
        )
        G.remove_edge(*min_edge[:2])
    return G


def _build_graph(T, R, C, rc_sum, rc_diff,
                     factors: list[str] | None = None,
                     threshold: float | None = None,
                     include_weights: bool = True,
                     enforce_dag: bool = True) -> nx.DiGraph:
        if threshold is None:
            threshold = T.mean()          # heurística simples
        G = nx.DiGraph()
        G.add_nodes_from(factors)
        for i, j in itertools.product(range(n), repeat=2):
            if i != j and T[i, j] > threshold:
                w = round(T[i, j], 3) if include_weights else 1
                G.add_edge(factors[i], factors[j], weight=w)
        if enforce_dag:
            G = ensure_dag(G)
        return G



G = _build_graph(T, R, C, rc_sum, rc_diff, fatores, threshold=0.1, enforce_dag=True)

def plot(G, rc_sum, rc_diff,
             node_size_scale: float = 50.0,
             edge_width_scale: float = 3.0,
             title: str | None = None):
        pos  = nx.spring_layout(G, seed=42)
        nx.set_node_attributes(G, pos, "pos")

        # Dataframes para Plotly
        node_df = pd.DataFrame({
            "factor"  : list(G.nodes()),
            "x"       : [pos[n][0] for n in G.nodes()],
            "y"       : [pos[n][1] for n in G.nodes()],
            "RC_sum"  : rc_sum,
            "RC_diff" : rc_diff
        })
        # scatter para arestas (um traço por aresta para permitir larguras diferentes)
        edge_fig = go.Figure()
        for u, v, d in G.edges(data=True):
            w = d["weight"]
            edge_fig.add_trace(
                go.Scatter(
                    x=[pos[u][0], pos[v][0]],
                    y=[pos[u][1], pos[v][1]],
                    mode="lines",
                    line=dict(width=w * edge_width_scale),
                    hoverinfo="none",
                    showlegend=False
                )
            )
        # scatter para nós
        node_fig = px.scatter(
            node_df,
            x="x", y="y", text="factor",
            size=(node_df["RC_sum"]*node_size_scale),
            hover_data=["RC_sum", "RC_diff"]
        )
        # combinação
        for tr in node_fig.data: edge_fig.add_trace(tr)
        edge_fig.update_layout(showlegend=False,
                               title=title or "Topologia DEMATEL-LLM")
        edge_fig.update_xaxes(visible=False); edge_fig.update_yaxes(visible=False)
        edge_fig.show()
        
        
plot(G, rc_sum, rc_diff,
        node_size_scale=50.0,
        edge_width_scale=3.0,
        title="Topologia DEMATEL-LLM - Fatores de Influência")
