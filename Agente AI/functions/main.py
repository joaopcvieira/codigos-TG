from __future__ import annotations
import os, re, itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import openai
from dotenv import load_dotenv
import textwrap


# ---------- PASSO 4: plot_network ---------
def plot_network(
    self,
    node_size_scale: float = 50.0,
    edge_width_scale: float = 3.0,
    title: str | None = None,
    hierarchical: bool = True,
    layer_gap: float = 0.28,
    wrap_chars: int = 18,
 ):
    """
    Alternativa com **Graphviz** para gerar um DAG legível:
    • Layout em camadas verticais (rankdir=TB), causas acima dos efeitos
    • Caixas com quebra de linha; sem sobreposição (overlap=false, nodesep/ranksep)
    • Arestas ortogonais roteadas por fora das caixas (splines=ortho)
    • Bidirecionais como duas arestas separadas

    Dependências: `pip install graphviz` e o binário do Graphviz instalado no sistema
    (ex.: `brew install graphviz` no macOS, `apt-get install graphviz` no Linux, ou o
    instalador oficial no Windows).
    """
    try:
        import graphviz
    except Exception as e:
        raise ImportError(
            "A biblioteca 'graphviz' não está disponível. Instale com 'pip install graphviz' e certifique-se de ter o Graphviz instalado no sistema."
        ) from e

    # ------- helpers (iguais em espírito ao plot_graph2) -------
    def _levels_dag(graph: nx.DiGraph) -> dict:
        H = graph.copy()
        try:
            _ = list(nx.topological_sort(H))
        except nx.NetworkXUnfeasible:
            H = ensure_dag(H)
        roots = [n for n, d in H.in_degree() if d == 0] or list(H.nodes())
        level = {r: 0 for r in roots}
        for u in nx.topological_sort(H):
            for v in H.successors(u):
                level[v] = max(level.get(v, 0), level[u] + 1)
        return level

    def _wrap(label: str, width: int) -> str:
        clean = ' '.join(str(label).split())
        if len(clean) <= width:
            return clean
        return "\n".join(textwrap.wrap(clean, width=width))

    # --------- níveis e grupos por rank ---------
    level = _levels_dag(self.G)
    max_level = max(level.values()) if level else 0
    nodes_by_lvl = {l: [n for n, lv in level.items() if lv == l] for l in range(max_level + 1)}

    # --------- prepara rótulos e pesos ---------
    rc_sum = self.T.sum(axis=1) + self.T.sum(axis=0)
    rc_diff = self.T.sum(axis=1) - self.T.sum(axis=0)

    # --------- cria o grafo Graphviz ---------
    ranksep = max(0.5, layer_gap * 2.8)  # separação vertical entre níveis
    dot = graphviz.Digraph(format="svg", engine="dot")
    dot.attr(
        rankdir="TB",
        splines="ortho",
        overlap="false",
        nodesep="0.45",
        ranksep=str(ranksep),
        pad="0.2",
        bgcolor="#f7f9fc",
        label=(title or "Topologia DEMATEL-LLM (DAG)"),
        labelloc="t",
        fontsize="18",
        fontname="Arial",
    )
    dot.node_attr.update(
        shape="box",
        style="rounded,filled",
        fillcolor="#3498DB",
        color="#3c3c3c",
        fontcolor="white",
        penwidth="1.5",
        margin="0.1,0.06",
        fontname="Arial",
    )
    dot.edge_attr.update(
        color="black",
        arrowsize="0.8",
        penwidth="1.2",
    )

    # Para estabilidade, damos um id interno seguro para cada nó
    node_ids = {name: f"n{idx}" for idx, name in enumerate(self.G.nodes())}

    # --- adiciona nós, agrupando por nível com rank=same ---
    for l in range(max_level + 1):
        with dot.subgraph(name=f"rank_{l}") as s:
            s.attr(rank="same")
            for n in nodes_by_lvl.get(l, []):
                label = _wrap(n, wrap_chars)
                # opcional: destacar causas (nível 0) com cor diferente
                attrs = {}
                if l == 0:
                    attrs["fillcolor"] = "#2E86C1"
                s.node(node_ids[n], label=label, **attrs)

    # --- adiciona arestas com espessura proporcional ao peso ---
    for u, v, d in self.G.edges(data=True):
        w = float(d.get("weight", 1))
        pen = max(1.0, w * edge_width_scale * 0.6)
        # força distância vertical coerente ao pular níveis
        lvl_diff = max(1, level.get(v, 0) - level.get(u, 0))
        dot.edge(node_ids[u], node_ids[v], penwidth=str(pen), minlen=str(lvl_diff))
    
    # Sempre salva como PNG na pasta Agente AI
    dot.render('Agente AI/dematel_graph', format='png', cleanup=True)
    print("Gráfico salvo como 'Agente AI/dematel_graph.png'")
    
    # Tenta exibir inline (Jupyter) se possível
    try:
        from IPython.display import display, SVG
        svg = dot.pipe(format="svg")
        display(SVG(svg))
    except Exception:
        pass  # Não faz nada se não conseguir exibir inline
    
    # Tenta abrir automaticamente o arquivo PNG
    try:
        import subprocess
        import sys
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", "Agente AI/dematel_graph.png"])
        elif sys.platform == "linux":  # Linux
            subprocess.run(["xdg-open", "Agente AI/dematel_graph.png"])
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", "Agente AI/dematel_graph.png"], shell=True)
    except Exception as e:
        print(f"Erro ao abrir arquivo: {e}")
        print("Código DOT do gráfico:")
        print(dot.source)

    return dot

def plot_influence_diagram(self, title="Diagrama de influências DEMATEL"):
    """
    Scatter (R+C  vs  R−C) com tamanho proporcional a R+C.
    R + C  → importância/prominence
    R − C  → tipo:
        positivo  → fator mais causador
        negativo → fator mais resultante
    """

    x = self.rc_sum            # Prominence
    y = self.rc_diff           # Net effect
    sizes = (x - x.min()) / (x.max() - x.min() + 1e-9) * 2000 + 300

    # Prepara os dados para o plot
    df = pd.DataFrame({
        "R_plus_C": x,
        "R_minus_C": y,
        "Factor": self.factors,
        "Size": sizes
    })

    # Formata o texto dos rótulos para múltiplas linhas se necessário
    def format_label(label):
        max_char = 1000
        return label if len(label) < max_char else "\n".join([label[i:i+max_char] for i in range(0, len(label), max_char)])
    df["Formatted_Factor"] = df["Factor"].apply(format_label)

    fig = px.scatter(
        df,
        x="R_plus_C",
        y="R_minus_C",
        size="Size",
        text="Formatted_Factor",
        title=title,
        labels={"R_plus_C": "R + C  (importância)", "R_minus_C": "R − C  (efeito líquido)"},
        size_max=60
    )
    fig.update_traces(textposition="middle center")
    
    # Adiciona linhas de referência em x=0 e y=0
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.add_hline(y=0, line_width=1, line_color="gray")
    
    fig.update_traces(textposition='middle center')
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    fig.show()
