"""
dematel_llm.py
Automatiza a construção de matrizes DEMATEL com apoio de LLM
e plota o grafo resultante em Plotly.

Requer: openai, numpy, pandas, networkx, plotly
$ pip install openai numpy pandas networkx plotly
Suporta provedores OpenAI e Gemini.
"""

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

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # Biblioteca opcional, usada apenas se provider == "gemini"


# ---------- CONFIGURAÇÃO BÁSICA ---------------------------------------------

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")            # definido no .env
LLM_MODEL       = "gpt-4o-mini"                         # mude se quiser
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")           # definido no .env
GEMINI_MODEL    = "gemini-pro"                          # mude se quiser
DEFAULT_PROVIDER= os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" ou "gemini"
SCALE_DESC      = "0 = sem influência nenhuma, 1 = muito baixa ... 9 = muito alta"
PROMPT_TMPL = (
    "Você é um engenheiro Aeroespacial, especialista em propulsão de foguetes, com vasto conhecimento teórico, mas também prático. Você foi colocado em um projeto de construção de um motor foguete a propelente híbrido e está realizando um teste no modelo DEMATEL para categorizar a influencia entre fatores e sua causalidade.\n"
    
    "Considere o par de fatores abaixo no contexto do projeto de desenvolvimento de um motor foguete híbrido, cujo requisito principal de sucesso é o empuxo gerado pelo motor. Julgue conforme a pergunta a seguir:\n"
    
    "Em uma escala de 0 a 9 ({scale}), qual o nível em que **{src}** influencia **{tgt}**? \n\n"
    
    "Considere que muitos fatores podem não ter influência direta, então sinta-se à vontade para responder 0 se achar que não há influência significativa.\n\n"

    "\nEntenda\n -{src} como: {description_src};\n -{tgt} como: {description_tgt}\n\n"

    "Avalie diferentes cenários em que é possível se ter {src} e como variações (pequenas ou grandes) em {src} pode influenciar {tgt}. Atente-se à magnitude dessa influência, e não à sua direção (positiva ou negativa).\n"
    
    "Para a definição de influência, considere também, se {tgt} é um fator que pode ser afetado por {src} considerando a lógica do projeto e a sua participação. Antes de responder, pense na origem dos fatores (por exemplo, se são aspectos externos, se são aspectos de projetos, se referem apenas a atributos de propelente ou atributos estruturais) para definir se o aspecto target é realmente passível de ser alterado. É possível que {tgt} possa ter influência direta, mas que não possam ser alterados diretamente, no contexto de um projeto, por {src}.\n\n"
    
    "Entenda que o resultado deve ser interpretado no contexto do projeto e a influência é uma relação unidirecional. Associe com o contexto de causalidade de A em B.\n\n"

    "A resposta deve ser apenas o número."
)

# ---------- DETECÇÃO DA VERSÃO DO openai ----------------------------------
try:
    from importlib.metadata import version as _pkg_version
    _OPENAI_V0 = _pkg_version("openai").startswith("0.")
except Exception:
    # Fallback heurístico
    _OPENAI_V0 = hasattr(openai, "ChatCompletion") and not hasattr(openai, "OpenAI")

if not _OPENAI_V0:
    # openai>=1.x usa cliente explícito
    # Aceita quaisquer kwargs (api_key, organization, etc.)
    def _make_openai_client(**kwargs):
        return openai.OpenAI(**kwargs)
else:
    _make_openai_client = None

# ---------- CLASSE PRINCIPAL -------------------------------------------------

class DematelLLM:
    def __init__(
        self,
        factors: list[str],
        descriptions: list[str],
        provider: str = DEFAULT_PROVIDER,
        api_key: str | None = None,
        model: str | None = None,
        prompt_tmpl: str = PROMPT_TMPL,
        cache: dict[tuple[str, str], int] | None = None,
     ):
        """
        Parameters
        ----------
        factors : list[str]
            Lista de fatores que irão compor a matriz DEMATEL.
        provider : {"openai", "gemini"}
            Provedor de LLM; default vem de `DEFAULT_PROVIDER`.
        api_key : str | None
            Chave da API; se None, é lida das variáveis de ambiente.
        model : str | None
            Nome do modelo; se None, usa o default do provedor.
        prompt_tmpl : str
            Template de prompt utilizado para as comparações par‑a‑par.
        cache : dict | None
            Cache opcional para evitar consultas repetidas à LLM.
        """
        self.provider = provider.lower()
        self.prompt_tmpl = prompt_tmpl
        self.cache = cache or {}

        # Configuração específica do provedor ------------
        if self.provider == "openai":
            self._setup_openai(api_key, model)
        elif self.provider == "gemini":
            self._setup_gemini(api_key, model)
        else:
            raise ValueError("Provider deve ser 'openai' ou 'gemini'.")

        # Atributos de domínio DEMATEL -------------------
        self.factors = factors
        self.descriptions = descriptions
        self.n = len(factors)
        self.A = np.zeros((self.n, self.n), dtype=float)

        # --- Pipeline DEMATEL ---------------------------
        self._build_direct_matrix()
        self._dematel()
        self.G = self._build_graph(numeric_filter=True)

    # ---------- Helpers de configuração -----------------
    def _setup_openai(self, api_key: str | None, model: str | None):
        """
        Configura cliente OpenAI (v0 ou v1).
        Caso exista a variável de ambiente OPENAI_ORG ou OPENAI_ORGANIZATION,
        ela será passada para o cliente, evitando o erro 401
        `invalid_organization`.
        """
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY não definido.")

        org = os.getenv("OPENAI_ORG_ID")
        self.model = model or LLM_MODEL

        if _OPENAI_V0:
            # Interface clássica (<1.0)
            if org:
                if org:
                    self._oai_client = openai.OpenAI(api_key=api_key, organization=org)
                else:
                    self._oai_client = openai.OpenAI(api_key=api_key)
                # openai.organization = org
            self._oai_client = openai
        else:
            # Interface nova (>=1.0)
            kwargs = dict(api_key=api_key)
            if org:
                kwargs["organization"] = org
            self._oai_client = _make_openai_client(**kwargs)

    def _setup_gemini(self, api_key: str | None, model: str | None):
        if genai is None:
            raise ImportError("google-generativeai não instalado. `pip install google-generativeai`")
        api_key = api_key or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY não definido.")
        genai.configure(api_key=api_key)
        self.model = model or GEMINI_MODEL

    # ---------- PASSO 1: conversa com a LLM para preencher A -----------------
    def _ask_llm(self, src: str, tgt: str, description_src: str, description_tgt: str) -> int:
        if (src, tgt) in self.cache:
            return self.cache[(src, tgt)]

        prompt = self.prompt_tmpl.format(
                src=src, tgt=tgt, 
                description_src=description_src, description_tgt=description_tgt, 
                scale=SCALE_DESC
            )

        if self.provider == "openai":
            if _OPENAI_V0:
                resp = self._oai_client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
            else:  # openai>=1
                resp = self._oai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
        elif self.provider == "gemini":
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(prompt)
            txt = resp.text.strip()
        else:
            raise RuntimeError("Provider não suportado.")

        m = re.search(r"[0-9]", txt)
        score = int(m.group()) if m else 0
        self.cache[(src, tgt)] = score
        
        print(f'{"-"*80}\n')
        print('\n\nPergunta:', prompt)
        print('Resposta:', txt, f"({score})\n\n")
        print(f'{"-"*80}\n')
        
        return score

    def _build_direct_matrix(self) -> None:
        # self.A = np.array([[0, 4, 7, 4, 3, 4, 5, 0, 3, 3, 0, 7.],
        #     [6, 0, 7, 3, 3, 6, 7, 3, 5, 7, 0, 7.],
        #     [7, 7, 0, 5, 3, 5, 5, 0, 3, 4, 0, 7.],
        #     [5, 3, 5, 0, 2, 3, 5, 0, 3, 4, 0, 7.],
        #     [5, 4, 3, 6, 0, 7, 6, 3, 5, 6, 0, 7.],
        #     [2, 0, 0, 2, 3, 0, 5, 0, 6, 3, 0, 7.],
        #     [3, 3, 0, 3, 3, 5, 0, 3, 4, 2, 0, 7.],
        #     [0, 0, 0, 0, 0, 3, 5, 0, 0, 0, 0, 7.],
        #     [3, 2, 0, 0, 2, 3, 5, 0, 0, 3, 0, 7.],
        #     [0, 0, 0, 0, 2, 2, 3, 0, 0, 0, 0, 6.],
        #     [0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 5.],
        #     [3, 3, 3, 3, 3, 3, 5, 0, 0, 0, 0, 0.],]
        # )

        
        for i, src in enumerate(self.factors):
            for j, tgt in enumerate(self.factors):
                if i == j: continue
               
                if check_pergunta_valida(src, tgt):
                    self.A[i, j] = self._ask_llm(src, tgt, self.descriptions[i], self.descriptions[j])
                else:
                    self.A[i, j] = 0  # Impõe zero se há restrição externa
        
        print('Matriz formada:', self.A)
        
        # salva num txt
        np.savetxt('Agente AI/matriz_dematel.txt', self.A, fmt='%d', delimiter=', ')

    # ---------- PASSO 2: normalização (M) e matriz de relação total (T) -----
    def _dematel(self) -> None:
        row_sums = self.A.sum(axis=1)
        col_sums = self.A.sum(axis=0)
        k = min(1/row_sums.max(), 1/col_sums.max())
        self.M = self.A * k
        self.T = self.M @ np.linalg.inv(np.eye(self.n) - self.M)
        self.R = self.T.sum(axis=1)            # influência exercida
        self.C = self.T.sum(axis=0)            # influência recebida
        self.rc_sum = self.R + self.C
        self.rc_diff= self.R - self.C

    # ---------- util: garante que o grafo seja DAG -----------------
    def _ensure_dag(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Remove arestas até que o grafo fique acíclico.
        Estratégia: enquanto houver ciclo, encontra um deles e
        remove a aresta de menor peso no ciclo.
        """
        G = G.copy()
        try:
            find_cycle = nx.find_cycle  # networkx ≥ 3
        except AttributeError:
            from networkx.algorithms.cycles import find_cycle  # retro‑compat

        while True:
            try:
                cycle_edges = find_cycle(G, orientation="original")
            except nx.exception.NetworkXNoCycle:
                break  # já é DAG
            # menor peso dentro do ciclo
            min_edge = min(
                cycle_edges,
                key=lambda e: G.get_edge_data(e[0], e[1]).get("weight", 1)
            )
            G.remove_edge(*min_edge[:2])
        return G

    # ---------- util: layout hierárquico (top‑down) -----------------
    def _hierarchical_pos(self, G: nx.DiGraph, layer_gap: float = 0.25) -> dict:
        """
        Retorna um dicionário {nó: (x, y)} onde y decresce de 0 para -1
        conforme a profundidade no DAG. Usado para plotar pais acima dos filhos.
        """
        # nível (profundidade) de cada nó = distância máxima até uma raiz
        roots = [n for n, d in G.in_degree() if d == 0] or list(G.nodes())
        level = {root: 0 for root in roots}
        for node in nx.topological_sort(G):
            for child in G.successors(node):
                level[child] = max(level.get(child, 0), level[node] + 1)

        # nós por nível
        max_level = max(level.values())
        nodes_by_lvl = {l: [n for n, lv in level.items() if lv == l]
                        for l in range(max_level + 1)}

        pos = {}
        for l, nodes in nodes_by_lvl.items():
            # x espaçado uniformemente em cada nível
            if len(nodes) == 1:
                xs = [0.5]
            else:
                xs = np.linspace(0.1, 0.9, len(nodes))
            y = -l * layer_gap
            for x, n in zip(xs, nodes):
                pos[n] = (x, y)
        return pos

    # ---------- PASSO 3: constrói grafo usando threshold --------------------
    def _build_graph(self,
                 threshold: float | None = None,
                 include_weights: bool = True,
                 enforce_dag: bool = True,
                 numeric_filter: bool = False) -> nx.DiGraph:
        if threshold is None:
            threshold = self.T.mean()          # heurística simples
        G = nx.DiGraph()
        G.add_nodes_from(self.factors)
        # --- Filtro DEMATEL numérico opcional ---------------------------
        if numeric_filter:
            prom_threshold = np.percentile(self.rc_sum, 50)
            # mantém arestas cujo peso esteja acima de (μ + 0.5·σ)
            edge_threshold = self.T.mean() + 0.5 * self.T.std()
        else:
            prom_threshold = None
            edge_threshold = None
            
            
        for i, j in itertools.product(range(self.n), repeat=2):
            if i == j:
                continue
            if self.T[i, j] <= threshold:
                continue
            if numeric_filter:
                # só mantém se nó de origem é proeminente E peso é alto
                if not (self.rc_sum[i] > prom_threshold and self.T[i, j] > edge_threshold):
                    continue
            # passou em todos os filtros → adiciona aresta
            w = round(self.T[i, j], 3) if include_weights else 1
            G.add_edge(self.factors[i], self.factors[j], weight=w)
        if enforce_dag:
            G = self._ensure_dag(G)
        return G

    # ---------- PASSO 4: plota em plotly express ----------------------------
    def plot(
        self,
        node_size_scale: float = 50.0,
        edge_width_scale: float = 3.0,
        title: str | None = None,
        hierarchical: bool = True,
     ):
        if hierarchical:
            pos = self._hierarchical_pos(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        nx.set_node_attributes(self.G, pos, "pos")

        # -----------------------------------------
        # DataFrame de nós
        node_df = pd.DataFrame(
            {
                "factor": list(self.G.nodes()),
                "x": [pos[n][0] for n in self.G.nodes()],
                "y": [pos[n][1] for n in self.G.nodes()],
                "RC_sum": self.rc_sum,
                "RC_diff": self.rc_diff,
            }
        )

        fig = go.Figure()

        # -----------------------------------------
        # 1) Relações bidirecionais → linha vermelha sem seta
        drawn_pairs = set()
        for u, v in self.G.edges():
            if (v, u) in self.G.edges() and (v, u) not in drawn_pairs:
                fig.add_trace(
                    go.Scatter(
                        x=[pos[u][0], pos[v][0]],
                        y=[pos[u][1], pos[v][1]],
                        mode="lines",
                        line=dict(width=edge_width_scale, color="red"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                drawn_pairs.add((u, v))
                drawn_pairs.add((v, u))

        # -----------------------------------------
        # 2) Relações unilaterais → seta preta
        bbox = {}  # guarda meio-largura e meio-altura de cada nó
        # shapes e textos também usam bbox, então preenche junto com shapes/textos
        shapes = []
        text_traces = []
        for _, row in node_df.iterrows():
            txt = row["factor"]
            x, y = row["x"], row["y"]

            # --- dimensões com padding generoso ---
            txt_len = len(txt)
            pad = 0.02
            w = 0.01 + txt_len * 0.008 + pad   # largura
            h = 0.1 + pad                     # altura
            bbox[txt] = (w / 2, h / 2)

            shapes.append(
                dict(
                    type="rect",
                    x0=x - w / 2,
                    x1=x + w / 2,
                    y0=y - h / 2,
                    y1=y + h / 2,
                    line=dict(color="rgba(60,60,60,0.9)", width=1.5),
                    fillcolor="rgba(52,152,219,0.85)",  # azul mais vibrante
                    layer="below",
                )
            )

            text_traces.append(
                go.Scatter(
                    x=[x],
                    y=[y],
                    text=[txt],
                    mode="text",
                    textfont=dict(color="white", size=14, family="Arial"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for u, v, d in self.G.edges(data=True):
            if (v, u) in self.G.edges():  # já tratada como bidirecional
                continue
            w = d.get("weight", 1)
            # vetores direção
            ux, uy = pos[u]
            vx, vy = pos[v]
            dx, dy = vx - ux, vy - uy
            # desloca saída até borda do retângulo de origem
            hw_u, hh_u = bbox[u]
            sx = hw_u / abs(dx) if dx else float("inf")
            sy = hh_u / abs(dy) if dy else float("inf")
            t_u = min(sx, sy, 1)  # escalonamento máximo 1
            ox = ux + dx * t_u
            oy = uy + dy * t_u
            # desloca ponto de chegada até borda do retângulo de destino
            hw_v, hh_v = bbox[v]
            scale_x = hw_v / abs(dx) if dx else float("inf")
            scale_y = hh_v / abs(dy) if dy else float("inf")
            t_v = min(scale_x, scale_y) - 0.01  # -0.01 para garantir fora
            tx = vx - dx * t_v
            ty = vy - dy * t_v

            fig.add_trace(
                go.Scatter(
                    x=[ox, tx],
                    y=[oy, ty],
                    mode="lines",
                    line=dict(width=w * edge_width_scale, color="black"),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=tx,
                y=ty,
                ax=ox,
                ay=oy,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=2.5,
                arrowwidth=w * edge_width_scale,
                arrowcolor="black",
            )

        # -----------------------------------------
        # 3) Nós como retângulos azuis (shapes/texts já preenchidos acima)
        # adiciona os textos ao gráfico
        for tr in text_traces:
            fig.add_trace(tr)
        # adicionar os shapes
        fig.update_layout(shapes=shapes)

        fig.update_layout(
            showlegend=False,
            title=title or "Topologia DEMATEL‑LLM (DAG)",
            plot_bgcolor="#f7f9fc",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()



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

# --- use dentro da classe ---
DematelLLM.plot_influence_diagram = plot_influence_diagram


def check_pergunta_valida(fator1, fator2):
        df_check = df_relacao_fatores.loc[
            (df_relacao_fatores['Fator1'] == fator1) & (df_relacao_fatores['Fator2'] == fator2)
        ]
        if not df_check.empty:
            return False
        else:
            return True
    


# ------------------ EXEMPLO DE USO ------------------------------------------

if __name__ == "__main__":
    os.system("clear" if os.name == "posix" else "cls")

    print("DEMATEL-LLM: Construção automatizada de matrizes DEMATEL com LLM")
    print("Provedor:", DEFAULT_PROVIDER.upper())
    print("Modelo:", LLM_MODEL if DEFAULT_PROVIDER == "openai" else GEMINI_MODEL)
    print("SCALE_DESC:", SCALE_DESC)


    # Lê quais os fatores e descrições
    df_fatores = pd.read_csv('Agente AI/Fatores.csv')

    fatores = df_fatores['fator'].tolist()
    descricoes = df_fatores['descricao'].tolist()
    
    # Lê as restrições de não influência implementadas externamente
    df_relacao_fatores = pd.read_csv("Agente AI/relacao_fatores.csv", sep=",", header=0)
    df_relacao_fatores = df_relacao_fatores.melt(id_vars=["Fatores"], var_name="Fator2", value_name="Relacao")
    df_relacao_fatores = df_relacao_fatores.loc[df_relacao_fatores["Relacao"] == 'não']
    df_relacao_fatores.columns = ['Fator1', 'Fator2', 'Relacao']
    df_relacao_fatores = pd.concat([
        df_relacao_fatores[['Fator1', 'Fator2']],
        df_relacao_fatores[['Fator2', 'Fator1']].rename(columns={'Fator2': 'Fator1', 'Fator1': 'Fator2'})
    ], ignore_index=True)

    



    # mude para "openai" ou "gemini" conforme necessário
    project = DematelLLM(fatores, descricoes, provider="openai")


    project.plot(title="Influência entre fatores no projeto de Propulsão de Foguete Híbrido")

    project.plot_influence_diagram()