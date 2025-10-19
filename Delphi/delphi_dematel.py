"""
delphi_dematel.py
Sistema DEMATEL com processo de auditoria inspirado na metodologia Delphi.
Permite comparar e revisar as avaliações do LLM com base no consenso de especialistas.

Funcionalidades principais:
- Armazenamento de justificativas e níveis de confiança
- Agregação de respostas de especialistas
- Processo de auditoria Delphi com múltiplas rodadas
- Métricas de concordância e análise de viés
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from datetime import datetime
import logging

from functions.main import plot_network, plot_influence_diagram, check_pergunta_valida

# Importações serão feitas dinamicamente para evitar problemas circulares
# from main import DematelLLM

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResponseRecord:
    """Registro de uma resposta do LLM para um par de fatores"""
    score: int
    rationale: str
    confidence: int  # 1-5 escala
    timestamp: str
    round_number: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseRecord':
        return cls(**data)


@dataclass
class ExpertStatistics:
    """Estatísticas agregadas das respostas dos especialistas"""
    median: float
    q1: float
    q3: float
    iqr: float
    mean: float
    std: float
    count: int
    responses: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DelphiMemory:
    """
    Gerencia a memória e histórico das avaliações do processo Delphi.
    Armazena justificativas, confiança e múltiplas rodadas de avaliação.
    """
    
    def __init__(self):
        # Estrutura: {(factor_i, factor_j): [ResponseRecord1, ResponseRecord2, ...]}
        self.responses: Dict[Tuple[str, str], List[ResponseRecord]] = {}
        # Estrutura: {(factor_i, factor_j): ExpertStatistics}
        self.expert_stats: Dict[Tuple[str, str], ExpertStatistics] = {}
    
    def add_response(self, factor_i: str, factor_j: str, response: ResponseRecord):
        """Adiciona uma nova resposta do LLM para um par de fatores"""
        key = (factor_i, factor_j)
        if key not in self.responses:
            self.responses[key] = []
        self.responses[key].append(response)
        logger.info(f"Resposta adicionada para {factor_i} -> {factor_j}: score={response.score}, confidence={response.confidence}")
    
    def get_latest_response(self, factor_i: str, factor_j: str) -> Optional[ResponseRecord]:
        """Retorna a resposta mais recente do LLM para um par de fatores"""
        key = (factor_i, factor_j)
        if key in self.responses and self.responses[key]:
            return self.responses[key][-1]
        return None
    
    def get_all_responses(self, factor_i: str, factor_j: str) -> List[ResponseRecord]:
        """Retorna todas as respostas do LLM para um par de fatores"""
        key = (factor_i, factor_j)
        return self.responses.get(key, [])
    
    def set_expert_stats(self, factor_i: str, factor_j: str, stats: ExpertStatistics):
        """Define as estatísticas dos especialistas para um par de fatores"""
        key = (factor_i, factor_j)
        self.expert_stats[key] = stats
        # logger.info(f"Estatísticas de especialistas definidas para {factor_i} -> {factor_j}: median={stats.median}, IQR={stats.iqr}")
    
    def get_expert_stats(self, factor_i: str, factor_j: str) -> Optional[ExpertStatistics]:
        """Retorna as estatísticas dos especialistas para um par de fatores"""
        key = (factor_i, factor_j)
        return self.expert_stats.get(key)
    
    def export_to_json(self, filepath: str):
        """Exporta toda a memória para um arquivo JSON"""
        data = {
            'responses': {
                f"{k[0]}->{k[1]}": [r.to_dict() for r in v] 
                for k, v in self.responses.items()
            },
            'expert_stats': {
                f"{k[0]}->{k[1]}": v.to_dict() 
                for k, v in self.expert_stats.items()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Memória exportada para {filepath}")
    
    def load_from_json(self, filepath: str):
        """Carrega memória de um arquivo JSON"""
        if not os.path.exists(filepath):
            logger.warning(f"Arquivo {filepath} não encontrado. Iniciando com memória vazia.")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Carregar respostas
        self.responses = {}
        for key_str, responses_data in data.get('responses', {}).items():
            factor_i, factor_j = key_str.split('->')
            key = (factor_i, factor_j)
            self.responses[key] = [ResponseRecord.from_dict(r) for r in responses_data]
        
        # Carregar estatísticas de especialistas
        self.expert_stats = {}
        for key_str, stats_data in data.get('expert_stats', {}).items():
            factor_i, factor_j = key_str.split('->')
            key = (factor_i, factor_j)
            self.expert_stats[key] = ExpertStatistics(**stats_data)
        
        logger.info(f"Memória carregada de {filepath}")
    
    def get_all_justifications(self) -> Dict[str, str]:
        """
        Retorna todas as justificativas mais recentes por par de fatores
        """
        justifications = {}
        for (factor_i, factor_j), responses in self.responses.items():
            if responses:
                latest_response = responses[-1]
                key = f"{factor_i}->{factor_j}"
                justifications[key] = latest_response.rationale
        return justifications
    
    def get_justifications_by_round(self, round_number: int) -> Dict[str, str]:
        """
        Retorna justificativas de uma rodada específica
        round_number: 1 = primeira rodada, 2 = segunda rodada, etc.
        """
        justifications = {}
        for (factor_i, factor_j), responses in self.responses.items():
            if len(responses) >= round_number:
                response = responses[round_number - 1]  # -1 porque lista é 0-indexed
                key = f"{factor_i}->{factor_j}"
                justifications[key] = response.rationale
        return justifications


class ExpertAggregator:
    """
    Agrega e analisa respostas de múltiplos especialistas.
    Calcula estatísticas descritivas para comparação com o LLM.
    
    IMPORTANTE: Converte automaticamente escalas de especialistas (0-4) para escala LLM (0-9)
    para garantir comparabilidade nas análises Delphi.
    """
    
    @staticmethod
    def convert_expert_scale_to_llm(expert_score: float) -> float:
        """
        Converte escala de especialistas (0-4) para escala LLM (0-9).
        
        Usa mapeamento linear: novo_valor = (valor_original / 4) * 9
        
        Parameters
        ----------
        expert_score : float
            Pontuação na escala original dos especialistas (0-4)
            
        Returns
        -------
        float
            Pontuação convertida para escala LLM (0-9)
        """
        if expert_score < 0 or expert_score > 4:
            logger.warning(f"Valor de especialista fora da escala esperada (0-4): {expert_score}")
        
        # Mapeamento linear de 0-4 para 0-9
        converted = (expert_score / 4.0) * 9.0
        
        # Garante que está dentro dos limites
        converted = max(0.0, min(9.0, converted))
        
        return converted
    
    @staticmethod
    def test_scale_conversion():
        """
        Testa e demonstra a conversão de escalas.
        Útil para validar que a conversão está funcionando corretamente.
        """
        print("Teste de Conversão de Escalas:")
        print("Especialistas (0-4) → LLM (0-9)")
        print("-" * 35)
        
        test_values = [0, 1, 2, 3, 4, 2.5, 3.5]
        for original in test_values:
            converted = ExpertAggregator.convert_expert_scale_to_llm(original)
            print(f"{original:4.1f} → {converted:4.1f}")
        
        print("\nMapeamento completo:")
        print("0 (nenhuma) → 0.0 | 1 (baixa) → 2.25 | 2 (média) → 4.5")
        print("3 (alta) → 6.75 | 4 (muito alta) → 9.0")
        
        return True
    
    @staticmethod
    def load_expert_matrix(filepath: str) -> np.ndarray:
        """
        Carrega matriz de especialistas de um arquivo.
        Formato esperado: CSV com matriz NxN ou múltiplas matrizes empilhadas.
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            # Assumindo que a primeira coluna são os nomes dos fatores
            if df.columns[0] in ['factor', 'fator', 'factors']:
                matrix = df.iloc[:, 1:].values
            else:
                matrix = df.values
        elif filepath.endswith('.npy'):
            matrix = np.load(filepath)
        else:
            raise ValueError("Formato de arquivo não suportado. Use .csv ou .npy")
        
        return matrix.astype(float)
    
    @staticmethod
    def aggregate_expert_responses(expert_matrices: List[np.ndarray]) -> Dict[Tuple[int, int], ExpertStatistics]:
        """
        Agrega respostas de múltiplos especialistas.
        
        IMPORTANTE: Converte automaticamente as escalas dos especialistas (0-4) 
        para a escala do LLM (0-9) antes de calcular estatísticas, garantindo 
        comparabilidade nas análises Delphi.
        
        Parameters
        ----------
        expert_matrices : List[np.ndarray]
            Lista de matrizes NxN de cada especialista (escala original 0-4)
        
        Returns
        -------
        Dict[Tuple[int, int], ExpertStatistics]
            Estatísticas agregadas para cada par (i, j) na escala convertida (0-9)
        """
        if not expert_matrices:
            raise ValueError("Lista de matrizes de especialistas está vazia")
        
        # Verifica se todas as matrizes têm a mesma forma
        shape = expert_matrices[0].shape
        if not all(m.shape == shape for m in expert_matrices):
            raise ValueError("Todas as matrizes devem ter a mesma dimensão")
        
        n = shape[0]
        aggregated = {}
        
        for i in range(n):
            for j in range(n):
                if i == j:  # Pula diagonal principal
                    continue
                
                # Coleta todas as respostas dos especialistas para o par (i, j)
                raw_responses = [matrix[i, j] for matrix in expert_matrices if not np.isnan(matrix[i, j])]
                
                if not raw_responses:
                    continue
                
                # Converte escala de especialistas (0-4) para escala LLM (0-9)
                converted_responses = [ExpertAggregator.convert_expert_scale_to_llm(score) for score in raw_responses]
                responses = np.array(converted_responses)
                
                # Log da conversão para auditoria
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Conversão escala ({i},{j}): {raw_responses} -> {converted_responses}")
                
                # Calcula estatísticas (agora na escala 0-9)
                stats = ExpertStatistics(
                    median=float(np.median(responses)),
                    q1=float(np.percentile(responses, 25)),
                    q3=float(np.percentile(responses, 75)),
                    iqr=float(np.percentile(responses, 75) - np.percentile(responses, 25)),
                    mean=float(np.mean(responses)),
                    std=float(np.std(responses)),
                    count=len(responses),
                    responses=responses.tolist()  # Armazena valores convertidos
                )
                
                aggregated[(i, j)] = stats
        
        return aggregated
    
    @staticmethod
    def load_and_aggregate_from_file(filepath: str, factors: List[str]) -> Dict[Tuple[str, str], ExpertStatistics]:
        """
        Carrega dados de especialistas de um arquivo e agrega as respostas.
        Suporta dois formatos de arquivo CSV:
        
        FORMATO 1 (Linha por linha - RECOMENDADO):
        expert_id,factor_source,factor_target,influence_score
        1,Fator A,Fator B,2
        1,Fator A,Fator C,3
        2,Fator A,Fator B,1
        ...
        
        FORMATO 2 (Matrizes empilhadas):
        factor,Fator A,Fator B,Fator C
        Fator A,0,2,3
        Fator B,1,0,2
        Fator C,2,1,0
        Fator A,0,1,4
        ...
        
        IMPORTANTE: As respostas dos especialistas são automaticamente convertidas 
        da escala original (0-4) para a escala do LLM (0-9) para garantir 
        comparabilidade nas análises Delphi.
        
        Parameters
        ----------
        filepath : str
            Caminho para o arquivo contendo as avaliações dos especialistas (escala 0-4)
        factors : List[str]
            Lista dos nomes dos fatores
        
        Returns
        -------
        Dict[Tuple[str, str], ExpertStatistics]
            Estatísticas agregadas mapeadas por nomes dos fatores (escala convertida 0-9)
        """
        df = pd.read_csv(filepath)
        
        # Detecta formato do arquivo
        columns = df.columns.tolist()
        
        if 'expert_id' in columns and 'factor_source' in columns and 'factor_target' in columns and 'influence_score' in columns:
            # FORMATO 1: Linha por linha
            return ExpertAggregator._load_rowwise_format(df, factors)
        else:
            # FORMATO 2: Matrizes empilhadas
            return ExpertAggregator._load_matrix_format(df, factors)
    
    @staticmethod
    def _load_rowwise_format(df: pd.DataFrame, factors: List[str]) -> Dict[Tuple[str, str], ExpertStatistics]:
        """
        Processa formato linha por linha:
        expert_id,factor_source,factor_target,influence_score
        """
        # Agrupa por par de fatores
        factor_pairs = {}
        
        for _, row in df.iterrows():
            factor_src = row['factor_source']
            factor_tgt = row['factor_target']
            score = float(row['influence_score'])
            
            # Valida que os fatores existem na lista
            if factor_src not in factors or factor_tgt not in factors:
                continue
            
            pair_key = (factor_src, factor_tgt)
            if pair_key not in factor_pairs:
                factor_pairs[pair_key] = []
            
            factor_pairs[pair_key].append(score)
        
        # Calcula estatísticas para cada par
        aggregated = {}
        for pair_key, scores in factor_pairs.items():
            if scores:  # Se há pelo menos uma avaliação
                # Converte escala de especialistas (0-4) para LLM (0-9)
                converted_scores = [ExpertAggregator.convert_expert_scale_to_llm(score) for score in scores]
                
                # Calcula estatísticas
                stats = ExpertStatistics(
                    mean=np.mean(converted_scores),
                    median=np.median(converted_scores),
                    std=np.std(converted_scores),
                    q1=np.percentile(converted_scores, 25),
                    q3=np.percentile(converted_scores, 75),
                    iqr=np.percentile(converted_scores, 75) - np.percentile(converted_scores, 25),
                    count=len(converted_scores),
                    responses=converted_scores
                )
                
                aggregated[pair_key] = stats
        
        return aggregated
    
    @staticmethod
    def _load_matrix_format(df: pd.DataFrame, factors: List[str]) -> Dict[Tuple[str, str], ExpertStatistics]:
        """
        Processa formato de matrizes empilhadas (formato legado)
        """
        # Identifica quantos especialistas existem baseado na estrutura do arquivo
        n_factors = len(factors)
        n_experts = len(df) // n_factors
        
        expert_matrices = []
        for expert_idx in range(n_experts):
            start_row = expert_idx * n_factors
            end_row = (expert_idx + 1) * n_factors
            matrix = df.iloc[start_row:end_row, 1:].values  # Assumindo primeira coluna é identificador
            expert_matrices.append(matrix)
        
        # Agrega as respostas
        numeric_aggregated = ExpertAggregator.aggregate_expert_responses(expert_matrices)
        
        # Converte índices numéricos para nomes dos fatores
        factor_aggregated = {}
        for (i, j), stats in numeric_aggregated.items():
            factor_key = (factors[i], factors[j])
            factor_aggregated[factor_key] = stats
        
        return factor_aggregated


class DelphiDematel:
    """
    Extensão do DematelLLM que implementa o processo de auditoria Delphi.
    
    Funcionalidades:
    - Memória persistente de justificativas e confiança
    - Comparação com consenso de especialistas
    - Múltiplas rodadas de reavaliação
    - Métricas de concordância
    """
    
    def __init__(
        self,
        factors: List[str],
        descriptions: List[str],
        expert_data_path: Optional[str] = None,
        memory_path: Optional[str] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        factors : List[str]
            Lista de fatores para análise DEMATEL
        descriptions : List[str]
            Descrições detalhadas de cada fator
        expert_data_path : str, optional
            Caminho para arquivo com dados dos especialistas
        memory_path : str, optional
            Caminho para arquivo de memória persistente
        **kwargs
            Argumentos adicionais para DematelLLM
        """
        # Importa configurações centralizadas
        from config import DEFAULT_PROVIDER, PROMPT_TMPL
        
        # Herda atributos essenciais para DEMATEL
        self.factors = factors
        self.descriptions = descriptions
        self.n = len(factors)
        self.A = np.zeros((self.n, self.n), dtype=float)
        
        # Configurações do provider
        self.provider = kwargs.get('provider', DEFAULT_PROVIDER).lower()
        self.prompt_tmpl = kwargs.get('prompt_tmpl', PROMPT_TMPL)
        
        # Configuração específica do provedor
        if self.provider == "openai":
            self._setup_openai(kwargs.get('api_key'), kwargs.get('model'))
        elif self.provider == "gemini":
            self._setup_gemini(kwargs.get('api_key'), kwargs.get('model'))
        else:
            raise ValueError("Provider deve ser 'openai' ou 'gemini'.")
        
        # Inicializa componentes Delphi
        self.memory = DelphiMemory()
        self.memory_path = memory_path or "delphi_memory.json"
        
        # Carrega memória existente se disponível
        if os.path.exists(self.memory_path):
            self.memory.load_from_json(self.memory_path)
        
        # Carrega e agrega dados de especialistas se disponível
        if expert_data_path and os.path.exists(expert_data_path):
            self.load_expert_data(expert_data_path)
        else:
            logger.warning("Dados de especialistas não encontrados. Processo Delphi limitado.")
        
        self.current_round = 1
        
        # Template de prompt para reavaliação Delphi
        self.audit_prompt_template = self._create_audit_prompt_template()
    
    def _setup_openai(self, api_key: str | None, model: str | None):
        """Configura cliente OpenAI"""
        from config import OPENAI_API_KEY, LLM_MODEL, _make_openai_client, _OPENAI_V0
        import openai
        
        api_key = api_key or OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY não definido.")

        org = os.getenv("OPENAI_ORG_ID")
        self.model = model or LLM_MODEL

        if _OPENAI_V0:
            # Interface clássica (<1.0)
            if org:
                self._oai_client = openai.OpenAI(api_key=api_key, organization=org)
            else:
                self._oai_client = openai.OpenAI(api_key=api_key)
        else:
            # Interface nova (>=1.0)
            kwargs = dict(api_key=api_key)
            if org:
                kwargs["organization"] = org
            self._oai_client = _make_openai_client(**kwargs)

    def _setup_gemini(self, api_key: str | None, model: str | None):
        """Configura cliente Gemini"""
        from config import GEMINI_API_KEY, GEMINI_MODEL
        
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai não instalado. `pip install google-generativeai`")
            
        api_key = api_key or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY não definido.")
        genai.configure(api_key=api_key)
        self.model = model or GEMINI_MODEL
    
    def _dematel(self) -> None:
        """Executa cálculos DEMATEL (copiado de DematelLLM)"""
        row_sums = self.A.sum(axis=1)
        col_sums = self.A.sum(axis=0)
        k = min(1/row_sums.max(), 1/col_sums.max())
        self.M = self.A * k
        self.T = self.M @ np.linalg.inv(np.eye(self.n) - self.M)
        self.R = self.T.sum(axis=1)            # influência exercida
        self.C = self.T.sum(axis=0)            # influência recebida
        self.rc_sum = self.R + self.C
        self.rc_diff= self.R - self.C
    
    def _ensure_dag(self, G) -> 'nx.DiGraph':
        """
        Remove arestas até que o grafo fique acíclico.
        Estratégia: enquanto houver ciclo, encontra um deles e
        remove a aresta de menor peso no ciclo.
        """
        import networkx as nx
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
    
    def _build_graph(self,
                 threshold: float | None = None,
                 include_weights: bool = True,
                 enforce_dag: bool = True,
                 numeric_filter: bool = False) -> 'nx.DiGraph':
        """
        Constrói grafo usando threshold baseado na matriz T do DEMATEL.
        """
        import networkx as nx
        import itertools
        
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
    
    def load_expert_data(self, filepath: str):
        """Carrega e processa dados dos especialistas"""
        try:
            expert_aggregated = ExpertAggregator.load_and_aggregate_from_file(filepath, self.factors)
            
            for (factor_i, factor_j), stats in expert_aggregated.items():
                self.memory.set_expert_stats(factor_i, factor_j, stats)
            
            logger.info(f"Dados de {len(expert_aggregated)} pares de fatores carregados de {filepath}")
        except Exception as e:
            logger.error(f"Erro ao carregar dados de especialistas: {e}")
    
    def _create_audit_prompt_template(self) -> str:
        """Cria template de prompt para processo de auditoria Delphi"""
        return """Você é um engenheiro Aeroespacial especialista em propulsão de foguetes realizando uma REAVALIAÇÃO de sua análise anterior no contexto do método DEMATEL.

        CONTEXTO DO PROJETO: Desenvolvimento de motor foguete híbrido com foco no empuxo gerado.

        SUA AVALIAÇÃO ANTERIOR:
        - Fator origem: {src} ({description_src})
        - Fator destino: {tgt} ({description_tgt})
        - Sua nota anterior: {previous_score}/9
        - Sua justificativa anterior: {previous_rationale}
        - Seu nível de confiança anterior: {previous_confidence}/5

        CONSENSO DOS ESPECIALISTAS (escala 0-9, normalizada):
        - Mediana das avaliações: {expert_median:.1f}/9
        - Intervalo interquartil (Q1-Q3): {expert_q1:.1f} - {expert_q3:.1f}
        - Número de especialistas: {expert_count}
        - Desvio padrão: {expert_std:.2f}

        INSTRUÇÃO PARA REAVALIAÇÃO:
        Analise cuidadosamente sua avaliação anterior E o consenso dos especialistas. Mantenha sua AUTONOMIA TÉCNICA - não mude simplesmente para seguir a maioria. Altere sua avaliação APENAS se houver motivo técnico substantivo.

        Considere:
        1. Sua justificativa técnica original ainda é válida?
        2. O consenso dos especialistas revela algum aspecto técnico que você não considerou?
        3. Há evidências técnicas suficientes para alterar sua posição?

        RESPOSTA REQUERIDA (formato exato):
        NOTA: [0-9]
        JUSTIFICATIVA: [Explique detalhadamente seu raciocínio técnico, mencionando se e por que mantém ou altera sua avaliação]
        CONFIANÇA: [1-5]
        MUDANÇA: [SIM/NÃO - se alterou a nota original]

        Baseie sua decisão exclusivamente em argumentos técnicos de engenharia aeroespacial e propulsão."""
    
    def _ask_llm_with_memory(self, src: str, tgt: str, description_src: str, description_tgt: str) -> ResponseRecord:
        """
        Faz pergunta ao LLM e armazena resposta completa com justificativa e confiança.
        Estende o método _ask_llm original para capturar informações adicionais.
        """
        
        # Monta prompt estendido para capturar justificativa e confiança
        from config import SCALE_DESC
        extended_prompt = f"""{self.prompt_tmpl.format(
                src=src, tgt=tgt, 
                description_src=description_src, description_tgt=description_tgt, 
                scale=SCALE_DESC
            )}

            Além da nota, forneça também:
            1. Uma justificativa técnica detalhada para sua avaliação
            2. Seu nível de confiança nesta avaliação (1=muito baixa, 5=muito alta)

            FORMATO DA RESPOSTA:
            NOTA: [0-9]
            JUSTIFICATIVA: [Explique detalhadamente o raciocínio técnico]
            CONFIANÇA: [1-5]
        """
        
        # Chama LLM
        from config import _OPENAI_V0
        if self.provider == "openai":
            if _OPENAI_V0:
                resp = self._oai_client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": extended_prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
            else:  # openai>=1
                resp = self._oai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": extended_prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
        elif self.provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(extended_prompt)
            txt = resp.text.strip()
        else:
            raise RuntimeError("Provider não suportado.")
        
        # Parseia resposta estruturada
        response_record = self._parse_llm_response(txt)
        
        # Adiciona à memória
        self.memory.add_response(src, tgt, response_record)
        
        print(f'{"-"*80}\n')
        print(f'RODADA {self.current_round} - {src} -> {tgt}')
        # print('Pergunta:', extended_prompt)
        print('Resposta:', txt)
        print(f'Parsed: score={response_record.score}, confidence={response_record.confidence}')
        print(f'{"-"*80}\n')
        
        return response_record
    
    def _parse_llm_response(self, response_text: str) -> ResponseRecord:
        """
        Parseia resposta estruturada do LLM extraindo nota, justificativa e confiança.
        """
        import re
        
        # Extrai nota
        score_match = re.search(r'NOTA:\s*([0-9])', response_text)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extrai justificativa
        rationale_match = re.search(r'JUSTIFICATIVA:\s*(.+?)(?=CONFIANÇA:|$)', response_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "Sem justificativa fornecida"
        
        # Extrai confiança
        confidence_match = re.search(r'CONFIANÇA:\s*([1-5])', response_text)
        confidence = int(confidence_match.group(1)) if confidence_match else 3
        
        return ResponseRecord(
            score=score,
            rationale=rationale,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            round_number=self.current_round
        )
    
    def build_initial_matrix(self):
        """
        Constrói matriz inicial com memória estendida.
        Substitui o método _build_direct_matrix original.
        """
        logger.info(f"Iniciando construção da matriz inicial - Rodada {self.current_round}")
        
        for i, src in enumerate(self.factors):
            for j, tgt in enumerate(self.factors):
                if i == j: 
                    continue
                
                # Verifica se pergunta é válida (usando função original se existir)
                
                if not check_pergunta_valida(src, tgt):
                    print("Pergunta intencionalmente deixada de fora, eliminada na triagem inicial")
                    print(f'\t"{src}" não tem influência significativa em "{tgt}"\n\n')
                    self.A[i, j] = 0
                    continue

                # Obtém resposta com memória
                response = self._ask_llm_with_memory(
                    src, tgt, 
                    self.descriptions[i], 
                    self.descriptions[j]
                )
                
                self.A[i, j] = response.score
        
        print('Matriz inicial formada:', self.A)
        
        # Salva matriz
        np.savetxt('matriz_dematel_inicial.txt', self.A, fmt='%d', delimiter=', ')
        
        # Salva memória
        self.memory.export_to_json(self.memory_path)
        
        logger.info("Matriz inicial construída e salva")
    
    def audit_round(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executa uma rodada de auditoria Delphi.
        Compara avaliações do LLM com consenso dos especialistas e permite reavaliação.
        
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Nova matriz e estatísticas da rodada de auditoria
        """
        self.current_round += 1
        logger.info(f"Iniciando rodada de auditoria {self.current_round}")
        
        changes_made = 0
        disagreements = []
        new_matrix = self.A.copy()
        
        for i, src in enumerate(self.factors):
            for j, tgt in enumerate(self.factors):
                if i == j:
                    continue
                
                # Obtém resposta anterior do LLM
                previous_response = self.memory.get_latest_response(src, tgt)
                if not previous_response:
                    continue
                
                # Obtém estatísticas dos especialistas
                expert_stats = self.memory.get_expert_stats(src, tgt)
                if not expert_stats:
                    logger.warning(f"Sem dados de especialistas para {src} -> {tgt}")
                    continue
                
                # Monta prompt de auditoria
                audit_prompt = self.audit_prompt_template.format(
                    src=src,
                    tgt=tgt,
                    description_src=self.descriptions[i],
                    description_tgt=self.descriptions[j],
                    previous_score=previous_response.score,
                    previous_rationale=previous_response.rationale,
                    previous_confidence=previous_response.confidence,
                    expert_median=expert_stats.median,
                    expert_q1=expert_stats.q1,
                    expert_q3=expert_stats.q3,
                    expert_count=expert_stats.count,
                    expert_std=expert_stats.std
                )
                
                # Se a resposta do LLM já está alinhada com especialistas, pula reavaliação
                if abs(previous_response.score - expert_stats.median) <=  expert_stats.std:
                    continue

                # Chama LLM para reavaliação
                audit_response = self._call_llm_for_audit(audit_prompt)
                
                # Adiciona resposta à memória
                self.memory.add_response(src, tgt, audit_response)
                
                # Atualiza matriz
                new_matrix[i, j] = audit_response.score
                
                # Registra mudanças
                if audit_response.score != previous_response.score:
                    changes_made += 1
                
                # Registra desacordos significativos com especialistas
                if abs(audit_response.score - expert_stats.median) > 2:
                    disagreements.append({
                        'pair': f"{src} -> {tgt}",
                        'llm_score': audit_response.score,
                        'expert_median': expert_stats.median,
                        'difference': abs(audit_response.score - expert_stats.median)
                    })
        
        self.A = new_matrix
        
        # Estatísticas da rodada
        round_stats = {
            'round_number': self.current_round,
            'changes_made': changes_made,
            'disagreements': disagreements,
            'total_pairs_evaluated': len([(i, j) for i in range(self.n) for j in range(self.n) if i != j])
        }
        
        # Salva resultados
        np.savetxt(f'matriz_dematel_round_{self.current_round}.txt', 
                   self.A, fmt='%d', delimiter=', ')
        self.memory.export_to_json(self.memory_path)
        
        logger.info(f"Rodada {self.current_round} concluída: {changes_made} mudanças feitas")
        
        return new_matrix, round_stats
    
    def _call_llm_for_audit(self, audit_prompt: str) -> ResponseRecord:
        """Chama LLM para reavaliação durante processo de auditoria"""
        from config import _OPENAI_V0
        
        if self.provider == "openai":
            if _OPENAI_V0:
                resp = self._oai_client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": audit_prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
            else:
                resp = self._oai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": audit_prompt}],
                    temperature=0,
                )
                txt = resp.choices[0].message.content.strip()
        elif self.provider == "gemini":
            import google.generativeai as genai
            model = genai.GenerativeModel(self.model)
            resp = model.generate_content(audit_prompt)
            txt = resp.text.strip()
        else:
            raise RuntimeError("Provider não suportado.")
        
        # mostra tanto a resposta dada pelo LLM como a resposta dos experts
        print(f'AUDITORIA - Rodada {self.current_round}')
        print('Resposta LLM:', txt)
        print(f'')
        print('-' * 50)
        
        return self._parse_audit_response(txt)
    
    def _parse_audit_response(self, response_text: str) -> ResponseRecord:
        """Parseia resposta de auditoria do LLM"""
        import re
        
        # Extrai nota
        score_match = re.search(r'NOTA:\s*([0-9])', response_text)
        score = int(score_match.group(1)) if score_match else 0
        
        # Extrai justificativa
        rationale_match = re.search(r'JUSTIFICATIVA:\s*(.+?)(?=CONFIANÇA:|MUDANÇA:|$)', response_text, re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else "Sem justificativa fornecida"
        
        # Extrai confiança
        confidence_match = re.search(r'CONFIANÇA:\s*([1-5])', response_text)
        confidence = int(confidence_match.group(1)) if confidence_match else 3
        
        return ResponseRecord(
            score=score,
            rationale=rationale,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            round_number=self.current_round
        )
    
    def compute_agreement_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de concordância entre a matriz final do LLM e a mediana dos especialistas.
        
        NOTA: As comparações são realizadas na escala comum 0-9, onde os dados dos 
        especialistas foram automaticamente convertidos de sua escala original (0-4).
        
        Returns
        -------
        Dict[str, float]
            Dicionário com métricas de concordância
        """
        from scipy.stats import spearmanr, kendalltau
        
        llm_scores = []
        expert_medians = []
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                
                src, tgt = self.factors[i], self.factors[j]
                expert_stats = self.memory.get_expert_stats(src, tgt)
                
                if expert_stats:
                    llm_scores.append(self.A[i, j])
                    expert_medians.append(expert_stats.median)
        
        if not llm_scores:
            logger.warning("Sem dados suficientes para calcular métricas de concordância")
            return {}
        
        llm_array = np.array(llm_scores)
        expert_array = np.array(expert_medians)
        
        # Diferença absoluta média
        mad = np.mean(np.abs(llm_array - expert_array))
        
        # Correlação de Spearman
        spearman_corr, spearman_p = spearmanr(llm_array, expert_array)
        
        # Correlação de Kendall Tau
        kendall_corr, kendall_p = kendalltau(llm_array, expert_array)
        
        # Proporção de acordos exatos
        exact_agreements = np.sum(llm_array == expert_array) / len(llm_array)
        
        # Proporção de acordos dentro de 1 ponto
        close_agreements = np.sum(np.abs(llm_array - expert_array) <= 1) / len(llm_array)
        
        metrics = {
            'mean_absolute_difference': mad,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_tau': kendall_corr,
            'kendall_p_value': kendall_p,
            'exact_agreement_rate': exact_agreements,
            'close_agreement_rate': close_agreements,
            'total_comparisons': len(llm_scores)
        }
        
        logger.info(f"Métricas de concordância calculadas: MAD={mad:.3f}, Spearman={spearman_corr:.3f}")
        
        return metrics
    
    def generate_audit_report(self, output_path: str = "audit_report.json"):
        """
        Gera relatório completo do processo de auditoria Delphi.
        
        Parameters
        ----------
        output_path : str
            Caminho para salvar o relatório
        """
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'factors': self.factors,
                'total_rounds': self.current_round,
                'provider': self.provider,
                'model': self.model
            },
            'agreement_metrics': self.compute_agreement_metrics(),
            'round_summary': [],
            'factor_analysis': {}
        }
        
        # Análise por fator
        for i, factor in enumerate(self.factors):
            factor_stats = {
                'as_source': {
                    'total_influence': float(np.sum(self.A[i, :])),
                    'avg_confidence': 0,
                    'changes_made': 0
                },
                'as_target': {
                    'total_influence_received': float(np.sum(self.A[:, i])),
                    'avg_confidence': 0,
                    'changes_made': 0
                }
            }
            
            # Calcula estatísticas de confiança e mudanças
            source_confidences = []
            target_confidences = []
            source_changes = 0
            target_changes = 0
            
            for j in range(self.n):
                if i != j:
                    # Como fonte
                    responses = self.memory.get_all_responses(self.factors[i], self.factors[j])
                    if responses:
                        source_confidences.append(responses[-1].confidence)
                        if len(responses) > 1:
                            source_changes += 1 if responses[-1].score != responses[0].score else 0
                    
                    # Como alvo
                    responses = self.memory.get_all_responses(self.factors[j], self.factors[i])
                    if responses:
                        target_confidences.append(responses[-1].confidence)
                        if len(responses) > 1:
                            target_changes += 1 if responses[-1].score != responses[0].score else 0
            
            if source_confidences:
                factor_stats['as_source']['avg_confidence'] = float(np.mean(source_confidences))
                factor_stats['as_source']['changes_made'] = source_changes
            
            if target_confidences:
                factor_stats['as_target']['avg_confidence'] = float(np.mean(target_confidences))
                factor_stats['as_target']['changes_made'] = target_changes
            
            report['factor_analysis'][factor] = factor_stats
        
        # Salva relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Relatório de auditoria salvo em {output_path}")
        
        return report
    
    def run_full_delphi_process(self, max_rounds: int = 2) -> Dict[str, Any]:
        
        """
        Executa o processo completo Delphi: matriz inicial + rodadas de auditoria.
        
        Parameters
        ----------
        max_rounds : int
            Número máximo de rodadas de auditoria
        
        Returns
        -------
        Dict[str, Any]
            Resultados completos do processo
        """
        logger.info(f"Iniciando processo Delphi completo - máximo {max_rounds} rodadas")
        
        # Rodada inicial
        self.build_initial_matrix()
        
        # Executa DEMATEL na matriz inicial
        self._dematel()
        
        # Constrói grafo inicial
        self.G = self._build_graph(numeric_filter=True)
        
        results = {
            'initial_matrix': self.A.copy(),
            'rounds': [],
            'final_metrics': None
        }
        
        # rodadas de auditoria
        for round_num in range(max_rounds):
            if not any(self.memory.expert_stats.values()):
                logger.warning("Sem dados de especialistas disponíveis. Pulando auditoria.")
                break
            
            matrix, round_stats = self.audit_round()
            
            # Recalcula DEMATEL com nova matriz
            self._dematel()
            
            # Reconstrói grafo com nova matriz
            self.G = self._build_graph(numeric_filter=True)
            
            results['rounds'].append({
                'round_number': round_num + 2,  # +2 porque rodada 1 é inicial
                'matrix': matrix.copy(),
                'statistics': round_stats
            })
        
        # Métricas finais
        results['final_metrics'] = self.compute_agreement_metrics()
        results['final_matrix'] = self.A.copy()
        
        # Gera relatório
        report = self.generate_audit_report()
        results['audit_report'] = report
        
        logger.info("Processo Delphi concluído com sucesso")
        
        return results


# Importa e adiciona funções de plot à classe
try:
    from functions.main import plot_network, plot_influence_diagram
    DelphiDematel.plot_network = plot_network
    DelphiDematel.plot_influence_diagram = plot_influence_diagram
except ImportError as e:
    logger.warning(f"Não foi possível importar funções de plot: {e}")
    
    # Método de fallback simples para plot_network
    def simple_plot_network(self, title="Network Plot"):
        """Método simples de fallback para visualização"""
        print(f"Plotando rede: {title}")
        print(f"Nós: {len(self.factors)}")
        if hasattr(self, 'G'):
            print(f"Arestas: {self.G.number_of_edges()}")
        else:
            print("Grafo não disponível")
    
    DelphiDematel.plot_network = simple_plot_network