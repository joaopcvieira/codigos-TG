# Projeto TG - Análise de Fatores em Motores Foguete Híbridos

Este repositório contém os códigos e análises desenvolvidos para o Trabalho de Graduação (TG) sobre **análise de fatores críticos no desenvolvimento de motores foguete híbridos**, com foco na aplicação de técnicas de análise multicritério e inteligência artificial.

## 📋 Descrição do Projeto

O projeto utiliza metodologias avançadas para identificar e analisar as relações causais entre diferentes fatores que influenciam o sucesso no desenvolvimento de motores foguete híbridos. O objetivo principal é mapear as dependências entre variáveis técnicas, de gestão e operacionais que impactam o **empuxo gerado pelo motor** - requisito principal de sucesso do projeto.

## 🗂️ Estrutura do Repositório

### 📁 `Agente AI/`
Implementação de um **agente inteligente automatizado** para construção de matrizes DEMATEL utilizando LLMs (Large Language Models).

**Principais características:**
- Automação da análise DEMATEL com suporte a OpenAI GPT e Google Gemini
- Geração automática de matrizes de influência entre fatores
- Visualização interativa com Plotly e NetworkX
- Análise de centralidade e relações causais

**Arquivos principais:**
- `main.py`: Sistema principal do agente AI para análise DEMATEL
- `fatores.txt`: Lista dos fatores analisados (técnicos e de gestão)
- `resultado TG1/`: Resultados e diagramas gerados pelo sistema

### 📁 `forms/`
Processamento e análise dos **dados coletados via questionários**.

**Conteúdo:**
- `Fatores.csv`: Base de dados dos fatores identificados por categoria
- `relacao_fatores.csv`: Matriz de relações entre fatores
- `main.ipynb`: Notebook para processamento e análise dos dados coletados
- `perguntas.md`: Estrutura das perguntas aplicadas no questionário
- Planilhas com dados brutos coletados

### 📁 `Exploracao inicial/`
**Estudos exploratórios** e implementações de referência para metodologias de análise.

**Conteúdo:**
- `exemple_code.py/ipynb`: Exemplos de implementação de Redes Bayesianas
- `bnstruct_exemples.R`: Implementações em R para análise estrutural
- `teste.ipynb`: Experimentos e validações de conceitos

## 🔬 Metodologias Aplicadas

### 1. **Análise DEMATEL (Decision Making Trial and Evaluation Laboratory)**
- Identificação de relações causais entre fatores
- Construção de mapas de influência
- Classificação de fatores como causas ou efeitos

### 2. **Inteligência Artificial para Automação**
- Uso de LLMs para avaliação automatizada de relações entre fatores
- Processamento de linguagem natural para interpretação de contexto técnico
- Geração automatizada de matrizes de decisão

### 3. **Análise de Redes e Grafos**
- Visualização de dependências entre variáveis
- Análise de centralidade e importância de fatores
- Identificação de pontos críticos no sistema

## 🎯 Fatores Analisados

O projeto analisa dois grupos principais de fatores:

### **Fatores de Gestão e Equipe:**
- Experiência da Equipe
- Disponibilidade de Tempo
- Orçamento Disponível
- Qualidade da Comunicação
- Documentação de Projeto

### **Fatores Técnicos do Motor:**
- Geometria da Câmara (comprimento, diâmetro, volume)
- Parâmetros de Combustão (massa de parafina, razão O/F)
- Sistema de Alimentação (vazão de oxigênio, pressão do tanque)
- Controle e Estabilidade (pressão de câmara, energia de ignição)

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install openai numpy pandas networkx plotly matplotlib python-dotenv google-generativeai
```

### Configuração
1. Configure suas chaves de API no arquivo `.env`:
```env
OPENAI_API_KEY=sua_chave_openai
GEMINI_API_KEY=sua_chave_gemini
LLM_PROVIDER=openai  # ou "gemini"
```

### Execução
```bash
# Análise automatizada com IA
cd "Agente AI"
python main.py

# Análise manual dos dados coletados
cd forms
jupyter notebook main.ipynb
```

## 📊 Resultados

O projeto gera:
- **Diagramas DEMATEL**: Visualização das relações causais
- **Grafos de Influência**: Mapas de dependências entre fatores
- **Rankings de Importância**: Classificação dos fatores por criticidade
- **Matrizes de Decisão**: Bases quantitativas para tomada de decisão

## 🔧 Tecnologias Utilizadas

- **Python**: Linguagem principal para análises
- **OpenAI GPT / Google Gemini**: LLMs para automação de análises
- **Pandas/NumPy**: Manipulação e processamento de dados
- **NetworkX**: Análise de grafos e redes
- **Plotly/Matplotlib**: Visualização de dados
- **Jupyter Notebooks**: Ambiente de desenvolvimento e documentação

## 📝 Contexto Acadêmico

Este projeto integra conhecimentos de:
- **Engenharia Aeroespacial**: Desenvolvimento de sistemas de propulsão
- **Análise de Decisão**: Metodologias MCDM (Multi-Criteria Decision Making)
- **Inteligência Artificial**: Aplicação de LLMs em contextos técnicos
- **Gestão de Projetos**: Identificação de fatores críticos de sucesso

---

**Autor**: João Pedro Coelho Vieira  
**Instituição**: [Sua Instituição]  
**Orientador**: [Nome do Orientador]  
**Ano**: 2025

