"""
Exemplo de uso do sistema DelphiDematel
Demonstra como usar o processo de auditoria Delphi para validar avaliações do LLM
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from delphi_dematel import DelphiDematel, ExpertAggregator


def manage_round_cache(round_name: str, round_number: int, cache_data: dict = None):
    """
    Permite ao usuário escolher entre usar cache ou fazer novas consultas para uma rodada específica
    
    Parameters:
    - round_name: Nome descritivo da rodada (ex: "Rodada Inicial", "Rodada 2 - Auditoria")
    - round_number: Número da rodada (0=inicial, 1=segunda, 2=terceira, etc.)
    - cache_data: Dados do cache já carregados
    
    Returns:
    - use_cache: True se deve usar cache, False se fazer novas consultas
    - round_cache: Dados específicos da rodada se existir cache
    """
    print("\n" + "="*70)
    print(f"CONTROLE DE CACHE - {round_name.upper()}")
    print("="*70)
    
    # Verifica se existe cache para esta rodada específica
    round_cache = None
    if cache_data and 'rounds' in cache_data:
        for cached_round in cache_data['rounds']:
            if cached_round.get('round_number') == round_number:
                round_cache = cached_round
                break
    
    if round_cache:
        print(f"✅ Cache encontrado para {round_name}")
        print(f"   Data: {round_cache.get('timestamp', 'Desconhecida')}")
        print(f"   Descrição: {round_cache.get('description', 'N/A')}")
        if 'matrix' in round_cache:
            import numpy as np
            matrix = np.array(round_cache['matrix'])
            print(f"   Matriz: {matrix.shape[0]}x{matrix.shape[1]} elementos")
        if 'justifications' in round_cache:
            justif_count = len(round_cache['justifications'])
            print(f"   Justificativas: {justif_count} pares de fatores")
        
        print(f"\n🔄 Opções para {round_name}:")
        print("1. 📂 Usar cache salvo (RECOMENDADO - economiza API)")
        print("2. 🆕 Fazer novas consultas ao LLM (sobrescreve cache desta rodada)")
        
        while True:
            choice = input(f"\nEscolha para {round_name} (1 ou 2): ").strip()
            if choice == "1":
                print(f"✅ Usando cache para {round_name}")
                return True, round_cache
            elif choice == "2":
                print(f"🔄 Fazendo novas consultas para {round_name}")
                return False, None
            else:
                print("❌ Opção inválida. Digite 1 ou 2.")
    else:
        print(f"ℹ️  Nenhum cache encontrado para {round_name}")
        print(f"🔄 Prosseguindo com consultas ao LLM para {round_name}")
        return False, None


def load_global_cache():
    """
    Carrega o cache global se existir
    Returns: cache_data ou None
    """
    cache_file = "llm_results_cache.json"
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            print(f"\n📂 Cache global encontrado: {cache_file}")
            print(f"   Data: {cache_data.get('timestamp', 'Desconhecida')}")
            print(f"   Rodadas disponíveis: {len(cache_data.get('rounds', []))}")
            
            if cache_data.get('rounds'):
                print("   Detalhes das rodadas:")
                for cached_round in cache_data['rounds']:
                    round_num = cached_round.get('round_number', '?')
                    desc = cached_round.get('description', 'Sem descrição')
                    timestamp = cached_round.get('timestamp', 'N/A')
                    print(f"     - Rodada {round_num}: {desc} ({timestamp})")
            
            return cache_data
        except Exception as e:
            print(f"⚠️  Erro ao ler cache global: {e}")
            return None
    else:
        print("ℹ️  Nenhum cache global encontrado")
        return None


def save_round_to_cache(round_number: int, round_data: dict, description: str = None):
    """
    Salva uma rodada específica no cache, preservando outras rodadas existentes
    
    Parameters:
    - round_number: Número da rodada (0=inicial, 1=segunda, etc.)
    - round_data: Dados da rodada (matrix, justifications, etc.)
    - description: Descrição opcional da rodada
    """
    cache_file = "llm_results_cache.json"
    
    # Carrega cache existente ou cria novo
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Erro ao ler cache existente: {e}")
            cache_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': 'Cache de resultados LLM',
                'rounds': []
            }
    else:
        cache_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'description': 'Cache de resultados LLM',
            'rounds': []
        }
    
    # Atualiza timestamp global
    cache_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepara dados da nova rodada
    new_round = {
        'round_number': round_number,
        'description': description or f'Rodada {round_number}',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Adiciona dados específicos da rodada
    if 'matrix' in round_data:
        new_round['matrix'] = round_data['matrix'].tolist() if hasattr(round_data['matrix'], 'tolist') else round_data['matrix']
    if 'justifications' in round_data:
        new_round['justifications'] = round_data['justifications']
    if 'audit_metrics' in round_data:
        new_round['audit_metrics'] = round_data['audit_metrics']
    
    # Remove rodada existente com mesmo número (se houver) e adiciona nova
    cache_data['rounds'] = [r for r in cache_data['rounds'] if r.get('round_number') != round_number]
    cache_data['rounds'].append(new_round)
    
    # Ordena por número da rodada
    cache_data['rounds'].sort(key=lambda x: x.get('round_number', 0))
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Rodada {round_number} salva no cache: {cache_file}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar rodada no cache: {e}")


def save_complete_results_to_cache(results):
    """
    Salva resultados completos no cache (compatibilidade com versão anterior)
    """
    cache_file = "llm_results_cache.json"
    
    cache_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': 'Cache completo de resultados LLM',
        'rounds': []
    }
    
    # Salva rodada inicial
    if 'initial_round' in results:
        cache_data['rounds'].append({
            'round_number': 0,
            'description': 'Rodada Inicial - Matriz Base',
            'matrix': results['initial_round']['matrix'].tolist(),
            'justifications': results['initial_round'].get('justifications', {}),
            'timestamp': results['initial_round'].get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        })
    
    # Salva rodadas subsequentes
    for i, round_data in enumerate(results.get('rounds', [])):
        cache_data['rounds'].append({
            'round_number': i + 1,
            'description': f'Rodada {i + 1} - Auditoria Delphi',
            'matrix': round_data['matrix'].tolist(),
            'justifications': round_data.get('justifications', {}),
            'timestamp': round_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            'audit_metrics': round_data.get('audit_metrics')
        })
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Resultados completos salvos no cache: {cache_file}")
    except Exception as e:
        print(f"⚠️  Erro ao salvar cache completo: {e}")


def load_results_from_cache(cache_data):
    """
    Carrega resultados do cache e reconstrói a estrutura esperada
    """
    results = {
        'rounds': [],
        'final_matrix': None,
        'final_metrics': None
    }
    
    # Processa rodadas do cache
    for round_data in cache_data['rounds']:
        matrix = np.array(round_data['matrix'])
        
        if round_data['round_number'] == 0:
            results['initial_round'] = {
                'matrix': matrix,
                'justifications': round_data.get('justifications', {}),
                'timestamp': round_data.get('timestamp')
            }
        else:
            results['rounds'].append({
                'matrix': matrix,
                'justifications': round_data.get('justifications', {}),
                'timestamp': round_data.get('timestamp'),
                'audit_metrics': round_data.get('audit_metrics')
            })
    
    # Define matriz final
    if results['rounds']:
        results['final_matrix'] = results['rounds'][-1]['matrix']
    elif 'initial_round' in results:
        results['final_matrix'] = results['initial_round']['matrix']
    
    print(f"📂 Resultados carregados do cache com {len(results['rounds'])} rodadas")
    return results


def ensure_expert_data_exists(factors, expert_data_path='inputs/expert_responses.csv'):
    """
    Garante que existe arquivo de dados de especialistas, criando um se necessário
    """
    if os.path.exists(expert_data_path):
        print(f"✅ Dados de especialistas encontrados: {expert_data_path}")
        return
    
    print(f"⚠️  Arquivo de especialistas não encontrado: {expert_data_path}")
    print("🔄 Criando dados simulados de especialistas...")
    
    # Cria diretório se não existir
    os.makedirs(os.path.dirname(expert_data_path), exist_ok=True)
    
    # Gera dados simulados (escala 0-4 para especialistas)
    np.random.seed(42)  # Para reproducibilidade
    n_factors = len(factors)
    
    # Cria matriz simulada com valores realistas
    expert_matrix = np.zeros((n_factors, n_factors))
    
    for i in range(n_factors):
        for j in range(n_factors):
            if i != j:  # Não auto-influência
                # Gera influência baseada em padrões realistas
                base_influence = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.15, 0.05])
                expert_matrix[i, j] = base_influence
    
    # Converte para formato CSV esperado
    data = []
    expert_id = 1
    
    for i in range(n_factors):
        for j in range(n_factors):
            if i != j:
                data.append({
                    'expert_id': expert_id,
                    'factor_source': factors[i],
                    'factor_target': factors[j],
                    'influence_score': int(expert_matrix[i, j])
                })
    
    # Salva CSV
    df = pd.DataFrame(data)
    df.to_csv(expert_data_path, index=False)
    
    print(f"✅ Dados simulados criados: {expert_data_path}")
    print(f"   - {len(data)} avaliações geradas")
    print(f"   - Escala: 0-4 (compatível com conversão automática)")


def run_granular_delphi_process(delphi_system, global_cache, max_rounds=2):
    """
    Executa processo Delphi com controle granular de cache por rodada
    """
    results = {
        'initial_round': None,
        'rounds': [],
        'final_matrix': None,
        'final_metrics': None
    }
    
    # === RODADA INICIAL ===
    use_initial_cache, initial_cache = manage_round_cache("Rodada Inicial", 0, global_cache)
    
    if use_initial_cache and initial_cache:
        print("📂 Carregando matriz inicial do cache...")
        initial_matrix = np.array(initial_cache['matrix'])
        initial_justifications = initial_cache.get('justifications', {})
        delphi_system.A = initial_matrix
        
        results['initial_round'] = {
            'matrix': initial_matrix,
            'justifications': initial_justifications,
            'timestamp': initial_cache.get('timestamp')
        }
    else:
        print("🔄 Gerando matriz inicial com consultas ao LLM...")
        delphi_system.build_initial_matrix()
        
        # Salva no cache
        initial_data = {
            'matrix': delphi_system.A,
            'justifications': delphi_system.memory.get_all_justifications(),
        }
        save_round_to_cache(0, initial_data, "Rodada Inicial - Matriz Base")
        
        results['initial_round'] = {
            'matrix': delphi_system.A.copy(),
            'justifications': delphi_system.memory.get_all_justifications(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # Executa DEMATEL na matriz inicial
    delphi_system._dematel()
    delphi_system.G = delphi_system._build_graph(numeric_filter=True)
    
    # === RODADAS DE AUDITORIA ===
    for round_num in range(1, max_rounds + 1):
        round_name = f"Rodada {round_num + 1} - Auditoria"
        use_round_cache, round_cache = manage_round_cache(round_name, round_num, global_cache)
        
        if use_round_cache and round_cache:
            print(f"📂 Carregando {round_name} do cache...")
            round_matrix = np.array(round_cache['matrix'])
            round_justifications = round_cache.get('justifications', {})
            round_metrics = round_cache.get('audit_metrics', {})
            
            results['rounds'].append({
                'matrix': round_matrix,
                'justifications': round_justifications,
                'audit_metrics': round_metrics,
                'timestamp': round_cache.get('timestamp')
            })
            
            # Atualiza matriz do sistema
            delphi_system.A = round_matrix
            delphi_system._dematel()
            delphi_system.G = delphi_system._build_graph(numeric_filter=True)
            
        else:
            print(f"🔄 Executando {round_name} com consultas ao LLM...")
            audit_matrix, audit_metrics = delphi_system.audit_round()
            
            # Salva no cache
            round_data = {
                'matrix': audit_matrix,
                'justifications': delphi_system.memory.get_justifications_by_round(round_num + 1),
                'audit_metrics': audit_metrics
            }
            save_round_to_cache(round_num, round_data, round_name)
            
            results['rounds'].append({
                'matrix': audit_matrix.copy(),
                'justifications': delphi_system.memory.get_justifications_by_round(round_num + 1),
                'audit_metrics': audit_metrics,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        delphi_system.current_round += 1
    
    # Define matriz final
    if results['rounds']:
        results['final_matrix'] = results['rounds'][-1]['matrix']
    else:
        results['final_matrix'] = results['initial_round']['matrix']
    
    return results


def calculate_agreement_metrics(final_matrix, expert_data_path, factors):
    """
    Calcula métricas de concordância entre LLM e especialistas
    """
    try:
        expert_agg = ExpertAggregator.load_and_aggregate_from_file(expert_data_path, factors)
        
        # Converte dados de especialistas para matriz
        n = len(factors)
        expert_matrix = np.zeros((n, n))
        
        for (factor_src, factor_tgt), stats in expert_agg.items():
            i = factors.index(factor_src)
            j = factors.index(factor_tgt)
            expert_matrix[i, j] = stats.median  # Já convertido para escala 0-9
        
        from scipy.stats import spearmanr
        
        # Compara matrizes (excluindo diagonal)
        final_flat = final_matrix.flatten()
        expert_flat = expert_matrix.flatten()
        
        mask = ~np.eye(n, dtype=bool)
        final_clean = final_flat[mask.flatten()]
        expert_clean = expert_flat[mask.flatten()]
        
        # Calcula métricas
        mean_abs_diff = np.mean(np.abs(final_clean - expert_clean))
        spearman_corr, _ = spearmanr(final_clean, expert_clean)
        exact_agreement = np.mean(np.abs(final_clean - expert_clean) < 0.5)
        close_agreement = np.mean(np.abs(final_clean - expert_clean) <= 1.0)
        
        return {
            'mean_absolute_difference': mean_abs_diff,
            'spearman_correlation': spearman_corr,
            'exact_agreement_rate': exact_agreement,
            'close_agreement_rate': close_agreement
        }
    except Exception as e:
        print(f"⚠️  Erro ao calcular métricas: {e}")
        return {}


def demonstrate_delphi_process():
    """Demonstra o processo completo Delphi-DEMATEL com cache granular"""
    
    print("="*80)
    print("DEMONSTRAÇÃO DO PROCESSO DELPHI-DEMATEL")
    print("="*80)
    
    print("\n📋 FORMATO DE DADOS DE ESPECIALISTAS:")
    print("   Arquivo: input/expert_responses.csv")
    print("   Formato: expert_id,factor_source,factor_target,influence_score")
    print("   Escala: 0-4 (convertida automaticamente para 0-9)")
    print("   Exemplo disponível em: input/expert_responses_example.csv")
    print("   Documentação completa: README_expert_format.md")
    
    # Carrega cache global (se existir)
    global_cache = load_global_cache()
    
    # 1. Carrega fatores e descrições
    df_fatores = pd.read_csv('inputs/Fatores.csv')
    fatores = df_fatores['fator'].tolist()
    descricoes = df_fatores['descricao'].tolist()
    
    print(f"\nCarregados {len(fatores)} fatores para análise")
    
    # 2. Garante que existem dados de especialistas
    expert_data_path = 'inputs/expert_responses.csv'
    ensure_expert_data_exists(fatores, expert_data_path)

    # 3. Inicializa sistema Delphi
    print("\nInicializando sistema DelphiDematel...")
    delphi_system = DelphiDematel(
        factors=fatores,
        descriptions=descricoes,
        expert_data_path=expert_data_path,
        memory_path="delphi_memory.json",
        provider="openai"
    )
    
    # 4. Executa processo com controle granular de cache
    print("\n🎯 INICIANDO PROCESSO DELPHI COM CONTROLE GRANULAR DE CACHE")
    results = run_granular_delphi_process(delphi_system, global_cache, max_rounds=2)
    
    # 5. Calcula métricas finais se necessário
    if not results.get('final_metrics') and results.get('final_matrix') is not None:
        print("\n🔄 Calculando métricas de concordância...")
        results['final_metrics'] = calculate_agreement_metrics(results['final_matrix'], expert_data_path, fatores)
    
    # 5. Mostra resultados
    print("\n" + "="*50)
    print("RESULTADOS DO PROCESSO DELPHI")
    print("="*50)
    
    print(f"\nRodadas executadas: {len(results['rounds']) + 1}")
    
    if results['final_metrics']:
        metrics = results['final_metrics']
        print(f"\nMétricas de concordância com especialistas (escala normalizada 0-9):")
        print(f"- Diferença absoluta média: {metrics.get('mean_absolute_difference', 0):.3f}")
        print(f"- Correlação de Spearman: {metrics.get('spearman_correlation', 0):.3f}")
        print(f"- Taxa de acordo exato: {metrics.get('exact_agreement_rate', 0)*100:.1f}%")
        print(f"- Taxa de acordo próximo (±1): {metrics.get('close_agreement_rate', 0)*100:.1f}%")
        print(f"NOTA: Especialistas (0-4) foram convertidos para escala LLM (0-9)")
    
    # 6. Gera visualizações
    print(f"\nGerando visualizações...")
    delphi_system.plot_network(title="Rede de Influência - Processo Delphi")
    
    print(f"\nProcesso concluído! Arquivos gerados:")
    print(f"- delphi_memory.json: Memória completa do processo")
    print(f"- audit_report.json: Relatório detalhado de auditoria")
    print(f"- matriz_dematel_inicial.txt: Matriz da rodada inicial")
    for i in range(len(results['rounds'])):
        print(f"- matriz_dematel_round_{i+2}.txt: Matriz da rodada {i+2}")
    
    return results

def analyze_bias_and_changes():
    """Analisa viés e mudanças no processo a partir da memória salva"""
    from delphi_dematel import DelphiMemory
    
    print("\n" + "="*50)
    print("ANÁLISE DE VIÉS E MUDANÇAS")
    print("="*50)
    
    # Carrega memória
    memory = DelphiMemory()
    memory.load_from_json("delphi_memory.json")
    
    total_pairs = len(memory.responses)
    changed_pairs = 0
    high_disagreement = 0
    
    print(f"\nAnalisando {total_pairs} pares de fatores...")
    
    disagreement_details = []
    
    for (src, tgt), responses in memory.responses.items():
        if len(responses) > 1:  # Houve mais de uma rodada
            initial_score = responses[0].score
            final_score = responses[-1].score
            
            if initial_score != final_score:
                changed_pairs += 1
                change_info = {
                    'pair': f"{src} -> {tgt}",
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'change': final_score - initial_score,
                    'initial_confidence': responses[0].confidence,
                    'final_confidence': responses[-1].confidence
                }
                
                # Verifica se há dados de especialistas
                expert_stats = memory.get_expert_stats(src, tgt)
                if expert_stats:
                    change_info['expert_median'] = expert_stats.median
                    change_info['initial_diff_from_experts'] = abs(initial_score - expert_stats.median)
                    change_info['final_diff_from_experts'] = abs(final_score - expert_stats.median)
                    
                    if change_info['final_diff_from_experts'] > 2:
                        high_disagreement += 1
                        disagreement_details.append(change_info)
    
    print(f"\nResumo das mudanças:")
    print(f"- Pares que mudaram: {changed_pairs}/{total_pairs} ({changed_pairs/total_pairs*100:.1f}%)")
    print(f"- Desacordos significativos com especialistas: {high_disagreement}")
    
    if disagreement_details:
        print(f"\nMaiores desacordos (diferença > 2 pontos):")
        for detail in sorted(disagreement_details, key=lambda x: x['final_diff_from_experts'], reverse=True)[:5]:
            print(f"- {detail['pair']}: LLM={detail['final_score']}, Especialistas={detail['expert_median']:.1f}")
    
    return {
        'total_pairs': total_pairs,
        'changed_pairs': changed_pairs,
        'high_disagreement': high_disagreement,
        'disagreement_details': disagreement_details
    }

if __name__ == "__main__":
    # Garante que o diretório de trabalho está correto
    os.chdir('/Users/joaovieira/Documents/TG/codigos-TG/Delphi')
    
    try:
        # Executa demonstração completa
        results = demonstrate_delphi_process()
        
        # Analisa resultados
        bias_analysis = analyze_bias_and_changes()
        
        print(f"\n{'='*80}")
        print("PROCESSO DELPHI-DEMATEL CONCLUÍDO COM SUCESSO!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()