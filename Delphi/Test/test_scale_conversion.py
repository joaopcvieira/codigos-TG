"""
test_scale_conversion.py
Testa a conversão de escalas para garantir que está funcionando corretamente.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from delphi_dematel import ExpertAggregator, ExpertStatistics

def test_conversion_function():
    """Testa a função de conversão individual"""
    print("="*60)
    print("TESTE 1: Função de Conversão Individual")
    print("="*60)
    
    # Testa valores específicos
    test_cases = [
        (0, 0.0),      # Mínimo
        (1, 2.25),     # Baixo
        (2, 4.5),      # Médio
        (3, 6.75),     # Alto
        (4, 9.0),      # Máximo
        (2.5, 5.625),  # Meio-termo
        (3.5, 7.875)   # Outro meio-termo
    ]
    
    all_passed = True
    for original, expected in test_cases:
        converted = ExpertAggregator.convert_expert_scale_to_llm(original)
        passed = abs(converted - expected) < 0.001
        status = "✓" if passed else "✗"
        print(f"{status} {original} → {converted:.3f} (esperado: {expected})")
        if not passed:
            all_passed = False
    
    return all_passed

def test_matrix_conversion():
    """Testa conversão de matrizes completas"""
    print("\n" + "="*60)
    print("TESTE 2: Conversão de Matrizes de Especialistas")
    print("="*60)
    
    # Cria matrizes de teste na escala 0-4
    expert1 = np.array([[0, 1, 2], [3, 0, 4], [2, 1, 0]])
    expert2 = np.array([[0, 2, 3], [1, 0, 2], [4, 3, 0]])
    expert_matrices = [expert1, expert2]
    
    print("Matriz Especialista 1 (escala 0-4):")
    print(expert1)
    print("\nMatriz Especialista 2 (escala 0-4):")
    print(expert2)
    
    # Aplica conversão
    aggregated = ExpertAggregator.aggregate_expert_responses(expert_matrices)
    
    print(f"\nResultados agregados (escala convertida 0-9):")
    for (i, j), stats in aggregated.items():
        original_responses = []
        for matrix in expert_matrices:
            original_responses.append(matrix[i, j])
        
        print(f"Par ({i},{j}): {original_responses} → mediana={stats.median:.2f}")
        
        # Valida conversão manual
        converted_manual = [ExpertAggregator.convert_expert_scale_to_llm(r) for r in original_responses]
        expected_median = np.median(converted_manual)
        
        passed = abs(stats.median - expected_median) < 0.001
        status = "✓" if passed else "✗"
        print(f"  {status} Conversão correta (esperado: {expected_median:.2f})")
    
    return True

def test_edge_cases():
    """Testa casos extremos"""
    print("\n" + "="*60)
    print("TESTE 3: Casos Extremos")
    print("="*60)
    
    # Testa valores fora do intervalo
    edge_cases = [
        (-1, 0.0),     # Abaixo do mínimo
        (5, 9.0),      # Acima do máximo
        (0.1, 0.225),  # Valor muito baixo
        (3.9, 8.775),  # Próximo do máximo
    ]
    
    all_passed = True
    for original, expected in edge_cases:
        converted = ExpertAggregator.convert_expert_scale_to_llm(original)
        # Para casos fora do intervalo, verifica se está dentro dos limites
        if original < 0 or original > 4:
            passed = 0 <= converted <= 9
            print(f"{'✓' if passed else '✗'} {original} → {converted:.3f} (limitado a 0-9)")
        else:
            passed = abs(converted - expected) < 0.001
            print(f"{'✓' if passed else '✗'} {original} → {converted:.3f} (esperado: {expected})")
        
        if not passed:
            all_passed = False
    
    return all_passed

def test_statistical_properties():
    """Testa se as propriedades estatísticas são preservadas"""
    print("\n" + "="*60)
    print("TESTE 4: Propriedades Estatísticas")
    print("="*60)
    
    # Dados de exemplo
    original_data = np.array([0, 1, 2, 3, 4, 2, 1, 3, 2, 4])
    converted_data = np.array([ExpertAggregator.convert_expert_scale_to_llm(x) for x in original_data])
    
    print(f"Dados originais (0-4): {original_data}")
    print(f"Dados convertidos (0-9): {converted_data}")
    
    # Testa linearidade da conversão
    correlation = np.corrcoef(original_data, converted_data)[0, 1]
    print(f"\nCorrelação: {correlation:.6f} (deve ser 1.0 para conversão linear)")
    
    # Testa se a ordem é preservada
    original_order = np.argsort(original_data)
    converted_order = np.argsort(converted_data)
    order_preserved = np.array_equal(original_order, converted_order)
    print(f"Ordem preservada: {'✓' if order_preserved else '✗'}")
    
    return correlation > 0.999 and order_preserved

def main():
    """Executa todos os testes"""
    print("SISTEMA DE TESTES - CONVERSÃO DE ESCALAS DELPHI")
    print("="*60)
    
    # Primeiro, demonstra a conversão
    ExpertAggregator.test_scale_conversion()
    
    # Executa testes
    tests = [
        ("Conversão Individual", test_conversion_function),
        ("Conversão de Matrizes", test_matrix_conversion), 
        ("Casos Extremos", test_edge_cases),
        ("Propriedades Estatísticas", test_statistical_properties)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ ERRO no teste {name}: {e}")
            results.append((name, False))
    
    # Resumo final
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSOU" if passed else "✗ FALHOU"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM - CONVERSÃO FUNCIONANDO CORRETAMENTE!")
    else:
        print("❌ ALGUNS TESTES FALHARAM - REVISAR IMPLEMENTAÇÃO")
    print(f"{'='*60}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)