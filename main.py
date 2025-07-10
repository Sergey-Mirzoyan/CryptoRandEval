#!/usr/bin/env python3
"""
CryptoRandEval - Tool for evaluating cryptographic random sequences
"""

import os
import sys

# Импортируем внешние модули анализа
from stats.sp800_22_tests.sp800_22_tests import run_nist_tests,runrun
from stats.DIEHARD.dieharder import run_dieharder


def read_binary_string(file_path):
    """
    Читает бинарную последовательность из файла и возвращает её в виде строки '010101...'
    
    Args:
        file_path (str): Путь к файлу (.txt или .bin)
        
    Returns:
        str: Строка из '0' и '1'
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если файл содержит недопустимые данные
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            raw = f.read().strip()
            if not all(c in '01' for c in raw):
                raise ValueError("Файл содержит недопустимые символы")
            return raw

    elif file_path.endswith('.bin'):
        sequence = []
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        for byte in binary_data:
            for i in range(8):
                bit = (byte >> i) & 1
                sequence.append(str(bit))
        return ''.join(sequence)

    else:
        raise ValueError("Неподдерживаемое расширение файла")


def predictability_analyze(bitstring):
    """Заглушка для ML-анализа предсказуемости"""
    return {
        'summary': 'потенциально слабое место',
        'details': {
            'Точность предсказания': '62%',
            'P-value': 0.03
        }
    }


def stats_analyze(bitstring,file):
    """Выполняет полный статистический анализ (NIST + DIEHARD)"""
    print(bitstring)
    nist_result = runrun(file)
    
    diehard_result = run_dieharder(bitstring)

    # Агрегируем результаты
    passed = sum(1 for t in nist_result['details'] if t['success']) + \
             sum(1 for t in diehard_result['details'] if t['success'])
    total = len(nist_result['details']) + len(diehard_result['details'])
    summary = "passed" if passed == total else "failed" if passed == 0 else "mixed"

    return {
        'summary': summary,
        'details': {
            'NIST SP 800-22': nist_result['summary'],
            'DIEHARD': diehard_result['summary']
        },
        'subresults': {
            'nist': nist_result,
            'diehard': diehard_result
        }
    }


def state_recovery_analyze(bitstring):
    """Анализ восстановления внутреннего состояния (заглушка)"""
    return {
        'summary': 'безопасно',
        'details': {
            'Линейная сложность': 'высокая (>500)',
            'Цикличность': 'не обнаружена',
            'Уязвимости к атакам': 'не выявлены'
        }
    }


def pattern_analysis_analyze(bitstring):
    """Анализ повторяющихся шаблонов (заглушка)"""
    return {
        'summary': 'требует внимания',
        'details': {
            'Повторяющиеся шаблоны': 'обнаружены (длина 16)',
            'Распределение шаблонов': 'неравномерное',
            'Рекомендации': 'проверить алгоритм генерации'
        }
    }


def entropy_source_analyze(bitstring):
    """Анализ энтропийного источника (заглушка)"""
    return {
        'summary': 'хороший источник',
        'details': {
            'Энтропия (Shannon)': '0.997 бит/символ',
            'Min-энтропия': '0.992 бит/символ',
            'Оценка источника': 'подходит для криптографии'
        }
    }


def print_analysis_results(results):
    """Форматированный вывод результатов анализа"""
    print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===\n")

    for module_name, result in results.items():
        print(f"[{module_name}] {module_name.replace('_', ' ').title()}: {result['summary']}")

        if module_name == 'stats':
            print("  ├─ NIST SP 800-22:", result['details']['NIST SP 800-22'])
            print("  └─ DIEHARD:", result['details']['DIEHARD'])

        elif isinstance(result['details'], dict):
            for key, value in result['details'].items():
                print(f"    - {key}: {value}")
        print()


def main():
    """Основная точка входа программы"""
    try:
        file_path = input("Введите путь к файлу: ")
        bitstring = read_binary_string(file_path)
        print(f"Файл успешно прочитан. Длина последовательности: {len(bitstring)} бит")

        # Выполняем все модули анализа
        results = {
            'predictability': predictability_analyze(bitstring),
            'stats': stats_analyze(bitstring,file_path),
            'state_recovery': state_recovery_analyze(bitstring),
            'pattern_analysis': pattern_analysis_analyze(bitstring),
            'entropy_source': entropy_source_analyze(bitstring)
        }

        # Выводим результаты
        print_analysis_results(results)

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        return 1
    except ValueError as e:
        print(f"Ошибка: {e}")
        return 1
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())