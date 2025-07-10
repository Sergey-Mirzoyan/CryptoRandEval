# modules/stats/diehard_analyzer.py

import math
import numpy as np
from scipy.special import erfc
import tempfile
import os

def _birthday_spacings_test(bitstring):
    """Тест Birthday Spacings"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "birthday_spacings_test", "success": False, "error": "Invalid bitstring"}
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as f:
        data = f.read().strip()

    n = len(data)
    k = 10000  # количество пар последовательных байтов для проверки
    m = n // 2  # максимальное количество пар последовательных байтов
    t = min(m, k)
    matches = 0
    
    for i in range(t):
        if data[i] == data[i + 1]:
            continue
        for j in range(i + 1, t):
            if data[j] == data[j + 1]:
                continue
            if data[i] == data[j] and data[i + 1] == data[j + 1]:
                matches += 1

    p = (matches * 2) / ((t - 1) * t) if t > 1 else 1.0
    success = p >= 0.01
    return {
        "test_name": "birthday_spacings_test",
        "p_value": p,
        "success": success
    }

def _overlapping_permutations_test(bitstring):
    """Тест Overlapping Permutations"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "overlapping_permutations_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as f:
        data = f.read().strip()

    n = len(data)
    k = 8  # размер блока
    alpha = [0] * 256  # массив частот байтов

    for i in range(n - k + 1):
        block = data[i:i+k]
        freq = [0] * 256
        for b in block:
            freq[int(b)] += 1
        index = i % k
        if index == 0:
            alpha = [0] * 256
        for j in range(256):
            alpha[j] += freq[j]
        if index == k - 1:
            total = sum(alpha)
            expected = (k*(k-1)//2)*(n-k+1)/(256**k)
            if abs(total - expected) > 2*math.sqrt(expected):
                return {
                    "test_name": "overlapping_permutations_test",
                    "p_value": None,
                    "success": False
                }
    return {
        "test_name": "overlapping_permutations_test",
        "p_value": None,
        "success": True
    }

def _binary_rank_test(bitstring):
    """Бинарный ранговый тест"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "binary_rank_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read()

    if not binary_sequence:
        return {"test_name": "binary_rank_test", "success": False, "error": "Empty input"}

    rank = np.zeros(len(binary_sequence))
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i - 1]:
            if binary_sequence[i] == "1":
                rank[i] = rank[i - 1] + 1
            else:
                rank[i] = rank[i - 1] - 1
        else:
            rank[i] = rank[i - 1]

    s_obs = np.max(np.abs(rank))

    try:
        p_value = np.exp(-0.5 * s_obs ** 2)
    except:
        p_value = 0.0

    success = p_value >= 0.01
    return {
        "test_name": "binary_rank_test",
        "p_value": p_value,
        "success": success
    }

def _monkey_tests(bitstring):
    """Monkey Tests"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "monkey_tests", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read()

    if not binary_sequence:
        return {"test_name": "monkey_tests", "success": False, "error": "Empty input"}

    result = [False] * 1000
    for i in range(1000):
        num = random.randint(1, len(binary_sequence))
        text = binary_sequence[:num]
        left = 0
        right = len(text) - 1
        while left <= right:
            middle = (left + right) // 2
            if text[middle] == "0":
                left = middle + 1
            else:
                right = middle - 1
        if right == len(text) - 1 or left == 0:
            result[i] = True

    count = sum(result)
    success = count >= 980
    return {
        "test_name": "monkey_tests",
        "success": success
    }

def _count_the_1s_stream_test(bitstring):
    """Count The 1's Stream Test"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "count_the_1s_stream_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read()

    ones_count = 0
    count_sum = 0
    n = len(binary_sequence)

    for i in range(n):
        ones_count += int(binary_sequence[i])
        count_sum += ones_count / (i + 1)

    s_obs = abs((count_sum / n) - 0.5) * np.sqrt(n)
    try:
        p_value = erfc(s_obs / np.sqrt(2))
    except:
        p_value = 0.0

    success = p_value >= 0.01
    return {
        "test_name": "count_the_1s_stream_test",
        "p_value": p_value,
        "success": success
    }

def _parking_lot_test(bitstring):
    """Parking Lot Test"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "parking_lot_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read()

    n = len(binary_sequence)
    parking_lot = [False] * n

    for i in range(n):
        if binary_sequence[i] == "1":
            parked = False
            j = i
            while not parked and j < n:
                if not parking_lot[j]:
                    parking_lot[j] = True
                    parked = True
                else:
                    j += 1

    ones_count = sum(int(c) for c in binary_sequence)
    s_obs = 0
    for i in range(n):
        if binary_sequence[i] == "1" and parking_lot[i]:
            s_obs += 1

    s_obs = 2 * s_obs - ones_count
    try:
        p_value = erfc(abs(s_obs) / np.sqrt(2 * ones_count * (n - ones_count)))
    except:
        p_value = 0.0

    success = p_value >= 0.01
    return {
        "test_name": "parking_lot_test",
        "p_value": p_value,
        "success": success
    }

def _minimum_distance_test(bitstring):
    """Minimum Distance Test"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "minimum_distance_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read().strip()

    sequence_length = len(binary_sequence)
    if sequence_length < 3:
        return {"test_name": "minimum_distance_test", "success": False}

    for i in range(sequence_length - 2):
        for j in range(i + 1, sequence_length - 1):
            distance = 0
            if binary_sequence[i] != binary_sequence[j]:
                distance = 1
            for k in range(j + 1, sequence_length):
                if binary_sequence[i] != binary_sequence[k] and binary_sequence[j] != binary_sequence[k]:
                    distance += 1
            if distance < 3:
                return {
                    "test_name": "minimum_distance_test",
                    "success": False
                }

    return {
        "test_name": "minimum_distance_test",
        "success": True
    }

def _random_spheres_test(bitstring):
    """Random Spheres Test"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "random_spheres_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read().strip()

    sequence_length = len(binary_sequence)
    if sequence_length % 3 != 0:
        return {"test_name": "random_spheres_test", "success": False}

    num_triplets = sequence_length // 3
    for i in range(num_triplets):
        a = int(binary_sequence[3*i]) * 2 - 1
        b = int(binary_sequence[3*i+1]) * 2 - 1
        c = int(binary_sequence[3*i+2]) * 2 - 1
        if a*a + b*b + c*c > 1:
            return {"test_name": "random_spheres_test", "success": False}

    return {"test_name": "random_spheres_test", "success": True}

def _squeeze_test(bitstring):
    """Squeeze Test"""
    if not all(c in '01' for c in bitstring):
        return {"test_name": "squeeze_test", "success": False, "error": "Invalid bitstring"}

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
        tmpfile.write(bitstring)
        tmpfilename = tmpfile.name

    with open(tmpfilename, "r") as file:
        binary_sequence = file.read().strip()

    sequence_length = len(binary_sequence)
    block_size = 10000
    num_blocks = sequence_length // block_size

    for i in range(num_blocks):
        block = binary_sequence[i*block_size:(i+1)*block_size]
        num_zeros = block.count('0')
        proportion_zeros = num_zeros / block_size
        if not (0.45 <= proportion_zeros <= 0.55):
            return {"test_name": "squeeze_test", "success": False}

    final_block = binary_sequence[num_blocks*block_size:]
    if len(final_block) > 0:
        num_zeros = final_block.count('0')
        proportion_zeros = num_zeros / len(final_block)
        if not (0.45 <= proportion_zeros <= 0.55):
            return {"test_name": "squeeze_test", "success": False}

    return {"test_name": "squeeze_test", "success": True}

def run_diehard_tests(bitstring):
    """
    Запускает все тесты DIEHARD

    Args:
        bitstring (str): строка из '0' и '1'

    Returns:
        dict: {
            "summary": "passed / failed / mixed",
            "details": [
                {"test_name": str, "success": bool, "p_value": float}
            ]
        }
    """
    if not all(c in '01' for c in bitstring):
        raise ValueError("Bitstring содержит недопустимые символы")

    test_functions = [
        _birthday_spacings_test,
        _overlapping_permutations_test,
        _binary_rank_test,
        _monkey_tests,
        _count_the_1s_stream_test,
        _parking_lot_test,
        _minimum_distance_test,
        _random_spheres_test,
        _squeeze_test
    ]

    results = []
    passed = 0

    for test_func in test_functions:
        try:
            result = test_func(bitstring)
            results.append(result)
            if result.get("success", False):
                passed += 1
        except Exception as e:
            results.append({
                "test_name": test_func.__name__,
                "success": False,
                "error": str(e)
            })

    summary = "passed" if passed == len(results) else "failed" if passed == 0 else "mixed"

    return {
        "summary": summary,
        "details": results
    }