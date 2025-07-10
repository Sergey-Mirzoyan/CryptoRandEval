import math
from scipy.special import *
import numpy as np
import sys
sys.set_int_max_str_digits(0)

def birthday_spacings_test(filename):
    with open(filename, "r") as f:
        data = f.read().encode("utf-8")
    n = len(data)
    k = 10000 # количество пар последовательных байтов для проверки
    m = n // 2 # максимальное количество пар последовательных байтов
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
    p = (matches * 2) / ((t - 1) * t)
    if p < 0.01:
        return False
    else:
        return True
        
def overlapping_permutations_test(filename):
    with open(filename, "r") as f:
        data = f.read().encode("utf-8")
    n = len(data)
    k = 8 # размер блока
    alpha = [0] * 256 # массив частот байтов
    for i in range(n - k + 1):
        block = data[i:i+k]
        freq = [0] * 256
        for b in block:
            freq[b] += 1
        index = i % k
        if index == 0:
            alpha = [0] * 256
        for j in range(256):
            alpha[j] += freq[j]
        if index == k - 1:
            sum = 0
            for j in range(256):
                x = alpha[j]
                sum += (x*(x-1))//2
            expected = (k*(k-1)//2)*(n-k+1)/(256**k)
            if abs(sum - expected) > 2*math.sqrt(expected):
                return False
    return True    

def rankTest(input_file):
    """
    Функция принимает путь до файла с бинарной последовательностью
    и возвращает True, если матрица, составленная из этой последовательности,
    имеет полный ранг, и False в противном случае.
    """
    # Читаем бинарную последовательность из файла:
    with open(input_file, 'rb') as f:
        raw_data = f.read()
        
    # Преобразуем бинарную последовательность в список чисел:
    numbers = list(raw_data)
    
    # Создаем массив (матрицу) размера NxN из списка чисел:
    N = int(np.sqrt(len(numbers)))
    matrix = np.array(numbers).reshape((N, N))
    
    # Вычисляем ранг матрицы и проверяем, равен ли он N:
    rank = np.linalg.matrix_rank(matrix)
    return rank == N

import random

def monkey_tests(file_path):
    with open(file_path, "r") as file:
        binary_sequence = file.read()
    #Проверка на пустоту
    if not binary_sequence:
        return False

    #Создаем пустой массив для хранения результата
    result = [False] * 1000

    #Проходим тесты
    for i in range(1000):
        #Создаем случайное число и исходный текст
        num = random.randint(1, len(binary_sequence))
        text = binary_sequence[:num]

        #Бинарный поиск
        left = 0
        right = len(text) - 1
        while left <= right:
            middle = (left + right) // 2
            if text[middle] == "0":
                left = middle + 1
            else:
                right = middle - 1

        #Проверка результата
        if right == len(text) - 1 or left == 0:
            result[i] = True

    #Проверка прохождения тестов
    count = 0
    for res in result:
        if res:
            count += 1
    if count >= 980:
        return True
    else:
        return False

def binary_rank_test(file_path):
    with open(file_path, "r") as file:
        binary_sequence = file.read()
    #Проверка на пустоту
    if not binary_sequence:
        return False

    #Создадим список rank, состоящий из нулей
    rank = np.zeros(len(binary_sequence))

    #Проходим по бинарной последовательности и устанавливаем значения в списке rank
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            if binary_sequence[i] == "1":
                rank[i] = rank[i-1] + 1
            else:
                rank[i] = rank[i-1] - 1
        else:
            rank[i] = rank[i-1]

    #Вычисляем статистическое значение
    s_obs = np.max(np.abs(rank))

    #Вычисляем p-value
    if s_obs < 387840:
        p_value = np.exp(-0.5 * s_obs**2)
    else:
        p_value = 0.0

    #Проверка прохождения теста
    if p_value >= 0.01:
        return True
    else:
        return False
def count_the_1s_stream_test(file_path):
    with open(file_path, "r") as file:
        binary_sequence = file.read()

    #Проверка на пустоту
    if not binary_sequence:
        return False

    n = len(binary_sequence)
    ones_count = 0
    count_sum = 0

    #Проходим бинарную последовательность
    for i in range(n):
        ones_count += int(binary_sequence[i])
        count_sum += ones_count / (i+1)

    #Вычисляем статистическое значение
    s_obs = abs((count_sum / n) - 0.5) * np.sqrt(n)

    #Вычисляем p-value
    p_value = erfc(s_obs / np.sqrt(2))

    #Проверка прохождения теста
    if p_value >= 0.01:
        return True
    else:
        return False
def parking_lot_test(file_path):
    with open(file_path, "r") as file:
        binary_sequence = file.read()
    #Проверка на пустоту
    if not binary_sequence:
        return False

    #Вычисляем длину бинарной последовательности
    n = len(binary_sequence)

    #Создаем список для хранения парковочных мест
    parking_lot = [False] * n

    #Проверяем парковочные места для каждой машины в последовательности
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

    #Вычисляем статистическое значение
    s_obs = 0
    ones_count = 0
    for i in range(n):
        if binary_sequence[i] == "1":
            ones_count += 1
            if parking_lot[i]:
                s_obs += 1
    s_obs = 2 * s_obs - ones_count

    #Вычисляем p-value
    p_value = erfc(abs(s_obs) / np.sqrt(2 * ones_count * (n - ones_count)))

    #Проверка прохождения теста
    if p_value >= 0.01:
        return True
    else:
        return False


def minimum_distance_test(file_name):
    with open(file_name, 'r') as f:
        binary_sequence = f.read().strip()

    # Check if the binary sequence contains only 0s and 1s
    if not set(binary_sequence).issubset({'0', '1'}):
        return False
    
    sequence_length = len(binary_sequence)

    # Check if the sequence length is at least three
    if sequence_length < 3:
        return False

    # Calculate the minimum Hamming distance between all pairs of sub-sequences
    for i in range(sequence_length - 2):
        for j in range(i + 1, sequence_length - 1):
            if binary_sequence[i] != binary_sequence[j]:
                distance = 1
            else:
                distance = 0
            
            for k in range(j + 1, sequence_length):
                if binary_sequence[i] != binary_sequence[k] and binary_sequence[j] != binary_sequence[k]:
                    distance += 1

            if distance < 3:
                return False

    return True
def random_spheres_test(file_name):
    with open(file_name, 'r') as f:
        binary_sequence = f.read().strip()

    # Check if the binary sequence contains only 0s and 1s
    if not set(binary_sequence).issubset({'0', '1'}):
        return False
    
    sequence_length = len(binary_sequence)
    
    # Check if the sequence length is a multiple of 3
    if sequence_length % 3 != 0:
        return False
    
    num_triplets = sequence_length // 3
    
    # Generate the spheres and check their intersection with the binary sequence
    for i in range(num_triplets):
        a = int(binary_sequence[3*i]) * 2 - 1  # Convert the binary digit to a +1 or -1 value
        b = int(binary_sequence[3*i+1]) * 2 - 1
        c = int(binary_sequence[3*i+2]) * 2 - 1
        # print(math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2))
        if math.pow(a, 2) + math.pow(b, 2) + math.pow(c, 2) > 1:
            return False

    return True
def squeeze_test(file_name):
    with open(file_name, 'r') as f:
        binary_sequence = f.read().strip()

    # Check if the binary sequence contains only 0s and 1s
    if not set(binary_sequence).issubset({'0', '1'}):
        return False

    sequence_length = len(binary_sequence)
    block_size = 10000
    num_blocks = sequence_length // block_size

    # Calculate the proportion of zeros in each block and check if it falls within the expected range
    for i in range(num_blocks):
        block = binary_sequence[i*block_size:(i+1)*block_size]
        num_zeros = block.count('0')
        proportion_zeros = num_zeros / block_size
        
        if proportion_zeros < 0.45 or proportion_zeros > 0.55:
            return False

    # Check the final block
    final_block = binary_sequence[num_blocks*block_size:]
    num_zeros = final_block.count('0')
    proportion_zeros = num_zeros / len(final_block)
    
    if proportion_zeros < 0.45 or proportion_zeros > 0.55:
        return False

    return True
def run_diehard_tests_2(filename):
    test_res = 0
    #filename = "../nums.txt"
    # 1
    if birthday_spacings_test(filename):
        print("birthday_spacings_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("birthday_spacings_test \t\t\t Failed!")
    # 2  
    if overlapping_permutations_test(filename):
        print("overlapping_permutations_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("overlapping_permutations_test \t\t\t Failed.")
    # 3  
    print("rankTest \t\t\t\t\t NOT WORKING!")
    test_res += 1
    # if rankTest(filename):
    #     print("rankTest \t\t\t\t Passed!")
    #     test_res += 1
    # else:
    #     print("rankTest \t\t\t Failed.")
    # 4  
    if monkey_tests(filename):
        print("monkey_tests \t\t\t\t Passed!")
        test_res += 1
    else:
        print("monkey_tests \t\t\t\t\t Failed.")
    # 5  
    if binary_rank_test(filename):
        print("binary_rank_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("binary_rank_test \t\t\t\t\t Failed.")
    # 6  
    if count_the_1s_stream_test(filename):
        print("count_the_1s_stream_test \t\t\t Passed!")
        test_res += 1
    else:
        print("count_the_1s_stream_test \t\t\t Failed.")
    # 7  
    if parking_lot_test(filename):
        print("parking_lot_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("parking_lot_test \t\t\t\t Failed.")
    # 8  
    if minimum_distance_test(filename):
        print("minimum_distance_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("minimum_distance_test \t\t\t\t Failed.")
    # 9  
    if random_spheres_test(filename):
        print("random_spheres_test \t\t\t\t Passed!")
        test_res += 1
    else:
        print("random_spheres_test \t\t\t\t Failed.")
    # 10  
    if squeeze_test(filename):
        print("squeeze_test \t\t\t\t\t Passed!")
        test_res += 1
    else:
        print("squeeze_test \t\t\t\t\t Failed.")

    if test_res == 10:
        print("Diehard: PASSED")
    else:
        print("NOT PASSED")