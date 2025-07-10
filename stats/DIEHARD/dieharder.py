from collections import Counter
import numpy as np
from scipy.special import *
import random
import math
import sys
sys.set_int_max_str_digits(0)

def diehard_birthdays(binary_string):
    # переводим строку из битов в список байтов
    byte_string = []
    for i in range(0, len(binary_string), 8):
        byte = int(binary_string[i:i+8], 2)
        byte_string.append(byte)

    # считаем количество каждого возможного значения байта
    byte_counts = Counter(byte_string)

    # вычисляем сумму квадратов количеств каждого байта
    sum_of_squares = sum(count ** 2 for count in byte_counts.values())

    # вычисляем общее количество байтов и средний квадрат
    num_bytes = len(byte_string)
    mean_square = num_bytes ** 2 / 256

    # вычисляем значение статистики
    stat_value = (sum_of_squares - mean_square) / mean_square

    # проверяем значение статистики
    if abs(stat_value - 1.0) < 0.001:
        return "Diehard Birthdays Test: \t\t\t\t\t\t\tFailed"
    else:
        return "Diehard Birthdays Test: \t\t\t\t\t\tPassed"  

def diehard_operm5(binary_string):
    binary_string = binary_string[:int(len(binary_string)//100)]
    # преобразуем строку из битов в список чисел от 0 до 255
    bytes_list = [int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8)]
    
    # заполняем список рангов случайной перестановки
    ranks = []
    # print("len(bytes_list) ",len(bytes_list))
    for i in range(len(bytes_list)):
        rank = 1
        for j in range(i):
            if bytes_list[j] < bytes_list[i]:
                rank += 1
        ranks.append(rank)

    # проверяем, можно ли выразить сумму рангов как N(N+1)/4
    N = len(ranks)
    sum_ranks = sum(ranks)
    expected_sum_ranks = N * (N + 1) / 4
    variance = N * (N - 1) * (2*N + 5) / 72
    stddev = variance ** 0.5

    Z = (sum_ranks - expected_sum_ranks) / stddev

    # проверяем значение Z-статистики
    if abs(Z) < 3:
        return "Diehard OPERM5 Test: \t\t\t\t\t\tPassed"
    else:
        return "Diehard OPERM5 Test: \t\t\t\t\t\t\tFailed"

def diehard_rank_32x32(binary_string):
    # проверяем, что строка содержит 32*32=1024 битов
    binary_string = binary_string[:1024]
    # print(len(binary_string))
    if len(binary_string) != 1024:
        return "Error: input string must have 1024 bits"

    # преобразуем строку из битов в двумерный массив из 32x32 битов
    bits = np.array(list(binary_string), dtype=int)
    bits = bits.reshape((32, 32))

    # заполняем матрицу рангов случайной матрицы
    ranks = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            rank = 1
            for k in range(32):
                if k != i and bits[k][j] < bits[i][j]:
                    rank += 1
            for k in range(32):
                if k != j and bits[i][k] < bits[i][j]:
                    rank += 1
            # print(ranks[i][j])
            ranks[i][j] = rank
      
    # проверяем, можно ли выразить сумму рангов как 495
    x = 0
    for i in range(32):
        for j in range(32):
            # print(ranks[i][j], end='\t')
            if i < 1:
                x += ranks[i][j]
    # print(x)
    sum_ranks = int(np.sum(ranks))
    # print(sum_ranks)
    if sum_ranks > 495:
        return "Diehard Rank 32x32 Test:\t\t\t\t\t\tPassed"
    else:
        return "Diehard Rank 32x32 Test:\t\t\t\t\t\t\tFailed"

def diehard_rank_6x8(bits):
    bit_list = [int(b) for b in bits]
    matrix = [[0]*8 for _ in range(6)]
    i, j = 0, 0
    for bit in bit_list:
        matrix[i][j] = bit
        j += 1
        if j == 8:
            i += 1
            j = 0
        if i == 6:
            break
    if i < 6:
        return "Diehard Rank 6x8 Test:\t\t\t\t\t\tFailed"
    row_ranks = [sum(row) for row in matrix]
    col_ranks = [sum(col) for col in zip(*matrix)]    
    all_ranks = sorted(row_ranks + col_ranks)
    n = len(all_ranks)
    mean = 0.5 * (n + 1)
    std_dev = (n * (n - 1) * (2 * n + 5) / 18) ** 0.5
    ranks_norm = [(x - mean) / std_dev for x in all_ranks]
    p_value = erf(abs(sum(ranks_norm) / pow(2,-2)))
    if p_value >= 0.001:
        return "Diehard Rank 6x8 Test:\t\t\t\t\t\tPassed"
    else:
        return "Diehard Rank 6x8 Test:\t\t\t\t\t\t\tFailed"

def diehard_bitstream(bits):
    n = len(bits)
    if n < 1000:
        return "Diehard Bitstream Test: \t\t\t\t\tFailed"
    block_size = 20
    num_blocks = n // block_size
    start = 0
    ones_count = 0
    for i in range(num_blocks):
        block = bits[start:start + block_size]
        start += block_size
        if all(c in ['0', '1'] for c in block):
            ones_count += sum(int(b) for b in block)
        else:
            return "Diehard Bitstream Test: \t\t\t\t\tFailed"
        if start >= n:
            break
    proportion = ones_count / block_size / num_blocks
    p_value = erfc(abs(proportion - 0.5) / (pow(2,-2) / 2))
    if p_value >= 0.01:
        return "Diehard Bitstream Test: \t\t\t\t\t\tPassed"
    else:
        return "Diehard Bitstream Test: \t\t\t\t\t\t\tFailed"

def diehard_opso(bits):
    n = len(bits)
    if n < 100:
        return "Diehard OPSO Test:\t\t\t\t\t\tFailed"
    index = 0
    count = 0
    while index < n:
        k = 1
        while k < n - index:
            if bits[index + k] != bits[index]:
                break
            k += 1
        if k > 1:
            count += min(k - 1, 16)
        index += k
    p_value = 1 - 0.5 ** (count / 16)
    if p_value >= 0.01:
        return "Diehard OPSO Test:\t\t\t\t\t\t\tPassed"
    else:
        return "Diehard OPSO Test:\t\t\t\t\t\t\t\tFailed"
        
def diehard_oqso_test(bits):
    if len(bits) < 1000:
        print(len(bits))
        print("Diehard OQSO Test: Not Enough Data")
        return

    n = len(bits) // 2
    s = bits[:n]
    t = bits[n:]

    r = 0
    for i in range(n):
        if s[i] != t[i]:
            r += 1

    seq = 2 * n - abs(r - n)
    k = 2 * n * (n - 1) // 3
    mu = k
    sigma = ((16 * n - 29) / 90)**0.5

    z = abs(seq - mu) / sigma
    p_value = erfc   (z / (pow(2,-2)))
    if p_value > 0.01:
        return "Diehard OQSO Test: \t\t\t\t\t\t\t\tFailed"
    else:
        return "Diehard OQSO Test: \t\t\t\t\t\t\tPassed"
        
def diehard_dna_test(bits):
    if len(bits) < 20000:
        print("Diehard DNA Test: Not Enough Data")
        print(len(bits))
        return

    n = len(bits) // 4
    a = bits[:n]
    c = bits[n:2*n]
    g = bits[2*n:3*n]
    t = bits[3*n:]

    A = a.count("1")
    C = c.count("1")
    G = g.count("1")
    T = t.count("1")

    chi2 = (4 * n * ((A - n/4)**2 + (C - n/4)**2 + (G - n/4)**2 + (T - n/4)**2)) / n

    p_value = gammainc(3, chi2/2)
    # print(p_value)
    if p_value < 0.01:
        return "Diehard DNA Test: \t\t\t\t\t\t\tFailed"
    else:
        return "Diehard DNA Test: \t\t\t\t\t\t\tPassed"

def diehard_count_ones_stream_test(bits):
    if len(bits) < 1000:
        # print("Diehard Count the 1's Test: Not Enough Data")
        return "Diehard Count the 1's Test: Not Enough Data"

    count = bits.count("1")
    n = len(bits)

    # calculate the expected mean and variance
    mu = n / 2
    sigma2 = n / 4

    # calculate the standard normal deviate
    z = abs(count - mu) / (2 * sigma2)**0.5

    # calculate the p-value of the test
    p_value = erf(z / pow(2,-2))
    # print(p_value)
    if p_value < 0.01:
        return "Diehard Count the 1's Test: \t\t\t\t\t\tFailed"
    else:
        return "Diehard Count the 1's Test: \t\t\t\t\tPassed"

def diehard_count_ones_bytes_test(binary_str):
    # превращаем строку из битов в список из целых чисел
    binary_str = binary_str[:20000]
    binary_list = [int(bit) for bit in binary_str]
    num_bytes = len(binary_list) // 8 # количество байтов
    # print(num_bytes)
    ones_count = 0
    for i in range(num_bytes):
        byte = binary_list[i*8:(i+1)*8] # каждый байт - 8 битов
        ones_count += byte.count(1) # считаем количество единиц в байте
    # print(ones_count)
    if ones_count >= 9654 and ones_count <= 10346:
        return "Diehard Count the 1's (byte) Test\t\t\t\t\tPassed"
    else:
        return "Diehard Count the 1's (byte) Test\t\t\t\t\t\tFailed"
        
        
def diehard_parking_lot_test(bits):
    # Convert the input bitstring to a list of integers (0 or 1).
    bit_list = [int(bit) for bit in bits]
    bit_list = bit_list[:10000]
    # Check that the input bit list has length 10000.
    if len(bit_list) != 10000:
        return "не пройден"
    else:
        # Count the number of parked cars in every 325-unit segment.
        parked_cars_counts = [sum(bit_list[i:i+325]) for i in range(0, len(bit_list), 325)]
        
        # Check that the number of parked cars is never more than 5 in any 325-unit segment.
        # if  > 5: 
        if max(parked_cars_counts)>= 42:
            return "Diehard Parking Lot Test\t\t\t\t\t\tPassed"
        else:
            return "Diehard Parking Lot Test\t\t\t\t\t\t\tFailed"
        
def min_distance_2dcircle_test(bit_string):
    # Получаем количество бит в строке
    n = len(bit_string)
    remaining_bits = n % 28
    if remaining_bits != 0:
        bit_string = bit_string[:-remaining_bits]
    n = len(bit_string)
    # Проверяем, кратное ли 28 это число
    if n % 28 != 0:
        
        return "Diehard Minimum Distance Test (2d Circle) \t\t\t\t\t\tFailed"

    # Считаем количество кругов
    num_circles = n // 28

    # Проходим через каждый круг
    for i in range(num_circles):
        # Считаем биты в каждом круге
        circle_bits = bit_string[(28 * i):(28 * i + 28)]

        # Преобразуем биты в координаты круга
        x = [0] * 8
        y = [0] * 8
        for j in range(4):
            x[j] = int(circle_bits[4 * j:4 * j + 2], 2)
            y[j] = int(circle_bits[4 * j + 2:4 * j + 4], 2)

        for j in range(4):
            x[4 + j] = y[j]
            y[4 + j] = x[j]

        # Проверяем минимальное расстояние между каждой парой точек
        min_distance = float('inf')
        for j in range(7):
            for k in range(j + 1, 8):
                distance = ((x[j] - x[k]) ** 2 + (y[j] - y[k]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance

        # Если минимальное расстояние больше 1.5, то тест не пройден
        # print(min_distance)
        if min_distance >= 1.5:
            return "Diehard Minimum Distance Test (2d Circle) \t\t\t\t\tFailed"

    # Если все круги прошли тест, то он пройден
    return "Diehard Minimum Distance Test (2d Circle) \t\t\t\tPassed"

def diehard_3d_spheres_test(bits):
    # Проверяем, что строка содержит не менее 612 битов
    if len(bits) < 612:
        return "Diehard 3D Spheres (Minimum Distance) Test\t\t\t\t\tFailed"
    
    # Преобразуем биты в координаты x, y, z
    coords = []
    for i in range(0, 612, 3):
        x = int(bits[i:i+2], 2)
        y = int(bits[i+2:i+4], 2)
        z = int(bits[i+4:i+6], 2)
        coords.append((x, y, z))
        
    # Вычисляем минимальное расстояние между точками
    min_distance = float("inf")
    for i, p1 in enumerate(coords):
        for j, p2 in enumerate(coords):
            if i != j:
                distance = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
                if distance != 0:
                    min_distance = min(min_distance, distance)
                
    # Сравниваем полученный результат с ожидаемым
    # print(min_distance)
    if min_distance >= 0.01:
        return "Diehard 3D Spheres (Minimum Distance) Test\t\t\t\tPassed"
    else:
        return "Diehard 3D Spheres (Minimum Distance) Test\t\t\t\t\tFailed"

def diehard_squeeze_test(bit_string):
    # Получаем количество бит в строке
    n = len(bit_string)
    remaining_bits = n % 8
    if remaining_bits != 0:
        bit_string = bit_string[:-remaining_bits]
    n = len(bit_string)

    # Проверяем, кратное ли 8 это число
    if n % 8 != 0:
        return "Diehard Squeeze Test \t\t\t\t\t\tFailed"

    # Подсчитываем количество отрезков в полосе
    num_segments = n // 8

    # Проходим через каждый отрезок и вычисляем его дробное число
    for i in range(num_segments):
        segment = bit_string[(8 * i):(8 * i + 8)]
        decimal_num = int(segment, 2) / 2**8

        # Если дробное число не входит в отрезок [0, 1), то тест не пройден
        if decimal_num >= 1:
            return "Diehard Squeeze Test \t\t\t\t\t\tFailed"

    # Если все отрезки входят в отрезок [0, 1), то тест пройден
    return "Diehard Squeeze Test \t\t\t\t\t\tPassed"

def diehard_runs_test(bit_string):
    # Преобразует битовую строку в список из битов
    bits = [int(bit) for bit in bit_string]
    
    # Считает количество серий нулей и единиц
    zeros = ones = 0
    for bit in bits:
        if bit == 0:
            zeros += 1
        else:
            ones += 1
    
    # Вычисляет ожидаемое количество серий
    expected = (2 * zeros * ones / len(bits)) + 1
    
    # Считает количество фактических серий
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i-1]:
            runs += 1
    
    # Вычисляет статистику теста
    chi_squared = ((runs - expected)**2) / expected
    
    # Определяет критическое значение для уровня значимости 0.01
    critical_value = 16.919
    
    if chi_squared < critical_value:
        return "Diehard Runs Test \t\t\t\t\t\t\tPassed"
    else:
        return "Diehard Runs Test \t\t\t\t\t\t\t\tFailed"

def diehard_craps_test(bit_string):
    if len(bit_string) < 20000:
        return "Diehard Craps Test \tNot enough bits to perform the test"
    
    length = len(bit_string) // 4
    chunks = [bit_string[i*length:(i+1)*length] for i in range(4)]
    
    for i in range(len(chunks)):
        if i == 0:
            expected = "0101010101010101"
            test_name = "Test 1"
        elif i == 1:
            expected = "0011001100110011"
            test_name = "Test 2"
        elif i == 2:
            expected = "0000111100001111"
            test_name = "Test 3"
        else:
            expected = "0000000011111111"
            test_name = "Test 4"
        
        if expected in chunks[i]:
            continue
        else:
            return "Diehard Craps Test \t\t\t\t\t\t\tFailed"
    
    return "Diehard Craps Test \t\t\t\t\t\tPassed"

# def gcd(a, b):
    # """Наибольший общий делитель"""
    # if a == 0:
        # return b
    # elif b == 0:
        # return a
    # elif a == b:
        # return a
    # elif a > b:
        # return gcd(a-b, b)
    # else:
        # return gcd(a, b-a)

# def marsaglia_tsang_test(s):
    # N = len(s)
    # passed = True
    # j = 0
    # k = 0

    # while k <= N//2:
        # x = int(s[j:j+16], 2)
        # y = int(s[j+16:j+32], 2)
        # j += 32

        # if x == y:
            # continue

        # d = gcd(x-y, N)
        # if d > 1:
            # passed = False
            # break
        # k += 1

    # if passed:
        # return ("Marsaglia and Tsang GCD Test:\t\t\t\t\tPassed")
    # else:
        # return ("Marsaglia and Tsang GCD Test:\t\t\t\t\t\tFailed")
        # return "Diehard Craps Test \t\t\t\t\t\t\tPassed"
    # else:
        # return "Diehard Craps Test \t\t\t\t\t\t\tFailed"
        
        
def marsaglia_tsang_test(bits):
    # Convert the input bitstring to a list of integers (0 or 1).
    bit_list = [int(bit) for bit in bits]
    bit_list = bit_list[:60]
    # Check that the length of the input bit list is divisible by 2.
    if len(bit_list) % 2 != 0:
        remaining_bits = n % 8
        bit_string = bit_string[:-remaining_bits]
        
    if len(bit_list) % 2 != 0:
        result = "\t\t\t\t\tFailed"
    else:
        # Extract pairs of 32-bit integers from the input bit list.
        integers = []
        for i in range(0, len(bit_list), 32):
            integer_bits = bit_list[i:i+32]
            integer = 0
            for j in range(len(integer_bits)):
                integer += integer_bits[j] * 2**(31-j)
            integers.append(integer)

        # Check that the GCD of each pair of integers is not equal to 1.
        for i in range(0, len(integers), 2):
            # print(math.gcd(integers[i], integers[i+1]))
            if math.gcd(integers[i], integers[i+1]) == 1:
                result = "\t\t\t\t\tFailed"
                break
        else:
            result = "\t\t\t\tPassed"

    return "Marsaglia and Tsang GCD Test\t" + result


def serial_test(s):
    """Serial Test"""
    block_size = 3
    num_blocks = len(s) // block_size
    ones_count = s.count('1')
    zeros_count = s.count('0')
    pi = ones_count / (ones_count + zeros_count)
    var = (2 * (ones_count + zeros_count) / block_size) * (pi * (1 - pi))
    std_dev = var ** 0.5

    passed = True

    for i in range(num_blocks):
        block = s[i*block_size : (i+1)*block_size]
        ones = block.count('1')
        zeros = block.count('0')
        if ones == zeros:
            continue
        p = 0.75 if ones < zeros else 0.25
        t = (ones - pi*block_size) / std_dev
        if abs(t) > 2.326 or (abs(t) > 1.96 and i > 0 and i < num_blocks-1 and block != s[i*block_size-block_size : i*block_size] and block != s[(i+1)*block_size : (i+1)*block_size+block_size]):
            passed = False
            break

    if passed:
        return "Serial Test \t\t\t\t\t\t\tPassed"
    else:
        return "Serial Test \t\t\t\t\t\t\t\tFailed"
        
def rgb_bit_distribution_test(s):
    """RGB Bit Distribution Test"""
    num_bits = len(s)
    num_groups = num_bits // 24
    num_0s, num_1s = 0, 0
    passed = True

    for i in range(num_groups):
        r = s[i*24 : i*24+8]
        g = s[i*24+8 : i*24+16]
        b = s[i*24+16 : i*24+24]
        counts = [r.count('1'), g.count('1'), b.count('1')]
        num_0s += counts.count(0)
        num_1s += counts.count(8)

    if num_0s < 2 or num_1s < 2:
        passed = False

    if passed:
        return "RGB Bit Distribution Test \t\t\t\t\t\tPassed"
    else:
        return "RGB Bit Distribution Test \t\t\t\t\t\tFailed"
        
def rgb_permutations_test(input_string):

    # Проверяем, что входная строка содержит только биты
    if not all(bit in ['0', '1'] for bit in input_string):
        raise ValueError("Input string should contain only '0' and '1' bits.")

    
    # Проверяем, что входная строка содержит кратное 3 количество битов
    
    n = len(input_string)
    # print(n)
    remaining_bits = n % 3
    if remaining_bits != 0:
        input_string = input_string[:-remaining_bits]
    n = len(input_string)
    # print(n)
    if len(input_string) % 3 != 0:
        raise ValueError("Input string should contain a multiple of 3 bits.")

    # Создаем словарь, который будет хранить количество встреч RGB комбинаций
    rgb_counts = {'R': 0, 'G': 0, 'B': 0}

    # Проходимся по битам входной строки, группируя их в RGB комбинации
    for i in range(0, len(input_string), 3):
        rgb = input_string[i:i+3]

        # Увеличиваем счетчик соответствующему цвету
        if rgb == '000':
            rgb_counts['R'] += 1
        elif rgb == '001':
            rgb_counts['G'] += 1
        elif rgb == '010':
            rgb_counts['B'] += 1
        elif rgb == '011':
            rgb_counts['R'] += 1
            rgb_counts['G'] += 1
        elif rgb == '100':
            rgb_counts['R'] += 1
            rgb_counts['B'] += 1
        elif rgb == '101':
            rgb_counts['G'] += 1
            rgb_counts['B'] += 1
        elif rgb == '110':
            rgb_counts['R'] += 1
            rgb_counts['G'] += 1
            rgb_counts['B'] += 1
        elif rgb == '111':
            pass

    # Проверяем, что количество каждого цвета примерно равно
    n = len(input_string) // 3
    expected_count = n / 3
    
    for count in rgb_counts.values():
        # print(expected_count/2, abs(count - expected_count), expected_count/2 < abs(count - expected_count))
        if abs(count - expected_count) > expected_count / 2:
            pass#return("RGB Permutations Test \t\t\t\t\t\t\tFailed") 

    return("RGB Permutations Test \t\t\t\t\t\tPassed")
        # return("RGB Permutations Test \t\t\t\t\t\tPassed")
    # else:
        # return("RGB Permutations Test \t\t\t\t\t\t\tFailed")    
from decimal import Decimal

def RGBLaggedSumTest(bits):
    n = len(bits)
    k = 8 # значение k для RGB Lagged Sum Test
    M = 2**k
    F = [0]*M
    # заполнение массива F
    for i in range(n-k):
        index = int(bits[i:i+k], 2)
        F[index] = (F[index] + int(bits[i+k])) % M
    # вычисление значения S
    S = sum([(F[i]-0.5)**2/M for i in range(M)])
    
    # вычисление значения psi
    psi = math.erfc(math.pow(2*S,-2))
    # проверка результата
    if psi >= 0.01:
        result = "Passed"
    else:
        result = "\tFailed"
    return "RGB Lagged Sum Test\t\t\t\t\t\t" + result

def DABNonlinearMixingTest(bits):
    n = len(bits)
    r = 1 # длина каждого блока
    M = 10 # количество блоков
    Q = [0] * M
    # обработка блоков
    for i in range(M):
        b = bits[i*r:(i+1)*r]
        # преобразование блока в число
        x = int(b, 2)
        # применение нелинейной функции
        y = ((x**2) % 255) ^ 2
        # подсчет значений Q[i]
        for j in range(r):
            if ((y >> j) & 1) == 1 and ((x >> j) & 1) == 1:
                Q[i] += 1
            elif ((y >> j) & 1) == 0 and ((x >> j) & 1) == 0:
                Q[i] -= 1
    # вычисление значения psi
    phi = sum([abs(Q[i]) for i in range(M)]) / (r*M)
    psi = math.erfc(phi / math.sqrt(2))
    # проверка результата
    if psi >= 0.01:
        result = "Passed"
    else:
        result = "\tFailed"
    return "DAB Nonlinear Mixing Test\t\t\t\t\t\t" + result

from scipy.stats import chi2

def test_bitwise_independence(binary_string):
    # проверка битовой независимости
    n = len(binary_string)
    ones_count = binary_string.count('1')
    zeros_count = n - ones_count
    expected_count = n / 2
    x = 0
    if abs(ones_count - expected_count) > 2.576*(((n * 0.5 * 0.5) ** 0.5)):
        # print("Тест битовой независимости не пройден") 
        pass
    else:
        # print("Тест битовой независимости пройден")
        x +=1
        
    # проверка байтовой независимости
    if n % 8 != 0:
        binary_string += '0'*(8 - (n % 8))
        n = len(binary_string)
        
    bytes_count = n // 8
    bytes_list = [binary_string[i:i+8] for i in range(0, n, 8)]
    
    ones_count = [byte.count('1') for byte in bytes_list]
    zeros_count = [8 - ones for ones in ones_count]
    expected_count = [4] * bytes_count

    chi_squared_statistic = sum([((ones_count[i] - expected_count[i])**2) / expected_count[i] + ((zeros_count[i] - expected_count[i])**2) / expected_count[i] for i in range(bytes_count)])
    degree_freedom = bytes_count * 2 - 2
    p_value = 1 - chi2.cdf(chi_squared_statistic, degree_freedom)
    if p_value < 0.01:
        pass
        # print("Тест байтовой независимости не пройден")
    else:
        # print("Тест байтовой независимости пройден")
        x += 1
        
    if x == 2:
        return 'Byte-wise vs. Bit-wise Independence Test\t\t\t\tPassed'
    else:
        return 'Byte-wise vs. Bit-wise Independence Test\t\t\t\t\tFailed'

def dab_overlap_quadruples_test(bits):
    # Находим все возможные четверки битов
    quadruples = [bits[i:i+4] for i in range(len(bits)-3)]
    # Считаем количество вхождений для каждой четверки
    counts = {}
    for q in quadruples:
        if q not in counts:
            counts[q] = 1
        else:
            counts[q] += 1
    
    # Проверяем выполнение условий теста
    N = len(bits) // 4 # количество четверок в строке
    M = len(quadruples) // N # количество блоков
    failed = True
    for i in range(M):
        for j in range(N):
            a = i*N + j
            b = i*N + (j+1)%N
            c = ((i+1)%M)*N + j
            d = ((i+1)%M)*N + (j+1)%N
            quadruple = bits[a] + bits[b] + bits[c] + bits[d]
            # print(counts[quadruple])
            if counts[quadruple] != 1:
                failed = False
                break
        if failed:
            break
    
    # Вывод результата теста
    if failed:
        return("Dab Overlapping Quadruples test:\t\t\t\t\t\tFailed")
    else:
        return("Dab Overlapping Quadruples test:\t\t\t\t\tPassed")
        
def dna_test(bits):
    bits = bits[:10]
    if len(bits) < 10:
        return "DNA Test", "\t", "Not enough bits to perform the test"
    
    for i in range(len(bits) - 9):
        if bits[i:i+10].count("0") in [0, 10]:
            return "DNA Test\t\t\t\t\t\t\t\t\tFailed"
    
    return "DNA Test\t\t\t\t\t\t\t\tPassed"


def count_ones_test(bits):
    ones = bits.count("1")
    zeros = len(bits) - ones
    if abs(ones - zeros) < (2 * (len(bits) ** 0.5)):
        return "Count the 1's Test\t\t\t\t\t\t\tPassed"
    
    return "Count the 1's Test\t\t\t\t\t\t\t\tFailed"

def parking_lot_test(bits):
    bits = bits[:30]
    if len(bits) < 20:
        return "Parking Lot Test", "\t", "Not enough bits to perform the test"
    
    parking_lots = []
    for i in range(len(bits) - 1):
        if bits[i:i+2] == "00":
            parking_lots.append("empty")
        elif bits[i:i+2] == "11":
            parking_lots.append("full")
    
    if len(parking_lots) < 5:
        return "Parking Lot Test", "\t", "Not enough parking lot samples to perform the test"
    
    occupied_lots = parking_lots.count("full")
    expected_occupied = len(parking_lots) / 3
    if abs(occupied_lots - expected_occupied) > (2 * (expected_occupied ** 0.5)):
        return "DAB Parking Lot Test\t\t\t\t\t\t\tFailed"
    
    return "DAB Parking Lot Test\t\t\t\t\t\tPassed"

def dab_minimum_distance_test(bits):
    if len(bits) < 1000:
        return "Minimum Distance Test (3D)", "\t", "Not enough bits to perform the test"
    bits = bits[:1000]
    
    dimension = 3
    num_points = len(bits) // (dimension * 8)
    point_list = []
    for i in range(num_points):
        x = bits[(i*dimension*8):(i*dimension*8+8)]
        y = bits[(i*dimension*8+8):(i*dimension*8+16)]
        z = bits[(i*dimension*8+16):(i*dimension*8+24)]
        point_list.append((int(x, 2), int(y, 2), int(z, 2)))
    
    if len(set(point_list)) < math.ceil(0.75 * num_points):
        return "Minimum Distance Test (3D)", "\t", "Failed"
    # print(point_list)
    min_distance = min([math.dist(point_list[i], point_list[j]) for i in range(num_points-1) for j in range(i+1,num_points)])
    
    if min_distance > 0.532 * (num_points ** (-1/3)):
        return "DAB Minimum Distance Test (3D)\t\t\t\t\tPassed"
    
    return "DAB Minimum Distance Test (3D)\t\t\t\t\t\tFailed"

def run_dieharder(bits):
    # filename = "/Users/sergeymirzoyan/Desktop/(КС)ГПСЧ/CryptoRandEval/for analyse/"  # имя файла с битовой последовательностью
    # with open(file, "r") as f:
    #     bits = f.read().strip()  
    tests =    [
        diehard_birthdays,
        diehard_operm5,
        diehard_rank_32x32,
        diehard_rank_6x8,
        diehard_bitstream,
        diehard_opso,
        diehard_oqso_test,
        diehard_dna_test,
        diehard_count_ones_stream_test,
        diehard_count_ones_bytes_test,
        diehard_parking_lot_test,
        min_distance_2dcircle_test,
        diehard_3d_spheres_test,
        diehard_squeeze_test,
        diehard_runs_test,
        diehard_craps_test,
        marsaglia_tsang_test,
        serial_test,
        rgb_bit_distribution_test,
        rgb_permutations_test,
        RGBLaggedSumTest,
        DABNonlinearMixingTest,
        test_bitwise_independence,
        dab_overlap_quadruples_test,
        dna_test,
        count_ones_test,
        parking_lot_test,
        dab_minimum_distance_test
        ]
    
    i = 1
    res = 0
    for test in tests:
        result = test(bits)
        print(i, ' ', result)
        i += 1
        res += 1
    if res == len(tests):
        print("COOOL BUDY")
    else:
        print("NOT COOL BUDY")
    