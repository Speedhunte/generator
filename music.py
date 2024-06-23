import statistics
import numpy as np
from scipy.stats import norm, chi2
import time
import matplotlib.pyplot as plt
from nistrng import *
import random

seed = 42
a = 127007
b = 150287
M = 2**32

x = 123456789
y = 362436069
z = 521288629
w = 88675123
"""@brief генератор XOR-Shift"""
def gen_XOR():
    global x, y, z, w
    t = (x ^ (x << 11)) & 0xFFFFFFFF
    x, y, z = y, z, w
    w = ((w ^ (w >> 19)) ^ (t ^ (t >> 8))) & 0xFFFFFFFF
    return (w * x + y) % z
""""@brief конгруэнтный генератор"""
def gen_QCL():
    global seed
    c = 210193
    seed = (a * seed ** 2 + b * seed + c) % M
    seed = seed ^ (seed >> 5)
    return seed % M
"""@brief функция генерации выборок"""
def generate_samples(generator_func, num_samples, sample_size):
    samples = []
    for x in range(num_samples):
        sample = [generator_func() for y in range(sample_size)]
        samples.append(sample)
    return samples
"""@brief функция подсчета статистик"""
def calculate_statistics(samples):
    means = [statistics.mean(sample) for sample in samples]
    std_devs = [statistics.stdev(sample) for sample in samples]
    coefficients_of_variation = [std_dev / mean for mean, std_dev in zip(means, std_devs)]
    return means, std_devs, coefficients_of_variation
"""@brief функция вычисления хи-квадрат"""
def chi(sample):
    n = 1 + int(np.log2(len(sample)))
    observed, _ = np.histogram(sample, bins=n)
    expected = len(sample) / n
    stat = sum((observed - expected) ** 2 / expected)
    return stat

num_samples = 20
sample_size = 100

"""@brief Генератор XOR"""
xor_samples = generate_samples(gen_XOR, num_samples, sample_size)

"""@brief Вычисление статистик для выборок XOR"""
xor_means, xor_std_devs, xor_coefficients_of_variation = calculate_statistics(xor_samples)

for i, sample in enumerate(xor_samples):
    print(f"Выборка {i+1} (XOR): {sample}")
    print("Среднее:", xor_means[i])
    print("Стандартное отклонение:", xor_std_devs[i])
    print("Коэффициент вариации:", xor_coefficients_of_variation[i])

    chi_xor = chi(xor_samples[i])
    res_stat_xor = chi2.ppf(0.95, int(np.log2(len(xor_samples[i]))))

    if chi_xor <= res_stat_xor:
        print("Выборка равномерна")
    else:
        print("Выборка не равномерна")
    print()


"""@brief Генератор QCL"""
qcl_samples = generate_samples(gen_QCL, num_samples, sample_size)

"""@brief Вычисление статистик для выборок QCL"""
qcl_means, qcl_std_devs, qcl_coefficients_of_variation = calculate_statistics(qcl_samples)

for i, sample in enumerate(qcl_samples):
    print(f"Выборка {i+1} (QCL): {sample}")
    print("Среднее:", qcl_means[i])
    print("Стандартное отклонение:", qcl_std_devs[i])
    print(f"Коэффициент вариации:", qcl_coefficients_of_variation[i])

    chi_qcl = chi(qcl_samples[i])
    res_stat_qcl = chi2.ppf(0.95, int(np.log2(len(qcl_samples[i]))))

    if chi_qcl <= res_stat_qcl:
        print("Выборка равномерна")
    else:
        print("Выборка не равномерна")
    print()

print()

sequence: np.ndarray = np.array(generate_samples(gen_XOR, 1, 1000))
binary_sequence: np.ndarray = pack_sequence(sequence)
eligible_battery: dict = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
results = run_all_battery(binary_sequence, eligible_battery, False)
print("Результаты тестов для XOR:")
for result, elapsed_time in results[:7]:
    if result.passed:
        print(result.name + " - ПРОЙДЕН, score = " + str(np.round(result.score, 3)))
    else:
        print(result.name + " - НЕ ПРОЙДЕН, score = " + str(np.round(result.score, 3)))

print()

sequence: np.ndarray = np.array(generate_samples(gen_QCL, 1, 1000))
binary_sequence: np.ndarray = pack_sequence(sequence)
eligible_battery: dict = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
results = run_all_battery(binary_sequence, eligible_battery, False)
print("Результаты тестов для QCL:")
for result, elapsed_time in results[:7]:
    if result.passed:
        print(result.name + " - ПРОЙДЕН, score = " + str(np.round(result.score, 3)))
    else:
        print(result.name + " - НЕ ПРОЙДЕН, score = " + str(np.round(result.score, 3)))

"""@brief Генерация выборок разного объема"""
sizes = []
for i in range(1000, 1100000, 100000):
    sizes.append(i)

times_xor = []
for size in sizes:
    start_time = time.time()
    generate_samples(gen_XOR, 1, size)
    end_time = time.time()
    times_xor.append(end_time - start_time)

times_qcl = []
for size in sizes:
    start_time = time.time()
    generate_samples(gen_QCL, 1, size)
    end_time = time.time()
    times_qcl.append(end_time - start_time)

times_random = []
for size in sizes:
    start_time = time.time()
    for i in range(size):
        random.randint(10000, 1000000)
    end_time = time.time()
    times_random.append(end_time - start_time)

"""@brief Построение графиков"""
plt.plot(sizes, times_xor, label='XOR-Shift')
plt.plot(sizes, times_qcl, label='QCL')
plt.plot(sizes, times_random, label='Random')
plt.xlabel('Объем выборки')
plt.ylabel('Время генерации (секунды)')
plt.title('Сравнение скоростей генерации чисел')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()

