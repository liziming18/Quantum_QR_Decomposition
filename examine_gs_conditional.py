from utils import *
import matplotlib.pyplot as plt
import csv

qgs_errors = []
condition_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
for cond in condition_list:
    tmp3 = ill_condition_gs_test(size=8, test_number=30, condition_number=cond)
    qgs_errors.append(tmp3)
    print(f'condition number {cond} is done!')

qgs_average = []
for i in range(len(qgs_errors)):
    qgs_average.append(np.average(qgs_errors[i]))

qgs_radius = []
for i in range(len(qgs_errors)):
    qgs_radius.append(np.array([qgs_average[i] - np.min(qgs_errors[i]), (np.max(qgs_errors[i])) - qgs_average[i]]))


qgs_radius = np.array(qgs_radius).T

plt.errorbar(condition_list, qgs_average, yerr=qgs_radius, label='error range', marker='o', capsize=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix Condition Number')
plt.ylabel('average error of quantum QR algorithm') 
plt.legend(loc='upper left', shadow=True)
plt.savefig('gs_errors_conditional.png')