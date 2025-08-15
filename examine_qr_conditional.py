from utils import *
import matplotlib.pyplot as plt
import csv
import time

qqr_errors = []
condition_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
start_time = time.time()
for cond in condition_list:
    tmp3 = ill_condition_QR_test(size=8, test_number=3, condition_number=cond)
    qqr_errors.append(tmp3)
    print(f'condition number {cond} is done!')


with open('qr_errors_conditional.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['conditional_number', 'error'])
    for i in range(len(condition_list)):
        writer.writerow([condition_list[i], qqr_errors[i]])

# qqr_average = []
# for i in range(len(qqr_errors)):
#     qqr_average.append(np.average(qqr_errors[i]))

# qqr_radius = []
# for i in range(len(qqr_errors)):
#     qqr_radius.append(np.array([qqr_average[i] - np.min(qqr_errors[i]), (np.max(qqr_errors[i])) - qqr_average[i]]))


# qqr_radius = np.array(qqr_radius).T

# plt.errorbar(condition_list, qqr_average, yerr=qqr_radius, label='error range', marker='o', capsize=2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Matrix Condition Number')
# plt.ylabel('average error of quantum QR algorithm') 
# plt.legend(loc='upper left', shadow=True)
# plt.savefig('qr_errors_conditional.png')