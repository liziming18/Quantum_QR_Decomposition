from utils import *
import matplotlib.pyplot as plt
import csv

cgs_errors = []
mgs_errors = []
qqr_errors = []
for size in range(2, 16):
    tmp1, tmp3 = ill_condition_QR_test(size, test_number=100, condition_number=1e2, use_classical_method=True)
    cgs_errors.append(tmp1)
    qqr_errors.append(tmp3)
    print(f'size {size} is done')

with open('qr_errors_size.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['size', 'cgs_errors','qqr_errors'])
    for i in range(14):
        writer.writerow([i+2, cgs_errors[i], qqr_errors[i]])

# cgs_average = []
# mgs_average = []
# qqr_average = []
# for i in range(14):
#     cgs_average.append(np.average(cgs_errors[i]))
#     mgs_average.append(np.average(mgs_errors[i]))
#     qqr_average.append(np.average(qqr_errors[i]))

# cgs_radius = []
# mgs_radius = []
# qqr_radius = []
# for i in range(14):
#     cgs_radius.append(np.array([cgs_average[i] - np.min(cgs_errors[i]), (np.max(cgs_errors[i] - cgs_average[i]))]))
#     mgs_radius.append(np.array([mgs_average[i] - np.min(mgs_errors[i]), (np.max(mgs_errors[i] - mgs_average[i]))]))
#     qqr_radius.append(np.array([qqr_average[i] - np.min(qqr_errors[i]), (np.max(qqr_errors[i] - qqr_average[i]))]))

# cgs_radius = np.array(cgs_radius).T
# mgs_radius = np.array(mgs_radius).T
# qqr_radius = np.array(qqr_radius).T

# plt.errorbar(range(2, 16), cgs_average, yerr=cgs_radius, label='CGS', marker='o', capsize=2)
# plt.errorbar(range(2, 16), mgs_average, yerr=mgs_radius, label='MGS', marker='o', capsize=2)
# plt.errorbar(range(2, 16), qqr_average, yerr=qqr_radius, label='QQR', marker='o', capsize=2)
# plt.yscale('log')
# plt.xlabel('Matrix size')
# plt.ylabel('error of different QR algorithm') 
# plt.legend(loc='upper left', shadow=True)
# plt.savefig('qr_errors_compare.png')