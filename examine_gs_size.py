from utils import *
import matplotlib.pyplot as plt
import csv

cgs_errors = []
mgs_errors = []
qgs_errors = []
for size in range(2, 16):
    tmp1, tmp2, tmp3 = ill_condition_gs_test(size, test_number=100, condition_number=1e2)
    cgs_errors.append(tmp1)
    mgs_errors.append(tmp2)
    qgs_errors.append(tmp3)
    
cgs_average = []
mgs_average = []
qgs_average = []
for i in range(14):
    cgs_average.append(np.average(cgs_errors[i]))
    mgs_average.append(np.average(mgs_errors[i]))
    qgs_average.append(np.average(qgs_errors[i]))

cgs_radius = []
mgs_radius = []
qgs_radius = []
for i in range(14):
    cgs_radius.append(np.array([cgs_average[i] - np.min(cgs_errors[i]), (np.max(cgs_errors[i] - cgs_average[i]))]))
    mgs_radius.append(np.array([mgs_average[i] - np.min(mgs_errors[i]), (np.max(mgs_errors[i] - mgs_average[i]))]))
    qgs_radius.append(np.array([qgs_average[i] - np.min(qgs_errors[i]), (np.max(qgs_errors[i] - qgs_average[i]))]))

cgs_radius = np.array(cgs_radius).T
mgs_radius = np.array(mgs_radius).T
qgs_radius = np.array(qgs_radius).T

plt.errorbar(range(2, 16), cgs_average, yerr=cgs_radius, label='CGS', marker='o', capsize=2)
plt.errorbar(range(2, 16), mgs_average, yerr=mgs_radius, label='MGS', marker='o', capsize=2)
plt.errorbar(range(2, 16), qgs_average, yerr=qgs_radius, label='QGS', marker='o', capsize=2)
plt.yscale('log')
plt.xlabel('Matrix size')
plt.ylabel('error of different GS algorithm') 
plt.legend(loc='upper left', shadow=True)
plt.savefig('gs_errors_compare.png')