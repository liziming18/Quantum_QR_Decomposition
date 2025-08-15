import copy
from quantum_algorithms import *

# classical gram-schmidt process
def classical_gram_schmidt(vectors):
    vectors_record = copy.deepcopy(vectors)
    basis = []
    for v in vectors_record:
        tmp = copy.deepcopy(v)
        for b in basis:
            v -= np.dot(np.conjugate(b), tmp) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
    basis = np.array(basis)
    return basis, len(basis)

# classical modified gram-schmidt process
def classical_modified_gram_schmidt(vectors):
    vectors_record = copy.deepcopy(vectors)
    basis = []
    for v in vectors_record:
        for b in basis:
            v -= np.dot(np.conjugate(b), v) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
    basis = np.array(basis)
    return basis, len(basis)

# use gram-schhmidt process for QR decomposition
def QR_decomposition(A, choosen_method='mgs', circuit_return=False, error=1e-4):
    '''
    QR decomposition of matrix A
    A: input matrix, numpy array
    choosen_method: method for QR decomposition, 'cgs' or 'mgs' or 'quantum'
    '''
    current_A = copy.deepcopy(A)
    current_A = np.array(current_A, dtype=complex)
    if choosen_method == 'cgs':
        basis, length = classical_gram_schmidt(current_A.T)
    elif choosen_method == 'mgs':
        basis, length = classical_modified_gram_schmidt(current_A.T)
    elif choosen_method == 'quantum':
        if circuit_return:
            circuit_set, basis, length = quantum_gram_schmidt(A.T, circuit_return=circuit_return, error_rate=error)
        else:
            basis, length = quantum_gram_schmidt(A.T, error_rate=error)
    else:
        raise ValueError('choosen_method should be cgs or mgs or quantum')
    
    Q = basis.T
    R = np.zeros((length, len(A.T)), dtype=complex)
    for i in range(length):
        for j in range(i, len(A.T)):
            R[i, j] = np.dot(np.conjugate(basis[i]), A.T[j])

    if choosen_method == 'quantum' and circuit_return:
        return circuit_set, Q, R
    return Q, R

    
# use qr decomposition to do least square fitting
def least_square_fitting(A, b, choosen_method='mgs'):
    '''
    least square fitting
    A: input matrix, numpy array
    b: input vector, numpy array
    choosen_method: method for QR decomposition, 'cgs' or 'mgs' or 'quantum'
    '''
    Q, R = QR_decomposition(A, choosen_method)
    x = np.dot(np.linalg.inv(R), np.dot(np.conjugate(Q.T), b))
    return x, np.linalg.norm(np.dot(A, x) - b)

# use qr decomposition to solve linear equations
def solve_linear_equations(A, b, choosen_method='mgs'):
    '''
    solve linear equations
    A: input matrix, numpy array
    b: input vector, numpy array
    choosen_method: method for QR decomposition, 'cgs' or 'mgs' or 'quantum'
    '''
    Q, R = np.linalg.qr(A) #QR_decomposition(A, choosen_method)
    x = np.dot(np.linalg.inv(R), np.dot(np.conjugate(Q.T), b))
    return x

# use qr decomposition to do eigenvalue solving
def qr_algorithm(A, choosen_method='mgs', max_iteration=1000, eps=1e-6, record=False):
    '''
    qr algorithm
    A: input matrix, numpy array
    choosen_method: method for QR decomposition, 'cgs' or 'mgs' or 'quantum'
    max_iteration: maximum iteration number
    '''
    recorder = []
    Ak = copy.deepcopy(A)
    transform_matrix = np.eye(len(A))
    for i in range(max_iteration):
        Ak0 = Ak.copy()
        if record:
            recorder.append(np.diag(Ak))
        Qk, Rk = QR_decomposition(Ak, choosen_method)
        Ak = np.dot(Rk, Qk)
        transform_matrix = np.dot(transform_matrix, Qk)     
        if (np.sum(np.diag(np.abs(Ak-Ak0)))<eps):
            break
        else:
            print(f'iteration {i} done')
    if record:
        return Ak, transform_matrix, recorder
    else:
        return Ak, transform_matrix



# test functions for gram-schmidt process

def ill_condition_matrix_generation(size=10, number=100, condition_number=1e3):
    '''
    generate ill-conditioned matrices
    size: size of the matrix
    number: number of matrices to generate
    condition_number: condition number of the matrix
    note: the condition number is defined as the ratio of the largest singular value to the smallest singular value, the generated matrices are all full rank matrices
    '''
    record = []
    for _ in range(number):
        np.random.seed(np.random.randint(0, 1000000))
        random_matrix = np.random.rand(size, size) #+ 1j * np.random.rand(size, size)
        q, _ = np.linalg.qr(random_matrix)
        d = np.zeros(size)
        d[0] = 1
        d[size - 1] = 1 / condition_number
        # exponential decay
        for i in range(1, size - 1):
            d[i] = d[i - 1] / (condition_number ** (1 / (size - 1)))
        D = np.diag(d)

        ill_condition_matrix = np.dot(q, np.dot(D, np.conjugate(q.T)))
        record.append(ill_condition_matrix)
    return record

def hilbert_matrix_gs_test(size=10):
    '''
    test the gram-schmidt process on hilbert matrix
    '''
    vector_size = size
    vector_number = vector_size

    vectors = np.zeros((vector_number, vector_size))
    for i in range(vector_number):
        for j in range(vector_size):
            vectors[i, j] = 1 / (i + j + 1)

    basis1,size_of_generated_set1 = classical_gram_schmidt(vectors)
    basis2,size_of_generated_set2 = classical_modified_gram_schmidt(vectors)
    basis3,size_of_generated_set3 = quantum_gram_schmidt(vectors)
    print(basis2)
    print(size_of_generated_set2)
    print('-----------------')
    print(basis3)
    print(size_of_generated_set3)
    print('-----------------')
    print(f'error of CGS on hilbert matrix size {size} is: {np.linalg.norm(np.dot(basis1, np.conjugate(basis1.T)) - np.eye(size_of_generated_set1))}')
    print(f'error of MGS on hilbert matrix size {size} is: {np.linalg.norm(np.dot(basis2, np.conjugate(basis2.T)) - np.eye(size_of_generated_set2))}')
    print(f'error of quantum GS on hilbert matrix size {size} is: {np.linalg.norm(np.dot(basis3, np.conjugate(basis3.T)) - np.eye(size_of_generated_set3))}')


def ill_condition_gs_test(size=10, test_number=100, condition_number=1e3, use_classical_method=False):
    '''
    test the gram-schmidt process on ill-conditioned matrices
    '''
    record = ill_condition_matrix_generation(size=size, number=test_number, condition_number=condition_number)
    error_cgs = []
    error_mgs = []
    error_qgs = []
    if use_classical_method:
        for i in range(test_number):
            basis1, size_of_generated_set1 = classical_gram_schmidt(record[i])
            error_cgs.append(np.linalg.norm(np.dot(basis1, np.conjugate(basis1.T)) - np.eye(size_of_generated_set1)))

            basis2, size_of_generated_set2 = classical_modified_gram_schmidt(record[i])
            error_mgs.append(np.linalg.norm(np.dot(basis2, np.conjugate(basis2.T)) - np.eye(size_of_generated_set2)))

            basis3, size_of_generated_set3 = quantum_gram_schmidt(record[i])
            error_qgs.append(np.linalg.norm(np.dot(basis3, np.conjugate(basis3.T)) - np.eye(size_of_generated_set3)))

        return error_cgs, error_mgs, error_qgs
    else:
        for i in range(test_number):
            basis3, size_of_generated_set3 = quantum_gram_schmidt(record[i])
            error_qgs.append(np.linalg.norm(np.dot(basis3, np.conjugate(basis3.T)) - np.eye(size_of_generated_set3)))

        return error_qgs
        


# test functions for QR decomposition
def hilbert_matrix_QR_test(size=10):
    '''
    test the QR decomposition on hilbert matrix
    '''
    vector_size = size
    vector_number = vector_size

    vectors = np.zeros((vector_number, vector_size))
    for i in range(vector_number):
        for j in range(vector_size):
            vectors[i, j] = 1 / (i + j + 1)

    Q1, R1 = QR_decomposition(vectors, choosen_method='cgs')
    Q2, R2 = QR_decomposition(vectors, choosen_method='mgs')
    Q3, R3 = QR_decomposition(vectors, choosen_method='quantum')

    print(f'error of CGS based QR on hilbert matrix size {size} is: {np.linalg.norm(np.dot(Q1, R1) - vectors)}')
    print(f'error of MGS based QR on hilbert matrix size {size} is: {np.linalg.norm(np.dot(Q2, R2) - vectors)}')
    print(f'error of quantum GS based QR on hilbert matrix size {size} is: {np.linalg.norm(np.dot(Q3, R3) - vectors)}')

def ill_condition_QR_test(size=10, test_number=100, condition_number=1e3, use_classical_method=False, error=1e-4):
    '''
    test the QR decomposition on ill-conditioned matrices
    '''
    record = ill_condition_matrix_generation(size=size, number=test_number, condition_number=condition_number)
    error_cgs = []
    error_qgs = []
    if use_classical_method:
        for i in range(test_number):
            Q1, R1 = QR_decomposition(record[i], choosen_method='cgs')
            error_cgs.append(np.linalg.norm(np.dot(Q1, R1) - record[i]))

            # Q2, R2 = QR_decomposition(record[i], choosen_method='mgs')
            # error_mgs.append(np.linalg.norm(np.dot(Q2, R2) - record[i]))

            Q3, R3 = QR_decomposition(record[i], choosen_method='quantum', error=error)
            error_qgs.append(np.linalg.norm(np.dot(Q3, R3) - record[i]))

            print(f'run {i} is done')

        return error_cgs, error_qgs
    else:
        for i in range(test_number):
            Q3, R3 = QR_decomposition(record[i], choosen_method='quantum', error=error)
            error_qgs.append(np.linalg.norm(np.dot(Q3, R3) - record[i]))

        return error_qgs


# test functions for solving linear equations not least square fitting
def hilbert_matrix_linear_equations_test(size=5):
    '''
    test the linear equations solving on hilbert matrix
    '''
    vector_size = size
    vector_number = vector_size

    vectors = np.zeros((vector_number, vector_size))
    for i in range(vector_number):
        for j in range(vector_size):
            vectors[i, j] = 1 / (i + j + 1)

    b = np.random.rand(vector_size) + 1j * np.random.rand(vector_size)
    x = solve_linear_equations(vectors, b, choosen_method='mgs')

    print(f'error of linear equations solving on hilbert matrix size {size} is: {np.linalg.norm(np.dot(vectors, x) - b)}')

def ill_condition_linear_equations_test(size=5, test_number=100, condition_number=1e3):
    '''
    test the linear equations solving on ill-conditioned matrices
    '''
    record = ill_condition_matrix_generation(size=size, number=test_number, condition_number=condition_number)
    abs_error = []
    rela_error = []
    for i in range(test_number):
        b = np.random.rand(size) + 1j * np.random.rand(size)
        x = solve_linear_equations(record[i], b, choosen_method='mgs')
        abs_error.append(np.linalg.norm(np.dot(record[i], x) - b))
        rela_error.append(abs_error[-1] / np.linalg.norm(b))

    return abs_error, rela_error



# test functions for qr_algorithm
def hilbert_matrix_qr_algorithm_test(size=5):
    '''
    test the qr algorithm on hilbert matrix
    '''
    vector_size = size
    vector_number = vector_size

    vectors = np.zeros((vector_number, vector_size))
    for i in range(vector_number):
        for j in range(vector_size):
            vectors[i, j] = 1 / (i + j + 1)

    eigenvalues = qr_algorithm(vectors, choosen_method='mgs')

    print(f'error of QR algorithm on libert matrix size {size} is: {np.linalg.norm(np.sort(eigenvalues) - np.sort(np.linalg.eigvals(vectors)))}')

def ill_condition_qr_algorithm_test(size=2, test_number=100, condition_number=1e3):
    '''
    test the qr algorithm on ill-conditioned matrices
    '''
    record = ill_condition_matrix_generation(size=size, number=test_number, condition_number=condition_number)
    abs_error = []
    rela_error = []
    for i in range(test_number):
        eigenvalues = qr_algorithm(record[i], choosen_method='mgs')
        abs_error.append(np.linalg.norm(np.sort(eigenvalues) - np.sort(np.linalg.eigvals(record[i]))))
        rela_error.append(abs_error[-1] / np.abs(np.trace(record[i])))

    return abs_error, rela_error