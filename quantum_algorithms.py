# Description: This file contains the implementation of the quantum algorithms, including QIPE and QGS and QQR algorithms.

from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# quantum modified gram-schmidt process
def quantum_gram_schmidt(vectors, error_rate=1e-4, circuit_return=False):
    '''
    quantum modified gram-schmidt process
    vectors: input vectors, numpy array, vectors = [[v1], [v2], ..., [vn]]
    error_rate: error rate of the process
    '''
    vector_size = len(vectors[0])
    vector_number = len(vectors)
    qubit_number = int(np.ceil(np.log2(vector_size))) # qubit number for the 2nd register
    if 2 ** qubit_number < vector_size:
        qubit_number += 1
    total_qubit_number = qubit_number + 1 # qubit number for the quantum circuit
    max_runtime_of_circuit = int(np.log(1/error_rate) / error_rate)

    first_constructed_vector = vectors[0] / np.linalg.norm(vectors[0])
    first_constructed_vector_extended = np.concatenate((first_constructed_vector, np.zeros(2 ** qubit_number - vector_size)))
    # print(first_constructed_vector_extended)

    constructed_basis = [first_constructed_vector]

    current_hamiltonian = np.zeros((2**(total_qubit_number), 2**(total_qubit_number)), dtype=complex)
    current_hamiltonian += np.kron(np.outer(np.conjugate(first_constructed_vector_extended), first_constructed_vector_extended), np.array([[0, 0], [0, 1]], dtype=complex))
    # print(current_hamiltonian)
    # print(total_qubit_number)
    qubit_list = [i for i in range(total_qubit_number)]
    qc_set = []

    for i in range(1, vector_number):
        v = vectors[i] / np.linalg.norm(vectors[i]) # normalize input state
        v = np.concatenate((v, np.zeros(2 ** qubit_number - vector_size))) # input state as amplitude encoding
        count = 0
        qc = QuantumCircuit(total_qubit_number, 1)
        qc.initialize(v, range(1, total_qubit_number))
        qc.h(0)
        qc.hamiltonian(current_hamiltonian, np.pi, qubit_list, label='Hami')
        qc.h(0)
        qc.measure(0, 0)
        while count < max_runtime_of_circuit:
            count += 1
            simulator = Aer.get_backend('statevector_simulator')
            job = execute(qc, simulator, shots=1)
            result = job.result()
            if '0' in result.get_counts().keys():
                qc_set.append(qc)
                statevector = np.asarray(result.get_statevector()) 
                # print(statevector)
                current_hamiltonian += np.outer(np.conjugate(statevector), statevector)
                reduced_statevector = np.zeros(vector_size, dtype=complex)
                for j in range(vector_size):
                    reduced_statevector[j] = statevector[2 * j]
                constructed_basis.append(reduced_statevector)
                break
        print(f'count={count}')

    length = len(constructed_basis)
    constructed_basis = np.asarray(constructed_basis)

    if circuit_return:
        return qc_set, constructed_basis, length
    else:
        return constructed_basis, length
    

def swap_test(state1, state2, error=1e-3, return_circuit=False):
    '''
    This function calculates the overlap between two states using the swap test.
    The states are given as vectors.
    The function returns the overlap |<\psi_1|\psi_2>|^2.
    '''
    # create a quantum circuit with 3 quantum registers and 1 classical bit
    length = len(state1)
    qubit_number = int(np.ceil(np.log2(length))) # qubit number for the 2nd register
    if 2 ** qubit_number < length:
        qubit_number += 1

    qc = QuantumCircuit(qubit_number * 2 + 1, 1)
    # initialize the first two qubits with the states
    extended_state1 = np.zeros(2 ** qubit_number)
    extended_state2 = np.zeros(2 ** qubit_number)
    extended_state1[:length] = state1
    extended_state2[:length] = state2

    qc.initialize(extended_state1, range(qubit_number))
    qc.initialize(extended_state2, range(qubit_number, 2 * qubit_number))
    
    # apply the Hadamard gate to the third qubit
    qc.h(2 * qubit_number)
    
    # # apply the controlled swap gate
    for i in range(qubit_number):
        qc.cswap(2 * qubit_number, i, i + qubit_number)
    
    # apply the Hadamard gate to the third qubit
    qc.h(2 * qubit_number)
    
    # measure the 3rd qubit
    qc.measure(2 * qubit_number, 0)
    
    # use the qasm simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # execute the circuit
    shots = int(1 / (error) ** 2)
    result = execute(qc, simulator, shots=shots).result()
    
    # get the counts
    print(result.get_counts(qc))
    counts = result.get_counts(qc)
    
    # get the probability of success
    prob_success = counts['0'] / shots
    overlap = 2 * prob_success - 1
    
    if return_circuit:
        return overlap, qc
    else:
        return overlap
    