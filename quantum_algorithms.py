"""
    File 'quantum_algorithms.py' has implementation of quantum algorithms.
"""
import qiskit as qk
from qiskit.aqua.circuits.gates.multi_control_toffoli_gate import _cccx
import numpy as np
import simulation


"""
    Quantum Teleportation Algorithm implementation.
"""


def secret_unitary_applying(secret_unitary, qubit, quantum_circuit,
                            dagger=False):
    """
        Method to apply secret gates sequence for Quantum Teleportation
            Algorithm.
        param:
            1. secret_unitary - string of gates names
            2. qubit - qiskit QuantumRegister
            3. quantum_circuit - qiskit QuantumCircuit
            4. dagger - boolean value determin input or output (False as
                default)
    """
    # Determining map of gates
    function_map = {'x': quantum_circuit.x,
                    'y': quantum_circuit.y,
                    'z': quantum_circuit.z,
                    'h': quantum_circuit.h,
                    't': quantum_circuit.t}

    # Applying T gate
    if dagger:
        function_map['t'] = quantum_circuit.tdg

    # Reversing gates sequence
    if dagger:
        [function_map[unitary](qubit) for unitary in secret_unitary]
    else:
        [function_map[unitary](qubit) for unitary in secret_unitary[::-1]]

    return


def quantum_teleportation_algorithm(secret_unitary, folder):
    """
        Function to implement Quantum Teleportation Algorithm.
        param:
            1. secret_unitary - string of gates names
            2. folder - string name of folder where plot must be saved
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(3)
    classical_bits = qk.ClassicalRegister(3)
    quantum_circuit = qk.QuantumCircuit(qubits, classical_bits)

    # Applying the secret unitary
    secret_unitary_applying(secret_unitary, qubits[0], quantum_circuit)
    quantum_circuit.barrier()

    # Applying the teleportation protocol itself
    quantum_circuit.h(qubits[1])
    quantum_circuit.cx(qubits[1], qubits[2])
    quantum_circuit.barrier()

    quantum_circuit.cx(qubits[0], qubits[1])
    quantum_circuit.h(qubits[0])
    quantum_circuit.measure(qubits[:2], classical_bits[:2])
    quantum_circuit.cx(qubits[1], qubits[2])
    quantum_circuit.cz(qubits[0], qubits[2])
    quantum_circuit.barrier()

    # Applying the secret unitary for output
    secret_unitary_applying(secret_unitary, qubits[2], quantum_circuit,
                            dagger=True)

    # Measuring the output
    quantum_circuit.measure(qubits[2], classical_bits[2])

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Superdense Coding Protocol implementation.
"""


def superdense_coding_protocol(folder):
    """
        Function to implement Superdense Coding Protocol.
        param:
            folder - string name of folder where plot must be saved
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(2)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Generating the entangled pair
    quantum_circuit.h(qubits[0])
    quantum_circuit.cx(qubits[0], qubits[1])
    quantum_circuit.barrier()

    # Encoding message
    quantum_circuit.z(qubits[0])
    quantum_circuit.x(qubits[0])
    quantum_circuit.barrier()

    # Sending messsage
    quantum_circuit.cx(qubits[0], qubits[1])
    quantum_circuit.h(qubits[0])
    quantum_circuit.barrier()

    # Measuring output
    quantum_circuit.measure_all()

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Deutsch-Josza Algorithm implementation.
"""


def deutsch_josza_algorithm(string_length, folder,
                            quantum_oracle='b'):
    """
        Function to implement Superdense Coding Protocol.
        param:
            1. string_length - int length ob bit-string
            2. folder - string name of folder where plot must be saved
            3. quantum_oracle - char value of function type ('b' as default)
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(string_length + 1)
    classical_bits = qk.ClassicalRegister(string_length)
    quantum_circuit = qk.QuantumCircuit(qubits, classical_bits)

    # Flipping the last qubit
    quantum_circuit.x(qubits[string_length])
    quantum_circuit.barrier()

    # Applying Hadamard gates to all qubits
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Setting gates depending of function type
    if quantum_oracle == 'b':
        a = np.random.randint(1, 2**string_length)

        for i in range(string_length):
            if a & (1 << i):
                quantum_circuit.cx(qubits[i], qubits[string_length])
    else:
        a = np.random.randint(2)

        if a == 1:
            quantum_circuit.x(qubits[string_length])
        else:
            quantum_circuit.iden(qubits[string_length])

    quantum_circuit.barrier()

    # Applying Hadamard gates
    quantum_circuit.h(qubits[:string_length])

    # Measuring output
    quantum_circuit.measure(qubits[:string_length],
                            classical_bits[:string_length])

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Bernstein-Vazirani Algorithm implementation.
"""


def bernstein_vazirani_algorithm(secret_integer, folder):
    """
        Function to implement Bernstein-Vazirani Algorithm.
        param:
            1. secret_integer - int number
            2. folder - string name of folder where plot must be saved
    """
    # Determining the number of qubits
    qubits_number = secret_integer.bit_length()

    # Converting the number to bits
    secret_integer = secret_integer % 2 ** qubits_number

    # Creating quantum circuit
    qubits = qk.QuantumRegister(qubits_number)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Applying Hadamard gates
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Applying next gates depending of inner product
    for i in range(qubits_number):
        if secret_integer & (1 << i):
            quantum_circuit.z(qubits[i])
        else:
            quantum_circuit.iden(qubits[i])

    quantum_circuit.barrier()

    # Applying Hadamard gates
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Measuring output
    quantum_circuit.measure_all()

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Simon's Algorithm implementation.
"""


def simons_algorithm(secret_integer, folder):
    """
        Function to implement Simon's Algorithm.
        param:
            1. secret_integer - int number
            2. folder - string name of folder where plot must be saved
    """
    # Determining the number of qubits
    secret_bit_length = secret_integer.bit_length()
    qubits_number = secret_integer.bit_length() * 2

    # Creating quantum circuit
    qubits = qk.QuantumRegister(qubits_number)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Applying Hadamard gates
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Applying the query function
    for i in range(secret_bit_length):
        for j in range(secret_bit_length, secret_bit_length * 2):
            quantum_circuit.cx(qubits[i], qubits[j])

    quantum_circuit.barrier()

    # Applying hadamard gates
    quantum_circuit.h(qubits[:secret_integer.bit_length()])
    quantum_circuit.barrier()

    # Measuring output
    quantum_circuit.measure_all()

    # Simulating Simon's quantum circuit
    simulation.simons_simulation(quantum_circuit, secret_bit_length, folder)

    return


"""
    Quantum Fourier Transform implementation.
"""


def input_state(quantum_circuit, qubits_number):
    """
        Method to define qubits input state.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    for i in range(qubits_number):
        quantum_circuit.h(i)
        quantum_circuit.u1(-np.pi / (2 ** i), i)

    return quantum_circuit


def registers_swapping(quantum_circuit, qubits_number):
    """
        Method to swap register of qubits.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    for i in range(int(np.floor(qubits_number / 2))):
        quantum_circuit.swap(i, qubits_number - i - 1)

    return quantum_circuit


def quantum_fourier_transform(quantum_circuit, qubits_number):
    """
        Method to implement Quantum Fourier Transform for defined
            qubits number.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    for i in range(qubits_number):
        quantum_circuit.h(i)

        for j in range(i + 1, qubits_number):
            quantum_circuit.cu1(np.pi / (2 ** (j - i)), j, i)

        quantum_circuit.barrier()

    quantum_circuit = registers_swapping(quantum_circuit, qubits_number)

    return quantum_circuit


def quantum_fourier_transform_algorithm(qubits_number, folder):
    """
        Function to implement Quantum Fourier Transform.
        param:
            1. qubits_number - int number of qubits
            2. folder - string name of folder where plot must be saved
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(qubits_number)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Defining qubits input state
    quantum_circuit = input_state(quantum_circuit, qubits_number)
    quantum_circuit.barrier()

    # Transforming the qubits
    quantum_circuit = quantum_fourier_transform(quantum_circuit, qubits_number)

    # Measuring output
    quantum_circuit.measure_all()

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Quantum Phase Estimation implementation.
"""


def quantum_fourier_transform_estimation(quantum_circuit, qubits_number):
    """
        Method to implement inversed Quantum Fourier Transform for defined
            qubits number.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    for qubit in range(int(qubits_number / 2)):
        quantum_circuit.swap(qubit, qubits_number - qubit - 1)

    for i in range(qubits_number, 0, -1):
        for j in range(qubits_number - i):
            quantum_circuit.cu1(-np.pi / (2 ** (qubits_number - i - j)),
                                qubits_number - j - 1, i - 1)

        quantum_circuit.h(i - 1)

    return quantum_circuit


def quantum_phase_estimation_algorithm(folder):
    """
        Function to implement Quantum Phase Estimation.
        param:
            folder - string name of folder where plot must be saved
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(4)
    classical_bits = qk.ClassicalRegister(3)
    quantum_circuit = qk.QuantumCircuit(qubits, classical_bits)

    # Applying X and Hadamard gates
    quantum_circuit.x(qubits[3])
    quantum_circuit.h(qubits[:3])

    # Performing controlled unitary operations
    iterations = 4
    for qubit in range(3):
        for i in range(iterations):
            quantum_circuit.cu1(np.pi / 4, qubit, 3)

        iterations //= 2

    # Applying inverse QFT
    quantum_circuit = quantum_fourier_transform_estimation(quantum_circuit, 3)

    # Measuring output
    quantum_circuit.measure(0, 2)
    quantum_circuit.measure(1, 1)
    quantum_circuit.measure(2, 0)

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Grover's Algorithm implementation.
"""


def phase_quantum_oracle(quantum_circuit, qubits):
    """
        Method to mark states of qubits for searching.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits - qiskit QuantumRegister
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    quantum_circuit.cz(qubits[2], qubits[0])
    quantum_circuit.cz(qubits[2], qubits[1])

    return quantum_circuit


def controlled_z_gate(quantum_circuit, controls, target):
    """
        Method to apply controlled Z gates.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. controls - qiskit QuantumRegister for controling the gates
            3. target - qiskit QuantumRegister for gates to apply
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    if len(controls) == 1:
        quantum_circuit.h(target)
        quantum_circuit.cx(controls[0], target)
        quantum_circuit.h(target)
    else:
        quantum_circuit.h(target)
        quantum_circuit.ccx(controls[0], controls[1], target)
        quantum_circuit.h(target)

    return quantum_circuit


def inversion_about_average(quantum_circuit, qubits, qubits_number):
    """
        Method to apply inversion about the average step of Grover's algorithm.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. qubits - qiskit QuantumRegister
            3. qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    quantum_circuit.h(qubits)
    quantum_circuit.x(qubits)
    quantum_circuit.barrier()

    quantum_circuit = controlled_z_gate(quantum_circuit,
                                        qubits[:qubits_number - 1],
                                        qubits[qubits_number - 1])
    quantum_circuit.barrier()

    quantum_circuit.x(qubits)
    quantum_circuit.h(qubits)

    return quantum_circuit


def grovers_algorithm(folder):
    """
        Function to implement Grover's Algorithm.
        param:
            folder - string name of folder where plot must be saved
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(3)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Applying Hadamard gates
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Marking the states for searching
    quantum_circuit = phase_quantum_oracle(quantum_circuit, qubits)
    quantum_circuit.barrier()

    # Inverting circuit
    quantum_circuit = inversion_about_average(quantum_circuit, qubits, 3)
    quantum_circuit.barrier()

    # Measuring output
    quantum_circuit.measure_all()

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return


"""
    Quantum Counting implementation.
"""


def grovers_iterations():
    """
        Method to simulate Grover's iterations for quantum circuit.
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(4)
    quantum_circuit = qk.QuantumCircuit(qubits)

    # Using quantum oracle
    quantum_circuit.h(qubits[3])
    _cccx(quantum_circuit, qubits)
    quantum_circuit.x(qubits[0])
    _cccx(quantum_circuit, qubits)
    quantum_circuit.x(qubits[0])
    quantum_circuit.x(qubits[1])
    _cccx(quantum_circuit, qubits)
    quantum_circuit.x(qubits[1])
    quantum_circuit.x(qubits[2])
    _cccx(quantum_circuit, qubits)
    quantum_circuit.x(qubits[2])
    quantum_circuit.x(qubits[1])
    quantum_circuit.x(qubits[2])
    _cccx(quantum_circuit, qubits)
    quantum_circuit.x(qubits[2])
    quantum_circuit.x(qubits[1])
    quantum_circuit.h(qubits[3])
    quantum_circuit.z(qubits[3])

    # Making Diffusion Operation
    for qubit in qubits[:3]:
        quantum_circuit.h(qubit)
        quantum_circuit.x(qubit)

    _cccx(quantum_circuit, qubits)

    for qubit in qubits[:3]:
        quantum_circuit.x(qubit)
        quantum_circuit.h(qubit)

    quantum_circuit.z(qubits[3])

    return quantum_circuit


def quantum_fourier_transform_grovers(qubits_number):
    """
        Method to apply QFT on defined number of qubits.
        param:
            qubits_number - int number of qubits
        return:
            quantum_circuit - qiskit QuantumCircuit
    """
    # Creating quantum circuit
    qubits = qk.QuantumRegister(qubits_number)
    quantum_circuit = qk.QuantumCircuit(qubits)

    for i in range(qubits_number):
        quantum_circuit.h(qubits[i])

        for j in range(i + 1, qubits_number):
            quantum_circuit.cu1(np.pi / (2 ** (j - i)), qubits[j], qubits[i])

    # Swapping quantum circuit
    for i in range(int(qubits_number / 2)):
        quantum_circuit.swap(qubits[i], qubits[qubits_number - i - 1])

    return quantum_circuit


def quantum_counting(folder):
    """
        Function to implement Quantum Counting.
        param:
            folder - string name of folder where plot must be saved
    """
    # Determining the number of qubits for counting and searching
    qubits_number_counting = 4
    qubits_number_searching = 4

    # Creating quantum circuit
    qubits = \
        qk.QuantumRegister(qubits_number_counting + qubits_number_searching)
    classical_bits = qk.ClassicalRegister(qubits_number_searching)
    quantum_circuit = qk.QuantumCircuit(qubits, classical_bits)

    # Applying Hadamard gates
    quantum_circuit.h(qubits)
    quantum_circuit.barrier()

    # Applying controled Grover's iterations
    iterations = 1
    for qubit in reversed(qubits[:4]):
        for i in range(iterations):
            quantum_circuit.append(
                grovers_iterations().to_gate().control(),
                qargs=[qubit] + qubits[qubits_number_counting:])
        iterations *= 2

    quantum_circuit.barrier()

    # Applying inverse QFT
    quantum_circuit.append(
        quantum_fourier_transform_grovers(qubits_number_counting).to_gate().
        inverse(),
        qargs=qubits[:qubits_number_counting])
    quantum_circuit.barrier()

    # Measuring output
    quantum_circuit.measure(qubits[:qubits_number_counting], classical_bits)

    # Simulating quantum circuit
    simulation.simulation(quantum_circuit, folder)

    return
