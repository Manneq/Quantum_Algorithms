"""
    File 'simulation.py' is file that contains functions for quantum
        algorithm simulation.
"""
import qiskit as qk
import plotting


def simulation(quantum_circuit, folder,
               shots=8192):
    """
        Method to draw and simulate quantum circuit of quantum algorithm.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. folder - string name of folder where plot must be saved
            3. shots - int value of simulation numbers (8192 as default)
    """
    # Plot quantum circuit
    plotting.quantum_circuit_plotting(quantum_circuit, folder)

    # Simulate the quantum algorithm results
    backend = qk.Aer.get_backend('qasm_simulator')
    simulation_results = \
        qk.execute(quantum_circuit, backend, shots=shots).result()

    # Plot results of simulation
    plotting.results_plotting(simulation_results.get_counts(quantum_circuit),
                              folder)

    return


def simons_simulation(quantum_circuit, secret_bit_length, folder,
                      shots=8192):
    """
        Method to draw and simulate quantum circuit of Simon's Algorithm.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. folder - string name of folder where plot must be saved
            3. shots - int value of simulation numbers (8192 as default)
    """
    # Plot quantum circuit
    plotting.quantum_circuit_plotting(quantum_circuit, folder)

    # Simulate the quantum algorithm results
    backend = qk.Aer.get_backend('qasm_simulator')
    simulation_results = \
        qk.execute(quantum_circuit, backend, shots=shots).result()
    answer = simulation_results.get_counts()

    # Make a selection of states of interest
    answer_plot = {}
    for measurement_result in answer.keys():
        measurement_result_input = measurement_result[secret_bit_length:]

        if measurement_result_input in answer_plot:
            answer_plot[measurement_result_input] += answer[measurement_result]
        else:
            answer_plot[measurement_result_input] = answer[measurement_result]

    # Plot results of simulation
    plotting.results_plotting(answer_plot, folder)

    return
