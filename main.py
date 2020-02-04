"""
Name:       Quantum algorithms
Purpose:    Implement and simulate next quantum algorithms:
                1. Quantum Teleportation Algorithm
                2. Superdense Coding Protocol
                3. Deutsch-Josza Algorithm
                4. Bernstein-Vazirani Algorithm
                5. Simon's Algorithm
                6. Quantum Fourier Transformation (QFT)
                7. Quantum Phase Estimation
                8. Grover's Algorithm
                9. Quantum Counting
Author:     Artem "Manneq" Arkhipov
Created:    04/02/2020
"""
import quantum_algorithms


"""
    File 'main.py' is main file that controls the sequence of function calls.
"""


def main():
    """
        Main function.
    """
    # Simulating Quantum Teleportation Algorithm
    quantum_algorithms.quantum_teleportation_algorithm("hzyxht",
                                                       "quantum teleportation")

    # Simulating Superdense Coding Protocol
    quantum_algorithms.superdense_coding_protocol("superdense coding")

    # Simulating Deutsch-Jozsa Algorithm for balanced function for
    # 10-bit string
    quantum_algorithms.deutsch_josza_algorithm(10, "deutch josza/balanced",
                                               quantum_oracle='b')

    # Simulating Deutsch-Jozsa Algorithm for constant function for
    # 10-bit string
    quantum_algorithms.deutsch_josza_algorithm(10, "deutch josza/constant",
                                               quantum_oracle='c')

    # Simulating Bernstein-Vazirani Algorithm for number 137
    quantum_algorithms.bernstein_vazirani_algorithm(137, "bernstein vazirani")

    # Simulating Simon's Algorithm for 2-bit number
    quantum_algorithms.simons_algorithm(3, "simons algorithm")

    # Simulating Quantum Fourier Transformation for 3 qubits
    quantum_algorithms.\
        quantum_fourier_transform_algorithm(3,
                                            "quantum fourier transformation")

    # Simulating Quantum Phase Estimation for 3 qubits
    quantum_algorithms.\
        quantum_phase_estimation_algorithm("quantum phase estimation")

    # Simulating Grover's Algorithm for 3 qubits
    quantum_algorithms.grovers_algorithm("grovers algorithm")

    # Simulating Quantum Counting for 4 qubits
    quantum_algorithms.quantum_counting("quantum counting")

    return


if __name__ == "__main__":
    main()
