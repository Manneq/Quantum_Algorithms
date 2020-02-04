"""
    File 'plotting.py' has functions for plotting different data.
"""
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt


def quantum_circuit_plotting(quantum_circuit, folder,
                             font_size=14):
    """
        Method to quantum circuit plotting.
        param:
            1. quantum_circuit - qiskit QuantumCircuit
            2. folder - string name of folder where plot must be saved
            3. font_size - int value of text size on plot (14 as default)
    """
    plt.rcParams.update({'font.size': font_size})
    plot = quantum_circuit.draw(output='mpl')
    plt.title("Quantum circuit")
    plt.tight_layout()
    plot.savefig("plots/" + folder + "/quantum circuit.png")
    plt.close()

    return


def results_plotting(results, folder,
                     font_size=14):
    """
        Method to plot results of a simulation as histogram.
        param:
            1. results - qiskit simulation results
            2. folder - string name of folder where plot must be saved
            3. font_size - int value of text size on plot (14 as default)
    """
    plt.rcParams.update({'font.size': font_size})
    plot = plot_histogram(results)
    plt.title("Simulation results")
    plt.tight_layout()
    plot.savefig("plots/" + folder + "/simulation results.png")
    plt.close()

    return
