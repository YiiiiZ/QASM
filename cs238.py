#imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import Counter
import re

# Define quantum gates as unitary matrices
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # H gate
X = np.array([[0, 1], [1, 0]])  # X gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])  # T gate
T_dagger = np.conjugate(T.T)  # T-dagger gate
CX = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]]) 

class QuantumCircuit:
    """
    A class to represent a quantum circuit simulator with big endian state vector.
    
    Attributes:
        num_qubits (int): Number of qubits in the circuit.
        state (np.ndarray): The state vector representing the quantum state in big endian format.
        gates (list): A list of gates to be applied to the circuit.
    """
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        # Initialize the state |0...0⟩ in big endian format
        self.state[0] = 1
        self.gates = []  # to be appended

    def apply_gate(self, gate, target, control=None):
        gate_lower = gate.lower()
        if gate_lower == 'cx' and control is not None:
            self.state = self.apply_cx_gate(self.state, control, target)
        else:
            U = {'h': H, 'x': X, 't': T, 'tdg': T_dagger}.get(gate_lower)
            if U is None:
                raise ValueError(f"Unsupported gate: {gate}")
            if control is None:
                self.state = self.apply_single_qubit_gate(self.state, U, target)  # Update state vector
            else:
                self.state = self.apply_controlled_gate(self.state, U, control, target)  # Update state vector

    def apply_single_qubit_gate(self, state, U, target):
        """
        Applies a single-qubit gate to the specified target qubit (big endian).
        
        Args:
            state (np.ndarray): The current state vector of the quantum circuit.
            U (np.ndarray): The unitary matrix representing the gate.
            target (int): The index of the target qubit (0 is the most significant qubit).
        
        Returns:
            np.ndarray: The updated state vector after applying the gate.
        """
        num_qubits = self.num_qubits
        N = len(state)  # N = number of possible states
        indices = np.arange(N)
        # Adjust the target index for big endian format
        adjusted_target = num_qubits - 1 - target
        qubit_mask = 1 << adjusted_target  # Left shift 1 by 'adjusted_target' positions
        idx0 = indices[(indices & qubit_mask) == 0]  # Indices where the target qubit is |0⟩
        idx1 = idx0 + qubit_mask  # Indices where the target qubit is |1⟩
        a = state[idx0].copy()
        b = state[idx1].copy()
        # Apply gate matrix to paired amplitudes
        state[idx0] = U[0, 0] * a + U[0, 1] * b
        state[idx1] = U[1, 0] * a + U[1, 1] * b
        return state

    def apply_cx_gate(self, state, control, target):
        """
        Applies a Controlled-X (CNOT) gate to the quantum circuit (big endian).
        
        Args:
            state (np.ndarray): The current state vector of the quantum circuit.
            control (int): The index of the control qubit (0 is the most significant qubit).
            target (int): The index of the target qubit (0 is the most significant qubit).
        
        Returns:
            np.ndarray: The updated state vector after applying the CNOT gate.
        """
        num_qubits = self.num_qubits
        N = len(state)
        indices = np.arange(N)
        # Adjust indices for big endian format
        adjusted_control = num_qubits - 1 - control
        adjusted_target = num_qubits - 1 - target
        control_mask = 1 << adjusted_control
        target_mask = 1 << adjusted_target
        # Indices where control qubit is |1⟩ and target qubit is |0⟩
        idx_control1_target0 = indices[((indices & control_mask) != 0) & ((indices & target_mask) == 0)]
        idx_control1_target1 = idx_control1_target0 + target_mask
        # Swap the amplitudes of target qubit when control is |1⟩
        temp = state[idx_control1_target0].copy()
        state[idx_control1_target0] = state[idx_control1_target1]
        state[idx_control1_target1] = temp
        return state

    def run(self):
        """Execute gates in the quantum circuit in the order they were added."""
        for gate, target, control in self.gates:
            self.apply_gate(gate, target, control)

    def measure_all(self):
        """
        Performs a single measurement of all qubits in the quantum circuit (big endian).
        
        Returns:
            str: A binary string representing the measured state of all qubits.
        """
        probabilities = np.abs(self.state) ** 2
        result = np.random.choice(range(len(self.state)), p=probabilities)
        measured_state = bin(result)[2:].zfill(self.num_qubits)
        # The measured_state is already in big endian format
        return measured_state

    def measure_all_shots(self, num_shots=1024):
        """
        Performs multiple measurements (shots) of all qubits and returns the counts of each observed state (big endian).
        
        Args:
            num_shots (int, optional): The number of measurements to perform. Defaults to 1024.
        
        Returns:
            dict: A dictionary mapping each observed binary state to its count.
        """
        probabilities = np.abs(self.state) ** 2
        results = np.random.choice(range(len(self.state)), size=num_shots, p=probabilities)
        bitstrings = [bin(result)[2:].zfill(self.num_qubits) for result in results]
        counts = Counter(bitstrings)
        return dict(sorted(counts.items()))
    


# Parse QASM file
def parse_qasm(qasm_string):
    """
    Parses a QASM file and constructs a QuantumCircuit object based on the parsed gates.

    Args:
        qasm_str (str): string of the qasm file

    Returns:
        QuantumCircuit: The constructed quantum circuit with gates applied.
        int: The total number of qubits in the circuit.
        int: The total number of gate operations parsed.
    """
    #find effective number of qubits
    qubits_used = set()
    lines = qasm_string.strip().split('\n')
    gate_lines = lines[4:]#Skip first 4 lines (as they are not gates info)
    for i in gate_lines:
        gates = re.findall(r'\bq\[(\d+)\]', i)#extract the register number
        for qubit in gates:
            qubits_used.add(int(qubit))
    effective_qubits = max(qubits_used) + 1 # find the max register number, +1 because index starts from 0

    #create a circuit
    circuit = QuantumCircuit(effective_qubits)
    
    num_line = 0 #total lines/gates indicated by the file
    # Parse each gate operation and append it to the circuit's gate list
    for line in qasm_string.splitlines():
        num_line+= 1 # Increment the gate operation counter
        line = line.strip()

        if line.startswith('h '): # Check if the line defines a Hadamard gate
            qubit = int(line.split('[')[1].split(']')[0])# Extract the target qubit index
            circuit.gates.append(('h', qubit, None))# Append the gate as a tuple (gate_type, target, control)

        elif line.startswith('cx '):  # CNOT gate
            # Remove parentheses and semicolon, then split the line into parts
            parts = line.replace(')', '').replace('(', '').replace(';', '').split()
            control = int(parts[1].split('[')[1].split(']')[0])  # Extract the control qubit index
            target = int(parts[1].split(',q[')[1].split(']')[0])  # Extract the target qubit index
            circuit.gates.append(('cx', target, control))  # Append as ('cx', target, control)

        elif line.startswith('t '):  # Check if the line defines a T gate
            parts = line.split()
            qubit = int(parts[1].split('[')[1].split(']')[0])
            circuit.gates.append(('t', qubit, None))
        elif line.startswith('tdg '): # Check if the line defines a T-dagger gate
            parts = line.split()
            qubit = int(parts[1].split('[')[1].split(']')[0])
            circuit.gates.append(('tdg', qubit, None))
        elif line.startswith('x'):
            parts = line.split()
            qubit = int(parts[1].split('[')[1].split(']')[0])
            circuit.gates.append(('x', qubit, None))

    return circuit, effective_qubits, num_line # Return the circuit, total qubits, and number of gate operations

def simulate(string): #this is the function for compare_simulators.py to run the tests on benchmarks
    """
    Simulates a quantum circuit defined in a QASM file and returns the final state vector.

    Args:
        string (str): the string converted from qasm file

    Returns:
        np.ndarray: The final state vector of the quantum circuit.
    """

    # Parse the QASM file to get the quantum circuit
    circuit, num_qubits, num_gates = parse_qasm(string)

    # Execute the quantum circuit
    circuit.run()
    
    # Return the final state vector
    return circuit.state