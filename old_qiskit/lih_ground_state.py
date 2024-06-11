from qiskit import BasicAer

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import SLSQP

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock


driver = PySCFDriver(
    # atom="O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0",
    atom="H 0.0 0.0 0.0; Li 0.0 0.0 1.596",
    unit=UnitsType.ANGSTROM,
    charge=0,
    spin=0,
    basis="sto3g",
)
molecule = driver.run()

print("Hartree-Fock energy: {}".format(molecule.hf_energy))
print("Nuclear repulsion energy: {}".format(molecule.nuclear_repulsion_energy))
print("Number of molecular orbitals: {}".format(molecule.num_orbitals))
print("Number of alpha electrons: {}".format(molecule.num_alpha))
print("Number of beta electrons: {}".format(molecule.num_beta))


core = Hamiltonian(
    transformation=TransformationType.FULL,
    qubit_mapping=QubitMappingType.PARITY,
    two_qubit_reduction=True,
    freeze_core=True,
)
qubit_op, aux_ops = core.run(molecule)

print(qubit_op)

init_state = HartreeFock(
    num_orbitals=core._molecule_info["num_orbitals"],
    num_particles=core._molecule_info["num_particles"],
    qubit_mapping=core._qubit_mapping,
    two_qubit_reduction=core._two_qubit_reduction,
)

var_form = UCCSD(
    num_orbitals=core._molecule_info["num_orbitals"],
    num_particles=core._molecule_info["num_particles"],
    qubit_mapping=core._qubit_mapping,
    two_qubit_reduction=core._two_qubit_reduction,
    initial_state=init_state,
)

optimizer = SLSQP(maxiter=2500)

# setup backend on which we will run
backend = BasicAer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(backend=backend)

vqe = VQE(qubit_op, var_form, optimizer)
algo_result = vqe.run(quantum_instance)
result = core.process_algorithm_result(algo_result)

print("Ground state energy: {}".format(result.energy))
print(result)
print("Actual VQE evaluations taken: {}".format(algo_result.optimizer_evals))
