import pytest
from qibo import Circuit, gates

from qibolab.transpilers.blocks import (
    Block,
    BlockingError,
    _find_previous_gates,
    _find_successive_gates,
    block_decomposition,
    commute,
    count_2q_gates,
    fuse_blocks,
    gates_on_qubit,
    initial_block_decomposition,
    remove_gates,
)


def test_count_2q_gates():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1), gates.CZ(0, 1), gates.H(0)])
    assert block.count_2q_gates() == 2


def test_rename():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block.rename("renamed_block")
    assert block.name == "renamed_block"


def test_add_gate_and_entanglement():
    block = Block(qubits=(0, 1), gates=[gates.H(0)])
    assert block.entangled == False
    block.add_gate(gates.CZ(0, 1))
    assert block.entangled == True
    assert block.count_2q_gates() == 1


def test_add_gate_error():
    block = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    with pytest.raises(BlockingError):
        block.add_gate(gates.CZ(0, 2))


def test_fuse_blocks():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(0, 1), gates=[gates.H(0)])
    fused = fuse_blocks(block_1, block_2)
    assert fused.qubits == (0, 1)
    assert fused.entangled == True
    assert fused.count_2q_gates() == 1


def test_fuse_blocks_error():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(1, 2), gates=[gates.CZ(1, 2)])
    with pytest.raises(BlockingError):
        fused = fuse_blocks(block_1, block_2)


@pytest.mark.parametrize("qubits", [(0, 1), (2, 1)])
def test_commute_false(qubits):
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=qubits, gates=[gates.CZ(*qubits)])
    assert commute(block_1, block_2) == False


def test_commute_true():
    block_1 = Block(qubits=(0, 1), gates=[gates.CZ(0, 1)])
    block_2 = Block(qubits=(2, 3), gates=[gates.CZ(2, 3)])
    assert commute(block_1, block_2) == True


def test_gates_on_qubit():
    gatelist = [gates.H(0), gates.H(1), gates.H(2), gates.H(0)]
    assert len(gates_on_qubit(gatelist, 0)) == 2
    assert len(gates_on_qubit(gatelist, 1)) == 1


def test_remove_gates():
    gatelist = [gates.H(0), gates.CZ(0, 1), gates.H(2), gates.CZ(0, 2)]
    delete_list = [gatelist[0], gatelist[3]]
    remove_gates(gatelist, delete_list)
    assert len(gatelist) == 2
    assert count_2q_gates(gatelist) == 1


def test_find_previous_gates():
    gatelist = [gates.H(0), gates.H(1), gates.H(2)]
    previous_gates = _find_previous_gates(gatelist, (0, 1))
    assert len(previous_gates) == 2


def test_find_successive_gates():
    gatelist = [gates.H(0), gates.CZ(2, 3), gates.H(1), gates.H(2), gates.CZ(2, 1)]
    previous_gates = _find_successive_gates(gatelist, (0, 1))
    assert len(previous_gates) == 2
    assert count_2q_gates(previous_gates) == 0


def test_initial_block_decomposition():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.H(3))
    blocks = initial_block_decomposition(circ)
    assert len(blocks) == 4
    assert len(blocks[0].gates) == 3
    assert len(blocks[1].gates) == 1
    assert blocks[2].entangled == True
    assert blocks[3].entangled == False


def test_block_decomposition_error():
    circ = Circuit(1)
    with pytest.raises(BlockingError):
        block_decomposition(circ)


def test_block_decomposition_no_fuse():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.H(1))
    circ.add(gates.H(3))
    blocks = block_decomposition(circ, fuse=False)
    assert len(blocks) == 4
    assert len(blocks[0].gates) == 4
    assert len(blocks[1].gates) == 1
    assert blocks[2].entangled == True
    assert blocks[3].entangled == False


def test_block_decomposition():
    circ = Circuit(4)
    circ.add(gates.H(1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.CZ(0, 1))  # first block
    circ.add(gates.H(0))  # first block
    circ.add(gates.CZ(0, 1))  # first block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.CZ(1, 2))  # second block
    circ.add(gates.H(1))  # second block
    circ.add(gates.H(3))  # 4 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(0, 1))  # 3 block
    circ.add(gates.CZ(2, 3))  # 4 block
    circ.add(gates.CZ(0, 1))  # 3 block
    blocks = block_decomposition(circ)
    assert len(blocks) == 4
    assert blocks[0].count_2q_gates() == 2
    assert len(blocks[0].gates) == 5
    assert blocks[0].qubits == (0, 1)
    assert blocks[1].count_2q_gates() == 2
    assert len(blocks[1].gates) == 3
    assert blocks[3].count_2q_gates() == 1
    assert len(blocks[3].gates) == 2
    assert blocks[3].qubits == (2, 3)
    assert blocks[2].count_2q_gates() == 3
    assert len(blocks[2].gates) == 3
