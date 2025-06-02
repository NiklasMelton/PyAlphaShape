import pytest
from pyalphashape.GraphClosure import GraphClosureTracker


def test_initial_components():
    gct = GraphClosureTracker(5)
    assert len(gct) == 5
    for i in range(5):
        assert gct.find(i) == i
        assert gct.is_connected(i, i)


def test_union_merges_components():
    gct = GraphClosureTracker(4)
    gct.union(0, 1)
    assert gct.is_connected(0, 1)
    assert len(gct) == 3
    assert set(gct.find(i) for i in [0, 1]) == {gct.find(0)}


def test_add_edge_equivalent_to_union():
    gct = GraphClosureTracker(3)
    gct.add_edge(0, 2)
    assert gct.is_connected(0, 2)
    assert len(gct) == 2


def test_add_fully_connected_subgraph():
    gct = GraphClosureTracker(6)
    gct.add_fully_connected_subgraph([1, 2, 3])
    assert gct.subgraph_is_already_connected([1, 2, 3])
    assert len(gct) == 4  # number of independent components


def test_subgraph_is_already_connected_true_and_false():
    gct = GraphClosureTracker(4)
    gct.add_edge(0, 1)
    assert gct.subgraph_is_already_connected([0, 1]) is True
    assert gct.subgraph_is_already_connected([0, 2]) is False
    assert gct.subgraph_is_already_connected([]) is True


def test_dynamic_resize_via_find_and_union():
    gct = GraphClosureTracker(2)
    gct.union(0, 5)  # triggers dynamic resize
    assert gct.is_connected(0, 5)
    assert len(gct) == 5  # nodes 1–4 remain singleton


def test_components_iteration_and_indexing():
    gct = GraphClosureTracker(4)
    gct.union(0, 1)
    gct.union(2, 3)
    comps = list(gct)
    assert isinstance(comps[0], set)
    assert sorted(len(comp) for comp in comps) == [2, 2]
    # Indexing
    flat = gct[0] | gct[1]
    assert set(flat) == {0, 1, 2, 3}


def test_len_reflects_component_count():
    gct = GraphClosureTracker(6)
    assert len(gct) == 6
    gct.add_edge(0, 1)
    gct.add_edge(1, 2)
    gct.add_edge(3, 4)
    assert len(gct) == 3
    gct.add_edge(2, 3)
    assert len(gct) == 2
