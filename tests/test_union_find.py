import pytest
from headss2.union_find import UnionFind

def test_union_find_initialization():
    uf = UnionFind()
    assert uf.parent == {}

def test_find_sets_its_own_parent_if_not_present():
    uf = UnionFind()
    assert uf.find(1) == 1
    assert uf.parent == {1: 1}

def test_union_merges_two_clusters():
    uf = UnionFind()
    uf.union(1, 2)
    # After union, both should point to the same root
    root1 = uf.find(1)
    root2 = uf.find(2)
    assert root1 == root2

def test_union_chain():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(3, 4)
    root1 = uf.find(1)
    root4 = uf.find(4)
    assert root1 == root4

def test_path_compression():
    uf = UnionFind()
    # Create a deep tree first
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(3, 4)

    # Calling find should compress the path
    root = uf.find(1)
    assert root == uf.find(4)
    # All nodes should now point directly to the root
    for i in range(1, 5):
        assert uf.parent[i] == root
