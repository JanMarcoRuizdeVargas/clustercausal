import numpy as np
import causallearn
from itertools import combinations

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GeneralGraph import GeneralGraph

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


def test_cdag_initialization():
    """ """
    debug_nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    debug_edges = [
        ("0", "1"),
        ("2", "1"),
        ("2", "7"),
        ("1", "6"),
        ("0", "5"),
        ("3", "5"),
        ("4", "6"),
        ("3", "6"),
        ("5", "6"),
        ("6", "7"),
    ]
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z")]
    debug_no_clust_mapping = {"A": ["0", "1", "2", "3", "4", "5", "6", "7"]}
    debug_no_clust_edges = []
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    assert cdag.cluster_mapping == debug_cluster_mapping
    assert cdag.cluster_edges == debug_cluster_edges
    assert len(cdag.node_names) == len(debug_nodes)
    assert isinstance(cdag.cluster_graph, CausalGraph)
    assert isinstance(cdag.cluster_graph.G, GeneralGraph)

    assert no_clust_cdag.cluster_mapping == debug_no_clust_mapping
    assert no_clust_cdag.cluster_edges == debug_no_clust_edges
    assert len(no_clust_cdag.node_names) == len(debug_nodes)
    assert isinstance(no_clust_cdag.cluster_graph, CausalGraph)
    assert isinstance(no_clust_cdag.cluster_graph.G, GeneralGraph)


def test_cdag_to_mpdag():
    debug_nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    debug_edges = [
        ("0", "1"),
        ("2", "1"),
        ("2", "7"),
        ("1", "6"),
        ("0", "5"),
        ("3", "5"),
        ("4", "6"),
        ("3", "6"),
        ("5", "6"),
        ("6", "7"),
    ]
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z")]
    debug_no_clust_mapping = {"A": ["0", "1", "2", "3", "4", "5", "6", "7"]}
    debug_no_clust_edges = []
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    cdag.cdag_to_mpdag()
    no_clust_cdag.cdag_to_mpdag()

    assert isinstance(cdag.cg, CausalGraph)
    assert isinstance(cdag.cg.G, GeneralGraph)
    assert isinstance(no_clust_cdag.cg, CausalGraph)
    assert isinstance(no_clust_cdag.cg.G, GeneralGraph)


def test_cdag_to_circle_mpdag():
    cdag = ClusterDAG(
        cluster_mapping={
            "C1": ["X1", "X2"],
            "C2": ["X3", "X4"],
            "C3": ["X5", "X6"],
        },
        cluster_edges=[("C1", "C2"), ("C2", "C3")],
        cluster_bidirected_edges=[("C2", "C3")],
    )
    cdag.cdag_to_circle_mpdag()

    assert isinstance(cdag.cg, CausalGraph)
    assert isinstance(cdag.cg.G, GeneralGraph)
    assert cdag.bidir_paths == {
        "C1": [["C1"]],
        "C2": [["C2"], ["C2", "C3"]],
        "C3": [["C3"], ["C3", "C2"]],
    }
    assert cdag.collider_paths == {
        "C1": [["C1"], ["C1", "C2"], ["C1", "C2", "C3"]],
        "C2": [["C2"], ["C2", "C3"]],
        "C3": [["C3"], ["C3", "C2"]],
    }

    cdag = ClusterDAG(
        cluster_mapping={
            "C1": ["X1", "X2"],
            "C2": ["X3", "X4"],
            "C3": ["X5", "X6"],
            "C4": ["X7", "X8"],
        },
        cluster_edges=[("C1", "C2"), ("C2", "C3"), ("C1", "C4")],
        cluster_bidirected_edges=[("C2", "C3"), ("C3", "C4")],
    )
    cdag.cdag_to_circle_mpdag()

    assert isinstance(cdag.cg, CausalGraph)
    assert isinstance(cdag.cg.G, GeneralGraph)
    assert cdag.bidir_paths == {
        "C1": [["C1"]],
        "C2": [["C2"], ["C2", "C3"], ["C2", "C3", "C4"]],
        "C3": [["C3"], ["C3", "C2"], ["C3", "C4"]],
        "C4": [["C4"], ["C4", "C3"], ["C4", "C3", "C2"]],
    }
    assert cdag.collider_paths == {
        "C1": [
            ["C1"],
            ["C1", "C2"],
            ["C1", "C2", "C3"],
            ["C1", "C2", "C3", "C4"],
            ["C1", "C4"],
            ["C1", "C4", "C3"],
            ["C1", "C4", "C3", "C2"],
        ],
        "C2": [
            ["C2"],
            ["C2", "C3"],
            ["C2", "C3", "C4"],
            ["C2", "C3", "C4", "C1"],
        ],
        "C3": [["C3"], ["C3", "C2"], ["C3", "C4"]],
        "C4": [["C4"], ["C4", "C3"], ["C4", "C3", "C2"]],
    }


def test_draw_mpdag():
    pass


def test_draw_cluster_graph():
    pass


def test_cluster_topological_ordering():
    debug_nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    debug_edges = [
        ("0", "1"),
        ("2", "1"),
        ("2", "7"),
        ("1", "6"),
        ("0", "5"),
        ("3", "5"),
        ("4", "6"),
        ("3", "6"),
        ("5", "6"),
        ("6", "7"),
    ]
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z")]
    debug_no_clust_mapping = {"A": ["0", "1", "2", "3", "4", "5", "6", "7"]}
    debug_no_clust_edges = []
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    assert cdag.get_cluster_topological_ordering() == ["X", "Y", "Z"]
    assert no_clust_cdag.get_cluster_topological_ordering() == ["A"]


def test_get_parents_plus_get_local_graph_and_get_List_of_nodes_by_name():
    debug_nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    debug_edges = [
        ("0", "1"),
        ("2", "1"),
        ("2", "7"),
        ("1", "6"),
        ("0", "5"),
        ("3", "5"),
        ("4", "6"),
        ("3", "6"),
        ("5", "6"),
        ("6", "7"),
    ]
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z")]
    debug_no_clust_mapping = {"A": ["0", "1", "2", "3", "4", "5", "6", "7"]}
    debug_no_clust_edges = []
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    cdag.cdag_to_mpdag()
    no_clust_cdag.cdag_to_mpdag()

    X = cdag.get_node_by_name("X", cg=cdag.cluster_graph)
    X_node_names = cdag.cluster_mapping["X"]
    X_nodes = cdag.get_list_of_nodes_by_name(X_node_names, cdag.cg)
    parents_plus = cdag.get_parents_plus(X)
    assert parents_plus == ([X], X_nodes)
    assert X_node_names == ["0", "1", "2"]

    Z = cdag.get_node_by_name("Z", cg=cdag.cluster_graph)
    Z_node_names = cdag.cluster_mapping["Z"]
    Z_nodes = cdag.get_list_of_nodes_by_name(Z_node_names, cdag.cg)
    parents_plus = cdag.get_parents_plus(Z)

    Y = cdag.get_node_by_name("Y", cg=cdag.cluster_graph)
    Y_node_names = cdag.cluster_mapping["Y"]
    Y_nodes = cdag.get_list_of_nodes_by_name(Y_node_names, cdag.cg)

    assert Y.get_name() == "Y"
    # Careful, order of nodes is important
    assert parents_plus == ([Z, X, Y], Z_nodes + X_nodes + Y_nodes)


def test_get_local_graph_and_subgraph():
    # TODO
    pass


def test_max_degree_of_cluster_and_max_nonchild_degree_of_cluster_and_max_degree_of_cluster_parents():
    debug_nodes = ["0", "1", "2", "3", "4", "5", "6", "7"]
    debug_edges = [
        ("0", "1"),
        ("2", "1"),
        ("2", "7"),
        ("1", "6"),
        ("0", "5"),
        ("3", "5"),
        ("4", "6"),
        ("3", "6"),
        ("5", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "9"),
    ]
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
        "S": ["8", "9"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z"), ("Z", "S")]
    debug_no_clust_mapping = {
        "A": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    }
    debug_no_clust_edges = []
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    cdag.cdag_to_mpdag()
    no_clust_cdag.cdag_to_mpdag()
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    no_clust_cdag = ClusterDAG(debug_no_clust_mapping, debug_no_clust_edges)
    cdag.cdag_to_mpdag()
    no_clust_cdag.cdag_to_mpdag()
    X_cluster = cdag.get_node_by_name("X", cg=cdag.cluster_graph)
    Y_cluster = cdag.get_node_by_name("Y", cg=cdag.cluster_graph)
    Z_cluster = cdag.get_node_by_name("Z", cg=cdag.cluster_graph)
    S_cluster = cdag.get_node_by_name("S", cg=cdag.cluster_graph)
    assert cdag.max_degree_of_cluster(X_cluster) == 5
    assert cdag.max_nonchild_degree_of_cluster(X_cluster) == 2
    assert cdag.max_degree_of_cluster(Y_cluster) == 4
    assert cdag.max_nonchild_degree_of_cluster(Y_cluster) == 1
    assert cdag.max_degree_of_cluster(Z_cluster) == 9
    assert cdag.max_nonchild_degree_of_cluster(Z_cluster) == 7
    assert cdag.max_degree_of_cluster(S_cluster) == 4
    assert cdag.max_nonchild_degree_of_cluster(S_cluster) == 4
    assert cdag.max_degree_of_cluster_parents(X_cluster) == 0
    assert cdag.max_degree_of_cluster_parents(Y_cluster) == 0
    assert cdag.max_degree_of_cluster_parents(Z_cluster) == 5
    assert cdag.max_degree_of_cluster_parents(S_cluster) == 9


def test_get_cluster_connectedness():
    debug_cluster_mapping = {
        "X": ["0", "1", "2"],
        "Y": ["3", "4"],
        "Z": ["5", "6", "7"],
        "S": ["8", "9"],
    }
    debug_cluster_edges = [("X", "Z"), ("Y", "Z"), ("Z", "S")]
    cdag = ClusterDAG(debug_cluster_mapping, debug_cluster_edges)
    cdag.cdag_to_mpdag()
    cdag.true_dag = cdag.cg
    assert cdag.get_cluster_connectedness() == (1.0, 1.0, 0.5)
    cdag.get_cluster_connectedness() == (1.0, 1.0, 0.5)
    n0 = cdag.get_node_by_name("0", cg=cdag.true_dag)
    n1 = cdag.get_node_by_name("1", cg=cdag.true_dag)
    n3 = cdag.get_node_by_name("3", cg=cdag.true_dag)
    n4 = cdag.get_node_by_name("4", cg=cdag.true_dag)
    n6 = cdag.get_node_by_name("6", cg=cdag.true_dag)
    n9 = cdag.get_node_by_name("9", cg=cdag.true_dag)
    for nx, ny in combinations([n0, n1, n3, n4, n6, n9], 2):
        edge = cdag.true_dag.G.get_edge(nx, ny)
        if edge is not None:
            cdag.true_dag.G.remove_edge(edge)
    cdag.get_cluster_connectedness() == (
        0.6666666666666666,
        0.7592592592592592,
        0.3796296296296296,
    )


def test_get_node_indices_of_cluster():
    # TODO
    pass


def test_get_node_names_from_list():
    # TODO
    pass


def test_get_key_by_value():
    # TODO
    pass
