import numpy as np
import causallearn

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GeneralGraph import GeneralGraph

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


def test_cdag_initialization():
    """
    TODO based on noteboks, write a testcase generating a couple clusterings and verifying
    CDAG class produces correct outputs
    !!! make a test function for each method of CDAG
    """
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


def test_get_node_indices_of_cluster():
    # TODO
    pass


def test_get_node_names_from_list():
    # TODO
    pass


def test_get_key_by_value():
    # TODO
    pass
