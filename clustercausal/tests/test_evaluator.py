from clustercausal.experiments.Evaluator import Evaluator
from clustercausal.clusterdag.ClusterDAG import ClusterDAG
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint


def test_get_cluster_pruned_benchmark():
    cluster_mapping = {"C1": ["X1", "X2"], "C2": ["X3"], "C3": ["X4"]}
    cluster_edges = [("C1", "C3"), ("C2", "C3")]
    cdag = ClusterDAG(
        cluster_mapping=cluster_mapping, cluster_edges=cluster_edges
    )
    cdag.cdag_to_mpdag()
    n_2 = ClusterDAG.get_node_by_name("X2", cdag.cg)
    n_3 = ClusterDAG.get_node_by_name("X3", cdag.cg)
    edge = Edge(n_2, n_3, Endpoint.TAIL, Endpoint.ARROW)
    wrong_cg = cdag.cg
    wrong_cg.G.add_edge(edge)
    pruned_cg = Evaluator.get_cluster_pruned_benchmark(cdag=cdag, cg=wrong_cg)
    assert pruned_cg == cdag.cg
