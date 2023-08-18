import numpy as np
import causallearn
import networkx as nx

from causallearn.search.ConstraintBased.PC import pc
from clustercausal.clusterdag.ClusterDAG import ClusterDAG
from clustercausal.algorithms.ClusterPC import ClusterPC
from clustercausal.experiments.Simulator import Simulator
from clustercausal.experiments.Evaluator import Evaluator


def test_clust_pc_correct_output():
    """
    TODO based on noteboks, write a testcase generating and checking a couple cpc runs
    !!! make a test function for each method of CPC
    """
    pass


def test_clust_pc_to_base_pc():
    """
    TODO based on noteboks, write a testcase generating and checking a couple cpc runs
    against the baseline pc algorithm
    """
    simulation = Simulator(
        n_nodes=17, n_edges=28, n_clusters=1, seed=12343, sample_size=100
    )
    cluster_dag = simulation.run()
    cluster_pc = ClusterPC(
        cdag=cluster_dag,
        data=cluster_dag.data,
        alpha=0.05,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    est_graph = cluster_pc.run()
    evaluation = Evaluator(truth=cluster_dag.true_dag.G, est=est_graph.G)
    (
        adjacency_confusion,
        arrow_confusion,
        shd,
        sid,
    ) = evaluation.get_causallearn_metrics()

    causallearn_cg = pc(
        cluster_dag.data, alpha=0.05, verbose=False, show_progress=False
    )
    evaluation_causallearn = Evaluator(
        truth=cluster_dag.true_dag.G, est=causallearn_cg.G
    )
    (
        cl_adjacency_confusion,
        cl_arrow_confusion,
        cl_shd,
        cl_si,
    ) = evaluation_causallearn.get_causallearn_metrics()
    assert adjacency_confusion == cl_adjacency_confusion
    assert arrow_confusion == cl_arrow_confusion
    assert shd == cl_shd
    assert sid == cl_si
    # Check isomorphism
    est_graph.to_nx_graph()
    causallearn_cg.to_nx_graph()
    assert nx.is_isomorphic(cluster_dag.cg.nx_graph, causallearn_cg.nx_graph)

    simulation = Simulator(
        n_nodes=13, n_edges=21, n_clusters=1, seed=123, sample_size=100
    )
    cluster_dag = simulation.run()
    cluster_pc = ClusterPC(
        cdag=cluster_dag,
        data=cluster_dag.data,
        alpha=0.05,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    est_graph = cluster_pc.run()
    evaluation = Evaluator(truth=cluster_dag.true_dag.G, est=est_graph.G)
    (
        adjacency_confusion,
        arrow_confusion,
        shd,
        sid,
    ) = evaluation.get_causallearn_metrics()

    causallearn_cg = pc(
        cluster_dag.data, alpha=0.05, verbose=False, show_progress=False
    )
    evaluation_causallearn = Evaluator(
        truth=cluster_dag.true_dag.G, est=causallearn_cg.G
    )
    (
        cl_adjacency_confusion,
        cl_arrow_confusion,
        cl_shd,
        cl_sid,
    ) = evaluation_causallearn.get_causallearn_metrics()
    assert adjacency_confusion == cl_adjacency_confusion
    assert arrow_confusion == cl_arrow_confusion
    assert shd == cl_shd
    assert sid == cl_sid
    # Check isomorphism
    est_graph.to_nx_graph()
    causallearn_cg.to_nx_graph()
    assert nx.is_isomorphic(cluster_dag.cg.nx_graph, causallearn_cg.nx_graph)

    simulation = Simulator(
        n_nodes=8, n_edges=18, n_clusters=1, seed=12443, sample_size=100
    )
    cluster_dag = simulation.run()
    cluster_pc = ClusterPC(
        cdag=cluster_dag,
        data=cluster_dag.data,
        alpha=0.05,
        indep_test="fisherz",
        verbose=False,
        show_progress=False,
    )
    est_graph = cluster_pc.run()
    evaluation = Evaluator(truth=cluster_dag.true_dag.G, est=est_graph.G)
    (
        adjacency_confusion,
        arrow_confusion,
        shd,
        sid,
    ) = evaluation.get_causallearn_metrics()

    causallearn_cg = pc(
        cluster_dag.data, alpha=0.05, verbose=False, show_progress=False
    )
    evaluation_causallearn = Evaluator(
        truth=cluster_dag.true_dag.G, est=causallearn_cg.G
    )
    (
        cl_adjacency_confusion,
        cl_arrow_confusion,
        cl_shd,
        cl_sid,
    ) = evaluation_causallearn.get_causallearn_metrics()
    assert adjacency_confusion == cl_adjacency_confusion
    assert arrow_confusion == cl_arrow_confusion
    assert shd == cl_shd
    assert sid == cl_sid
    # Check isomorphism
    est_graph.to_nx_graph()
    causallearn_cg.to_nx_graph()
    assert nx.is_isomorphic(cluster_dag.cg.nx_graph, causallearn_cg.nx_graph)


def test_clust_pc_initialization():
    pass


def test_clust_pc_intra_phase():
    pass


def test_clust_pc_inter_phase():
    pass


def test_clust_pc_edge_orientation():
    pass
