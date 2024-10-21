import numpy as np
import pandas as pd
import networkx as nx
import causallearn
import castle
import os

# Import causallearn metrics
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD
from causallearn.graph.Graph import Graph
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge

from cdt.metrics import SID, SID_CPDAG, get_CPDAG

from clustercausal.clusterdag.ClusterDAG import ClusterDAG
from clustercausal.utils.Utils import *

os.environ[
    "R_HOME"
] = "C:\Program Files\R\R-4.4.1"  # replace with the actual R home directory
import rpy2.robjects as robjects


class Evaluator:
    def __init__(self, truth: Graph, est: Graph):
        """
        Evaluator class for comparing different algorithms
        Use tools from causallearn, gCastle and causal discovery toolbox
        Metrics:
            For now causallearn metrics:
            -arrowsTp/Fp/Fn/Tn: True postive/false positive/false negative/true negative arrows.
            -arrowPrec: Precision for arrows.
            -arrowRec: Recall for arrows.
            -adjTp/Fp/Fn/Tn: True postive/false positive/false negative/true negative edges.
            -adjPrec: Precision for the adjacency matrix.
            -adjRec: Recall for the adjacency matrix.
            -shd: Structural Hamming Distance.

            gCastle metrics:
            -fdr: (reverse + FP) / (TP + FP)
            -tpr: TP/(TP + FN)
            -fpr: (reverse + FP) / (TN + FP)
            -shd: undirected extra + undirected missing + reverse
            -nnz: TP + FP
            -precision: TP/(TP + FP)
            -recall: TP/(TP + FN)
            -F1: 2*(recall*precision)/(recall+precision)
            -gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

            causal discovery toolbox metrics:
            -precision
            -recall
            -structural hamming distance
            -structural intervention distance
        """
        self.truth = truth
        self.est = est
        assert (
            self.truth.get_nodes() == self.est.get_nodes()
        )  # Node lists must be same

    def get_causallearn_metrics(self, sid):
        """
        Name outdated, with cdt sid metric
        Calculate all causallearn metrics
            -adjacency confusion
            -arrow confusion
            -structural hamming distance
            -structural intervention distance from cdt
        Returns:
            -adjacency_confusion: dictionary
            -arrow_confusion: dictionary
            -shd: int
            -sid: (int, int) tuple, lower and upper bounds
        """
        adjacency_confusion = self.get_adjacency_confusion()
        arrow_confusion = self.get_arrow_confusion()
        shd = self.get_shd()
        if sid:
            sid = self.get_sid_bounds()
        else:
            sid = {"sid_lower": None, "sid_lower": None}

        return adjacency_confusion, arrow_confusion, shd, sid

    def get_adjacency_confusion(self):
        """
        Calculate adjacency confusion like in causallearn
        Returns:
            -adjacency_confusion: dictionary with the metrics
            Metrics:
            -adjTp/Fp/Fn/Tn: True postive/false positive/false negative/true negative edges.
            -adjPrec: Precision for the adjacency matrix.
            -adjRec: Recall for the adjacency matrix.
        """
        adj_conf = AdjacencyConfusion(self.truth, self.est)
        adjacency_confusion = {}
        adjacency_confusion["true_positive"] = adj_conf.get_adj_tp()
        adjacency_confusion["false_positive"] = adj_conf.get_adj_fp()
        adjacency_confusion["false_negative"] = adj_conf.get_adj_fn()
        adjacency_confusion["true_negative"] = adj_conf.get_adj_tn()
        adjacency_confusion["precision"] = adj_conf.get_adj_precision()
        adjacency_confusion["recall"] = adj_conf.get_adj_recall()
        adjacency_confusion["f1_score"] = (
            2
            * adjacency_confusion["precision"]
            * adjacency_confusion["recall"]
            / (
                adjacency_confusion["precision"]
                + adjacency_confusion["recall"]
            )
        )
        self.adjacency_confusion = adjacency_confusion
        return adjacency_confusion

    def get_arrow_confusion(self):
        """
        Calculate arrow confusion like in causallearn
        Returns:
            -arrow_confusion: dictionary with the metrics
            (entries with ce positives only get counted in
                    truth_positive (est_positive) if the nodes are
                    adjacent in est (truth))
        """
        arrow_conf = ArrowConfusion(self.truth, self.est)
        arrow_confusion = {}
        arrow_confusion["true_positive"] = arrow_conf.get_arrows_tp()
        arrow_confusion["false_positive"] = arrow_conf.get_arrows_fp()
        arrow_confusion["false_negative"] = arrow_conf.get_arrows_fn()
        arrow_confusion["true_negative"] = arrow_conf.get_arrows_tn()
        arrow_confusion["precision"] = arrow_conf.get_arrows_precision()
        arrow_confusion["recall"] = arrow_conf.get_arrows_recall()
        arrow_confusion["f1_score"] = (
            2
            * arrow_confusion["precision"]
            * arrow_confusion["recall"]
            / (arrow_confusion["precision"] + arrow_confusion["recall"])
        )
        arrow_confusion["true_positive_ce"] = arrow_conf.get_arrows_tp_ce()
        arrow_confusion["false_positive_ce"] = arrow_conf.get_arrows_fp_ce()
        arrow_confusion["false_negative_ce"] = arrow_conf.get_arrows_fn_ce()
        arrow_confusion["true_negative_ce"] = arrow_conf.get_arrows_tn_ce()
        arrow_confusion["precision_ce"] = arrow_conf.get_arrows_precision_ce()
        arrow_confusion["recall_ce"] = arrow_conf.get_arrows_recall_ce()
        arrow_confusion["f1_score_ce"] = (
            2
            * arrow_confusion["precision_ce"]
            * arrow_confusion["recall_ce"]
            / (arrow_confusion["precision_ce"] + arrow_confusion["recall_ce"])
        )
        self.arrow_confusion = arrow_confusion
        return arrow_confusion

    def get_shd(self):
        """
        Calculate structural hamming distance like in causallearn
        This is the number of edge insertions, deletions or flips
        in order to transform one graph to another graph.
        Returns:
            -shd: Structural Hamming Distance.
        """
        shd = SHD(self.truth, self.est)
        self.shd = shd.get_shd()
        return self.shd

    def get_sid_bounds(self):
        """
        Calculate the structural intervention distance bounds
        from the causal discovery toolbox R wrapper
        """
        nx_truth = causallearn_to_nx_adjmat(self.truth.graph)
        nx_est = causallearn_to_nx_adjmat(self.est.graph)
        # sid_lower, sid_upper = SID_CPDAG(nx_truth, get_CPDAG(nx_est))
        sid_lower, sid_upper = SID_CPDAG(nx_truth, nx_est)
        sid = {}
        sid["sid_lower"] = int(sid_lower)
        sid["sid_upper"] = int(sid_upper)
        self.sid = sid
        return self.sid

    @staticmethod
    def get_cluster_pruned_benchmark(cdag: ClusterDAG, cg: CausalGraph):
        """
        Prunes the cg (thought to be from baseline PC)
        with the missing edges of the cluster causal graph
        Used as an improved benchmark to see if a priori cluster knowledge
        makes a difference versus posteriori cluster knowledge
        """
        num_vars = len(cg.G.get_nodes())
        for i in range(num_vars):
            for j in range(num_vars):
                if cg.G.graph[i, j] != 0 and cg.G.graph[j, i] != 0:
                    n_i = ClusterDAG.get_key_by_value(cg.G.node_map, i)
                    n_j = ClusterDAG.get_key_by_value(cg.G.node_map, j)
                    name_i = n_i.get_name()
                    name_j = n_j.get_name()
                    c_i = ClusterDAG.find_key(cdag.cluster_mapping, name_i)
                    c_j = ClusterDAG.find_key(cdag.cluster_mapping, name_j)
                    if c_i != c_j and (
                        (c_i, c_j) not in cdag.cluster_edges
                        and (c_j, c_i) not in cdag.cluster_edges
                    ):
                        edge = cg.G.get_edge(n_i, n_j)
                        cg.G.remove_edge(edge)
        return cg

    @staticmethod
    def get_cluster_connectivity(cdag: ClusterDAG):
        """
        Calculates the percentage of clusters
        that are connected.
        Input: ClusterDAG
        Output: float in [0,1]
        """
        if len(cdag.cluster_edges) > 0:
            return (
                2
                * len(cdag.cluster_edges)
                / (
                    (
                        len(cdag.cluster_mapping)
                        * (len(cdag.cluster_mapping) - 1)
                    )
                )
            )
        else:
            return None
