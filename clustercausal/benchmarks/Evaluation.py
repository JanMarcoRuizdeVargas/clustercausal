import numpy as np
import pandas as pd
import networkx as nx
import causallearn
import castle

# Import causallearn metrics

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class Evaluator:
    def __init__(self):
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
        pass

    def true_positive_rate():
        pass
