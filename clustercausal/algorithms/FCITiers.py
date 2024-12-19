from __future__ import annotations

import warnings
import time
from tqdm.auto import tqdm
from itertools import combinations, permutations
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray
from functools import wraps

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node

# from causallearn.search.ConstraintBased.FCI import SepsetsPossibleDsep
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.Fas import *
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import *

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


def fci_tiers(
    tiers: List[str],
    cluster_mapping: Dict[str, List[str]],
    dataset: ndarray,
    independence_test_method: str = fisherz,
    alpha: float = 0.05,
    depth: int = -1,
    max_path_length: int = -1,
    verbose: bool = False,
    background_knowledge: BackgroundKnowledge | None = None,
    **kwargs,
) -> Tuple[CausalGraph, List[Edge]]:

    # create empty CausalGraph
    node_names = []
    for tier in tiers:
        node_names.extend(cluster_mapping[tier])

    cg = CausalGraph(no_of_var=len(node_names), node_names=node_names)
    # remove all edges
    for edge in cg.G.get_graph_edges():
        cg.G.remove_edge(edge)

    # TODO add that functionality
    cg.no_of_indep_tests_performed = 0

    n = len(tiers)
    for i in range(n, 0, -1):
        if verbose:
            print(f"Tier {i}")
        pass
        # Create A_i, B_i, O_i
        A_i = tiers[: i - 1]
        B_i = tiers[i - 1]
        # do FCI exogenous
        cg = fci_exogenous(
            cg,
            cluster_mapping,
            A_i,
            B_i,
            dataset,
            independence_test_method,
            alpha,
            depth,
            max_path_length,
            verbose,
            background_knowledge,
            **kwargs,
        )

    cg.G.set_pag(True)
    edges = get_color_edges(cg.G)

    return cg, edges


def fci_exogenous(
    cg: CausalGraph,
    cluster_mapping: Dict[str, List[str]],
    A_i: List[str],
    B_i: str,
    dataset: ndarray,
    independence_test_method: str = fisherz,
    alpha: float = 0.05,
    depth: int = -1,
    max_path_length: int = -1,
    verbose: bool = False,
    background_knowledge: BackgroundKnowledge | None = None,
    **kwargs,
) -> Tuple[CausalGraph, List[Edge]]:
    """
    Copied from causallearn and adapted to fci_exogenous
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn(
            "The number of features is much larger than the sample size!"
        )

    independence_test_method = CIT(
        dataset, method=independence_test_method, **kwargs
    )

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if (background_knowledge is not None) and type(
        background_knowledge
    ) != BackgroundKnowledge:
        raise TypeError(
            "'background_knowledge' must be 'BackgroundKnowledge' type!"
        )
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------

    nodes = []
    for i in range(dataset.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    # Adaptation: create CausalGraph for FCIExogenous
    A_i_node_names = []
    for tier in A_i:
        A_i_node_names.extend(cluster_mapping[tier])
    B_i_node_names = cluster_mapping[B_i]
    A_i_nodes = []
    B_i_nodes = []
    for node_name in A_i_node_names:
        A_i_nodes.append(ClusterDAG.get_node_by_name(node_name, cg=cg))
    for node_name in B_i_node_names:
        B_i_nodes.append(ClusterDAG.get_node_by_name(node_name, cg=cg))

    # Add edges A_i -> B_i and B_i o-o B_i to cg
    for node_a in A_i_nodes:
        for node_b in B_i_nodes:
            edge = cg.G.get_edge(node_a, node_b)
            # edge should be none
            if edge is not None:
                raise ValueError("Edge already exists")
            # create edge a -> b
            new_edge = Edge(node_a, node_b, Endpoint.TAIL, Endpoint.ARROW)
            cg.G.add_edge(new_edge)

    for node_b1, node_b2 in combinations(B_i_nodes, 2):
        edge = cg.G.get_edge(node_b1, node_b2)
        # edge should be none
        if edge is not None:
            raise ValueError("Edge already exists")
        # create edge b1 o-o b2
        new_edge = Edge(node_b1, node_b2, Endpoint.CIRCLE, Endpoint.CIRCLE)
        cg.G.add_edge(new_edge)

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    cg, sep_sets = fas_exogenous(
        cg,
        dataset,
        nodes,
        independence_test_method=independence_test_method,
        alpha=alpha,
        knowledge=background_knowledge,
        depth=depth,
        verbose=verbose,
    )
    # print(sep_sets)
    reorientAllWith(cg.G, Endpoint.CIRCLE)

    rule0(cg.G, nodes, sep_sets, background_knowledge, verbose)

    removeByPossibleDsep(cg.G, independence_test_method, alpha, sep_sets)

    reorientAllWith(cg.G, Endpoint.CIRCLE)

    rule0(cg.G, nodes, sep_sets, background_knowledge, verbose)
    # print(sep_sets)
    change_flag = True
    first_time = True

    while change_flag:
        change_flag = False
        change_flag = rulesR1R2cycle(
            cg.G, background_knowledge, change_flag, verbose
        )
        change_flag = ruleR3(
            cg.G, sep_sets, background_knowledge, change_flag, verbose
        )

        if change_flag or (
            first_time
            and background_knowledge is not None
            and len(background_knowledge.forbidden_rules_specs) > 0
            and len(background_knowledge.required_rules_specs) > 0
            and len(background_knowledge.tier_map.keys()) > 0
        ):
            change_flag = ruleR4B(
                cg.G,
                max_path_length,
                dataset,
                independence_test_method,
                alpha,
                sep_sets,
                change_flag,
                background_knowledge,
                verbose,
            )

            first_time = False

            if verbose:
                print("Epoch")

    return cg


def fas_exogenous(
    cg: CausalGraph,
    data: ndarray,
    nodes: List[Node],
    independence_test_method: CIT | None = None,
    alpha: float = 0.05,
    knowledge: BackgroundKnowledge | None = None,
    depth: int = -1,
    verbose: bool = False,
    stable: bool = True,
    show_progress: bool = True,
) -> Tuple[CausalGraph, Dict[Tuple[int, int], Set[int]]]:
    """
    Copied from causallearn and adapted to fci_exogenous
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    node_names = [node.get_name() for node in nodes]
    cg = cg
    cg.set_ind_test(independence_test_method)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f"Depth={depth}, working on node {x}")
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if knowledge is not None and (
                    knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                    and knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])
                ):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        sep_sets[(x, y)] = set()
                        sep_sets[(y, x)] = set()
                        break
                    else:
                        edge_removal.append(
                            (x, y)
                        )  # after all conditioning sets at
                        edge_removal.append(
                            (y, x)
                        )  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print(
                                "%d ind %d | %s with p-value %f\n"
                                % (x, y, S, p)
                            )
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            sep_sets[(x, y)] = set(S)
                            sep_sets[(y, x)] = set(S)
                            break
                        else:
                            edge_removal.append(
                                (x, y)
                            )  # after all conditioning sets at
                            edge_removal.append(
                                (y, x)
                            )  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                    else:
                        if verbose:
                            print(
                                "%d dep %d | %s with p-value %f\n"
                                % (x, y, S, p)
                            )
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for x, y in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)
            if cg.sepset[x, y] is not None:
                origin_list = []
                for l_out in cg.sepset[x, y]:
                    for l_in in l_out:
                        origin_list.append(l_in)
                sep_sets[(x, y)] = set(origin_list)
                sep_sets[(y, x)] = set(origin_list)

    # for x in range(no_of_var):
    #     for y in range(x, no_of_var):
    #         if cg.sepset[x, y] is not None:
    #             origin_list = []
    #             for l_out in cg.sepset[x, y]:
    #                 for l_in in l_out:
    #                     origin_list.append(l_in)
    #             sep_sets[(x, y)] = set(origin_list)

    if show_progress:
        pbar.close()

    return cg, sep_sets
