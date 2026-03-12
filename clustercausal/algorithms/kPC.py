from __future__ import annotations

import time
import warnings
from itertools import combinations
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Node import Node
from causallearn.search.ConstraintBased.FCI import (
    is_arrow_point_allowed,
    ruleR3,
    ruleR4B,
    rulesR1R2cycle,
)
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import CIT, fisherz


def kpc(
    data: ndarray,
    k: int,
    alpha: float = 0.05,
    indep_test=fisherz,
    stable: bool = True,
    uc_rule: int = 0,
    uc_priority: int = 2,
    mvpc: bool = False,
    correction_name: str = "MV_Crtn_Fisher_Z",
    verbose: bool = False,
    background_knowledge: BackgroundKnowledge | None = None,
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs,
) -> CausalGraph:
    """
    Implementation of k-PC based on the provided pseudo-code.

    Steps:
    0) initialize complete circle graph
    1-2) find separating sets with conditioning sets of size <= k and prune
    3) orient unshielded colliders using separating sets
    4) run an FCI-orientation pass (R1/R2/R3/R4B)
    5) apply R11 and R12

    Returns:
            - CausalGraph (same output format as causallearn.pc)
    """
    if data.shape[0] < data.shape[1]:
        warnings.warn(
            "The number of features is much larger than the sample size!"
        )
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if mvpc:
        raise NotImplementedError("kpc does not support mvpc mode")
    if uc_rule != 0:
        warnings.warn(
            "kpc ignores uc_rule and uses the k-PC orientation rules"
        )
    if uc_priority != 2:
        warnings.warn(
            "kpc ignores uc_priority and uses the k-PC orientation rules"
        )
    if correction_name != "MV_Crtn_Fisher_Z":
        warnings.warn("kpc ignores correction_name")

    max_path_length = kwargs.pop("max_path_length", -1)

    start = time.time()
    indep_test_method = CIT(data, indep_test, **kwargs)
    no_of_var = data.shape[1]

    cg = CausalGraph(no_of_var=no_of_var, node_names=node_names)
    cg.set_ind_test(indep_test_method)
    cg.no_of_indep_tests_performed = 0

    _reorient_all_with(cg, Endpoint.CIRCLE)

    sep_sets: Dict[Tuple[int, int], Set[int]] = {}
    _k_limited_skeleton_search(
        cg=cg,
        k=k,
        alpha=alpha,
        sep_sets=sep_sets,
        stable=stable,
        verbose=verbose,
        show_progress=show_progress,
    )

    _orient_unshielded_colliders(
        cg=cg,
        sep_sets=sep_sets,
        background_knowledge=background_knowledge,
        verbose=verbose,
    )

    _fci_orient(
        cg=cg,
        data=data,
        indep_test_method=indep_test_method,
        alpha=alpha,
        sep_sets=sep_sets,
        max_path_length=max_path_length,
        background_knowledge=background_knowledge,
        verbose=verbose,
    )

    _apply_r11_r12(cg=cg, verbose=verbose)

    cg.G.set_pag(True)
    end = time.time()
    cg.PC_elapsed = end - start
    return cg


def _reorient_all_with(cg: CausalGraph, endpoint: Endpoint) -> None:
    edges = cg.G.get_graph_edges()
    for edge in edges:
        cg.G.remove_edge(edge)
        edge.set_endpoint1(endpoint)
        edge.set_endpoint2(endpoint)
        cg.G.add_edge(edge)


def _remove_edge_by_indices(cg: CausalGraph, x: int, y: int) -> None:
    edge = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
    if edge is not None:
        cg.G.remove_edge(edge)


def _k_limited_skeleton_search(
    cg: CausalGraph,
    k: int,
    alpha: float,
    sep_sets: Dict[Tuple[int, int], Set[int]],
    stable: bool,
    verbose: bool,
    show_progress: bool,
) -> None:
    no_of_var = len(cg.G.nodes)

    for depth in range(k + 1):
        edge_removal: List[Tuple[int, int]] = []

        for x in range(no_of_var):
            neigh_x = cg.neighbors(x)
            if len(neigh_x) < depth:
                continue

            for y in neigh_x:
                if y <= x:
                    continue

                if not cg.G.is_adjacent_to(cg.G.nodes[x], cg.G.nodes[y]):
                    continue

                neigh_x_no_y = np.delete(neigh_x, np.where(neigh_x == y))
                if len(neigh_x_no_y) < depth:
                    continue

                found_sep = False
                for S in combinations(neigh_x_no_y, depth):
                    p_value = cg.ci_test(x, y, S)
                    cg.no_of_indep_tests_performed += 1

                    if p_value > alpha:
                        sep_set = set(S)
                        sep_sets[(x, y)] = sep_set
                        sep_sets[(y, x)] = set(sep_set)
                        append_value(cg.sepset, x, y, S)
                        append_value(cg.sepset, y, x, S)

                        if stable:
                            edge_removal.append((x, y))
                            edge_removal.append((y, x))
                        else:
                            _remove_edge_by_indices(cg, x, y)
                            _remove_edge_by_indices(cg, y, x)

                        found_sep = True
                        if verbose:
                            print(
                                f"Depth={depth}: {x} _||_ {y} | {S} (p={p_value:.4g})"
                            )
                        break

                if not found_sep:
                    append_value(cg.sepset, x, y, tuple())
                    append_value(cg.sepset, y, x, tuple())

        if stable:
            for x, y in list(set(edge_removal)):
                _remove_edge_by_indices(cg, x, y)


def _orient_edge_with_endpoints(
    cg: CausalGraph,
    node1: Node,
    node2: Node,
    endpoint1: Endpoint,
    endpoint2: Endpoint,
) -> None:
    edge = cg.G.get_edge(node1, node2)
    if edge is None:
        return
    cg.G.remove_edge(edge)
    cg.G.add_edge(Edge(node1, node2, endpoint1, endpoint2))


def _orient_unshielded_colliders(
    cg: CausalGraph,
    sep_sets: Dict[Tuple[int, int], Set[int]],
    background_knowledge: BackgroundKnowledge | None,
    verbose: bool,
) -> None:
    for node_b in cg.G.get_nodes():
        adj_nodes = cg.G.get_adjacent_nodes(node_b)
        if len(adj_nodes) < 2:
            continue

        for node_a, node_c in combinations(adj_nodes, 2):
            if cg.G.is_adjacent_to(node_a, node_c):
                continue

            a_idx = cg.G.node_map[node_a]
            b_idx = cg.G.node_map[node_b]
            c_idx = cg.G.node_map[node_c]

            sep_set = sep_sets.get((a_idx, c_idx))
            if sep_set is None or b_idx in sep_set:
                continue

            if not is_arrow_point_allowed(
                node_a, node_b, cg.G, background_knowledge
            ):
                continue
            if not is_arrow_point_allowed(
                node_c, node_b, cg.G, background_knowledge
            ):
                continue

            edge_ab = cg.G.get_edge(node_a, node_b)
            edge_cb = cg.G.get_edge(node_c, node_b)
            if edge_ab is None or edge_cb is None:
                continue

            _orient_edge_with_endpoints(
                cg,
                node_a,
                node_b,
                edge_ab.get_proximal_endpoint(node_a),
                Endpoint.ARROW,
            )
            _orient_edge_with_endpoints(
                cg,
                node_c,
                node_b,
                edge_cb.get_proximal_endpoint(node_c),
                Endpoint.ARROW,
            )

            if verbose:
                print(
                    f"Collider: {node_a.get_name()} *-> {node_b.get_name()} <-* {node_c.get_name()}"
                )


def _fci_orient(
    cg: CausalGraph,
    data: ndarray,
    indep_test_method,
    alpha: float,
    sep_sets: Dict[Tuple[int, int], Set[int]],
    max_path_length: int,
    background_knowledge: BackgroundKnowledge | None,
    verbose: bool,
) -> None:
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

        if change_flag or first_time:
            change_flag = ruleR4B(
                cg.G,
                max_path_length,
                data,
                indep_test_method,
                alpha,
                sep_sets,
                change_flag,
                background_knowledge,
                verbose,
            )
            first_time = False


def _apply_r11_r12(cg: CausalGraph, verbose: bool) -> None:
    nodes = cg.G.get_nodes()

    for node_a in nodes:
        incoming_arrow_nodes = cg.G.get_nodes_into(node_a, Endpoint.ARROW)
        if len(incoming_arrow_nodes) > 0:
            continue

        neighbors = cg.G.get_adjacent_nodes(node_a)
        B: List[Node] = []
        C: List[Node] = []
        for node_n in neighbors:
            endpoint_a = cg.G.get_endpoint(node_a, node_n)
            endpoint_n = cg.G.get_endpoint(node_n, node_a)

            if endpoint_a == Endpoint.CIRCLE and endpoint_n == Endpoint.ARROW:
                B.append(node_n)
            elif (
                endpoint_a == Endpoint.CIRCLE and endpoint_n == Endpoint.CIRCLE
            ):
                C.append(node_n)

        B_star = [
            node_b
            for node_b in B
            if all(not cg.G.is_adjacent_to(node_b, node_c) for node_c in C)
        ]
        C_star = [
            node_c
            for node_c in C
            if all(
                (node_c_other == node_c)
                or (not cg.G.is_adjacent_to(node_c, node_c_other))
                for node_c_other in C
            )
        ]

        for node_b in B_star:
            endpoint_a = cg.G.get_endpoint(node_a, node_b)
            endpoint_b = cg.G.get_endpoint(node_b, node_a)
            if endpoint_a == Endpoint.CIRCLE and endpoint_b == Endpoint.ARROW:
                _orient_edge_with_endpoints(
                    cg, node_a, node_b, Endpoint.TAIL, Endpoint.ARROW
                )
                if verbose:
                    print(
                        f"R11: {node_a.get_name()} o-> {node_b.get_name()} becomes {node_a.get_name()} -> {node_b.get_name()}"
                    )

        for node_c in C_star:
            endpoint_a = cg.G.get_endpoint(node_a, node_c)
            endpoint_c = cg.G.get_endpoint(node_c, node_a)
            if endpoint_a == Endpoint.CIRCLE and endpoint_c == Endpoint.CIRCLE:
                _orient_edge_with_endpoints(
                    cg, node_a, node_c, Endpoint.TAIL, Endpoint.TAIL
                )
                if verbose:
                    print(
                        f"R12: {node_a.get_name()} o-o {node_c.get_name()} becomes {node_a.get_name()} -- {node_c.get_name()}"
                    )
