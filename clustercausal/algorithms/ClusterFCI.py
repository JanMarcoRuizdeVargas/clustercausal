from __future__ import annotations

import warnings
import time
from tqdm.auto import tqdm
from itertools import combinations, permutations
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
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
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import *

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class ClusterFCI:
    def __init__(
        self,
        cdag: ClusterDAG,
        dataset: ndarray,
        independence_test_method: str = fisherz,
        stable: bool = True,
        alpha: float = 0.05,
        depth: int = -1,
        max_path_length: int = -1,
        verbose: bool = False,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        **kwargs,
    ):
        """
        Set parameters for the clust PC algorithm.
        """
        self.cdag = cdag
        self.dataset = dataset
        self.stable = stable
        self.alpha = alpha
        self.depth = depth
        self.max_path_length = max_path_length
        self.verbose = verbose
        self.background_knowledge = background_knowledge
        if background_knowledge is not None:
            raise ValueError("Background knowledge is not supported for now")
        self.show_progress = show_progress
        self.kwargs = kwargs
        # First bidirected edges are ignored, it will be updated later
        # self.cdag.cdag_to_mpdag()
        self.cdag.cdag_to_circle_mpdag()
        self.cdag.get_cluster_topological_ordering()  # Get topological ordering of CDAG

        # Set independence test for cluster_phase
        self.cdag.cg.test = CIT(
            self.dataset, independence_test_method, **kwargs
        )
        # Set independence test for run
        self.independence_test_method = CIT(
            self.dataset, independence_test_method, **kwargs
        )

    def run(self):
        """
        Runs the C-FCI algorithm.
        Warning: Its current version does not include the
        orientation of untested triples.
        """
        start = time.time()
        no_of_var = self.dataset.shape[1]
        self.no_of_indep_tests_performed = 0
        assert len(self.cdag.node_names) == no_of_var
        if self.verbose:
            print(
                f"Topological ordering {(self.cdag.cdag_list_of_topological_sort)}"
            )
        # self.cdag.cg.sepset = set()
        self.sep_sets: Dict[Tuple[int, int], Set[int]] = {}

        # As some nodes have no edge by CDAG definition, they never get tested so have Nonetype sepsets
        # manually have to add the parent set of i and parent set of j to sepset(i, j) and sepset(j, i)
        for i in range(no_of_var):
            for j in range(no_of_var):
                node_i = self.cdag.get_key_by_value(self.cdag.cg.G.node_map, i)
                node_j = self.cdag.get_key_by_value(self.cdag.cg.G.node_map, j)
                edge = self.cdag.cg.G.get_edge(node_i, node_j)
                if edge is None:
                    parents_i = self.cdag.cg.G.get_parents(node_i)
                    index_parents_i = [
                        self.cdag.cg.G.node_map[parent] for parent in parents_i
                    ]
                    parents_j = self.cdag.cg.G.get_parents(node_j)
                    index_parents_j = [
                        self.cdag.cg.G.node_map[parent] for parent in parents_j
                    ]
                    append_value(
                        self.cdag.cg.sepset, i, j, tuple(index_parents_i)
                    )
                    append_value(
                        self.cdag.cg.sepset, i, j, tuple(index_parents_j)
                    )
                    self.sep_sets[(i, j)] = set(index_parents_i).union(
                        index_parents_j
                    )
                    append_value(
                        self.cdag.cg.sepset, j, i, tuple(index_parents_i)
                    )
                    append_value(
                        self.cdag.cg.sepset, j, i, tuple(index_parents_j)
                    )
                    self.sep_sets[(j, i)] = set(index_parents_i).union(
                        index_parents_j
                    )

        for cluster_name in self.cdag.cdag_list_of_topological_sort:
            self.cluster_phase(cluster_name)

        # reorient remaining edges according to bidirected edges
        # analog to reorientAllWith(graph, Endpoint.CIRCLE)
        # self.cdag.cdag_to_circle_mpdag(cg=self.cdag.cg)
        self.cdag.reorient_cg_with_cdag()
        # reorientAllWith(self.cdag.cg.G, Endpoint.CIRCLE)

        self.rule0(
            self.cdag.cg.G,
            self.cdag.cg.G.nodes,
            self.sep_sets,
            self.background_knowledge,
            self.verbose,
        )

        removeByPossibleDsep(
            self.cdag.cg.G,
            self.independence_test_method,
            self.alpha,
            self.sep_sets,
        )

        # # analog to reorientAllWith(graph, Endpoint.CIRCLE), but keeps arrows from C-DAG
        # self.cdag.cdag_to_circle_mpdag(cg=self.cdag.cg)
        self.cdag.reorient_cg_with_cdag()
        # reorientAllWith(self.cdag.cg.G, Endpoint.CIRCLE)

        self.rule0(
            self.cdag.cg.G,
            self.cdag.cg.G.nodes,
            self.sep_sets,
            self.background_knowledge,
            self.verbose,
        )

        change_flag = True
        first_time = True

        while change_flag:
            change_flag = False
            change_flag = rulesR1R2cycle(
                self.cdag.cg.G,
                self.background_knowledge,
                change_flag,
                self.verbose,
            )
            change_flag = ruleR3(
                self.cdag.cg.G,
                self.sep_sets,
                self.background_knowledge,
                change_flag,
                self.verbose,
            )

            if change_flag or (
                first_time
                and self.background_knowledge is not None
                and len(self.background_knowledge.forbidden_rules_specs) > 0
                and len(self.background_knowledge.required_rules_specs) > 0
                and len(self.background_knowledge.tier_map.keys()) > 0
            ):
                change_flag = ruleR4B(
                    self.cdag.cg.G,
                    self.max_path_length,
                    self.dataset,
                    self.independence_test_method,
                    self.alpha,
                    self.sep_sets,
                    change_flag,
                    self.background_knowledge,
                    self.verbose,
                )

                first_time = False

                if self.verbose:
                    print("Epoch")

        self.cdag.cg.G.set_pag(True)

        edges = get_color_edges(self.cdag.cg.G)

        return self.cdag.cg, edges

    def cluster_phase(self, cluster_name) -> CausalGraph:
        """
        Same as cluster_phase for PC
        """
        start_cluster = time.time()
        assert type(self.dataset) == np.ndarray
        assert 0 < self.alpha < 1
        depth = -1
        cluster = ClusterDAG.get_node_by_name(
            cluster_name, self.cdag.cluster_graph
        )
        cluster_node_indices = self.cdag.get_node_indices_of_cluster(cluster)
        cluster_node_indices = np.array(sorted(cluster_node_indices))
        local_graph = self.cdag.get_local_graph(cluster)

        # Only to check max degree
        local_graph_node_indices = np.array(
            [self.cdag.cg.G.node_map[node] for node in local_graph.G.nodes]
        )
        local_graph_node_indices = np.array(sorted(local_graph_node_indices))

        if self.verbose:
            print(
                f"Cluster node indices of {cluster.get_name()} are {cluster_node_indices}"
            )

        if self.verbose:
            print(
                f"Local graph node indices of {cluster.get_name()} are {local_graph_node_indices}"
            )
        if self.show_progress:
            if len(local_graph_node_indices) == 1:
                pbar = tqdm(total=1)
                pbar.reset()
                pbar.update()
                pbar.set_description(
                    f"{cluster.get_name()} phase, no nonchild, nothing to do"
                )
            else:
                pbar = tqdm(total=cluster_node_indices.shape[0])

        # Depth loop
        # Depends on max nonchilds within cluster and max degree of parents
        # of cluster, as sets in neighbors of top cluster nodes are considered
        # as possible separating sets
        while (
            self.cdag.max_nonchild_degree_of_cluster(cluster) - 1 > depth
        ) or (self.cdag.max_degree_of_cluster_parents(cluster) - 1 > depth):
            depth += 1
            if self.verbose:
                print(f"Depth is {depth}")
            depth_start = time.time()
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in cluster_node_indices:
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(
                        f"{cluster.get_name()} phase, Depth={depth}, working on node {x}"
                    )
                Nonchilds_x = self.cdag.get_nonchilds(x)
                # Remove neighbors that are not in local_graph or are in cluster
                # cluster_mask = np.isin(Nonchilds_x, cluster_node_indices)
                # Pa_x_outside_cluster = Nonchilds_x[~cluster_mask]
                # local_mask = np.isin(Nonchilds_x, local_graph_node_indices)
                # Nonchilds_x_local_graph = Nonchilds_x[local_mask]
                if len(Nonchilds_x) < depth - 1:
                    continue
                if self.verbose:
                    print(f"Nonchilds of {x} are {Nonchilds_x}")
                for y in Nonchilds_x:
                    if self.verbose:
                        print("Testing edges from %d to %d" % (x, y))
                        # No other background knowledge functionality supported for now
                    sepsets = set()
                    Nonchilds_x_no_y = np.delete(
                        Nonchilds_x,
                        np.where(Nonchilds_x == y),
                    )
                    # y is either in cluster_node_indices or in local_graph_node_indices without
                    # cluster_node_indices. Either way first search for sepset in Nonchilds(x)
                    # and if y in "upper cluster", search for sepset in neighbors(y) too
                    for S in combinations(Nonchilds_x_no_y, depth):
                        # print(f'Set S to be tested is {S}')
                        p = self.cdag.cg.ci_test(x, y, S)
                        self.no_of_indep_tests_performed += 1
                        if p > self.alpha:
                            if self.verbose:
                                print(
                                    "%d ind %d | %s with p-value %f"
                                    % (x, y, S, p)
                                )
                            if not self.stable:
                                edge1 = self.cdag.cg.G.get_edge(
                                    self.cdag.cg.G.nodes[x],
                                    self.cdag.cg.G.nodes[y],
                                )
                                if edge1 is not None:
                                    self.cdag.cg.G.remove_edge(edge1)
                                edge2 = self.cdag.cg.G.get_edge(
                                    self.cdag.cg.G.nodes[y],
                                    self.cdag.cg.G.nodes[x],
                                )
                                if edge2 is not None:
                                    self.cdag.cg.G.remove_edge(edge2)
                                append_value(self.cdag.cg.sepset, x, y, S)
                                append_value(self.cdag.cg.sepset, y, x, S)
                                self.sep_sets[(x, y)] = set(S)
                                self.sep_sets[(y, x)] = set(S)
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
                            if self.verbose:
                                print(
                                    "%d dep %d | %s with p-value %f"
                                    % (x, y, S, p)
                                )
                    # print(f'Added sepset: {x} !- {y} | {tuple(sepsets)}')
                    # print(f'Type of sepsets is {type(sepsets)}')
                    append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                    append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
                    # x_node = self.cdag.get_key_by_value(
                    #     self.cdag.cg.G.node_map, x
                    # )
                    # y_node = self.cdag.get_key_by_value(
                    #     self.cdag.cg.G.node_map, y
                    # )
                    if y in np.intersect1d(
                        cluster_node_indices, local_graph_node_indices
                    ):
                        # Consider separating sets in neighbors(y)
                        Neigh_y = self.cdag.cg.neighbors(y)
                        if len(Neigh_y) < depth - 1:
                            continue
                        if self.verbose:
                            print(
                                f"Neighbors of {y} in local graph are {Neigh_y}"
                            )
                        Neighbors_y_no_x = np.delete(
                            Neigh_y, np.where(Neigh_y == x)
                        )
                        for S in combinations(Neighbors_y_no_x, depth):
                            # print(f'Set S to be tested is {S}')
                            p = self.cdag.cg.ci_test(x, y, S)
                            if p > self.alpha:
                                if self.verbose:
                                    print(
                                        "%d ind %d | %s with p-value %f"
                                        % (x, y, S, p)
                                    )
                                if not self.stable:
                                    edge1 = self.cdag.cg.G.get_edge(
                                        self.cdag.cg.G.nodes[x],
                                        self.cdag.cg.G.nodes[y],
                                    )
                                    if edge1 is not None:
                                        self.cdag.cg.G.remove_edge(edge1)
                                    edge2 = self.cdag.cg.G.get_edge(
                                        self.cdag.cg.G.nodes[y],
                                        self.cdag.cg.G.nodes[x],
                                    )
                                    if edge2 is not None:
                                        self.cdag.cg.G.remove_edge(edge2)
                                    append_value(self.cdag.cg.sepset, x, y, S)
                                    append_value(self.cdag.cg.sepset, y, x, S)
                                    self.sep_sets[(x, y)] = set(S)
                                    self.sep_sets[(y, x)] = set(S)
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
                                if self.verbose:
                                    print(
                                        "%d dep %d | %s with p-value %f"
                                        % (x, y, S, p)
                                    )
                        # print(f'Added sepset: {x} !- {y} | {tuple(sepsets)}')
                        # print(f'Type of sepsets is {type(sepsets)}')
                        append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                        append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
            if self.show_progress:
                pbar.refresh()

            for x, y in list(set(edge_removal)):
                edge1 = self.cdag.cg.G.get_edge(
                    self.cdag.cg.G.nodes[x], self.cdag.cg.G.nodes[y]
                )
                if edge1 is not None:
                    self.cdag.cg.G.remove_edge(edge1)
                    x_name = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, x
                    )
                    y_name = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, y
                    )
                    if self.verbose:
                        print(f"Deleted edge from {x_name} to {y_name}")
                if self.cdag.cg.sepset[x, y] is not None:
                    origin_list = []
                    for l_out in self.cdag.cg.sepset[x, y]:
                        for l_in in l_out:
                            origin_list.append(l_in)
                    self.sep_sets[(x, y)] = set(origin_list)
                    self.sep_sets[(y, x)] = set(origin_list)

            local_graph = self.cdag.get_local_graph(cluster)
            # print('LOCAL GRAPH DRAWN BELOW')
            # local_graph.draw_pydot_graph()
            depth_end = time.time()
            # print(f"Depth {depth} took {depth_end - depth_start:.2f}sec")

        end_cluster = time.time()
        time_elapsed = end_cluster - start_cluster

        if self.show_progress:
            pbar.set_postfix_str(
                f"duration: {time_elapsed:.2f}sec", refresh=True
            )
            pbar.close()

        return self.cdag.cg

    def rule0(
        self,
        graph: Graph,
        nodes: List[Node],
        sep_sets: Dict[Tuple[int, int], Set[int]],
        knowledge: BackgroundKnowledge | None,
        verbose: bool,
    ):
        # reorientAllWith(graph, Endpoint.CIRCLE)
        self.cdag.reorient_cg_with_cdag()
        # self.cdag.cdag_to_circle_mpdag(cg=self.cdag.cg)
        fci_orient_bk(knowledge, graph)
        for node_b in nodes:
            adjacent_nodes = graph.get_adjacent_nodes(node_b)
            if len(adjacent_nodes) < 2:
                continue

            cg = ChoiceGenerator(len(adjacent_nodes), 2)
            combination = cg.next()
            while combination is not None:
                node_a = adjacent_nodes[combination[0]]
                node_c = adjacent_nodes[combination[1]]
                combination = cg.next()

                if graph.is_adjacent_to(node_a, node_c):
                    continue
                if graph.is_def_collider(node_a, node_b, node_c):
                    continue
                # check if is collider
                sep_set = sep_sets.get(
                    (
                        graph.get_node_map()[node_a],
                        graph.get_node_map()[node_c],
                    )
                )
                if sep_set is not None and not sep_set.__contains__(
                    graph.get_node_map()[node_b]
                ):
                    if not is_arrow_point_allowed(
                        node_a, node_b, graph, knowledge
                    ):
                        continue
                    if not is_arrow_point_allowed(
                        node_c, node_b, graph, knowledge
                    ):
                        continue

                    edge1 = graph.get_edge(node_a, node_b)
                    graph.remove_edge(edge1)
                    graph.add_edge(
                        Edge(
                            node_a,
                            node_b,
                            edge1.get_proximal_endpoint(node_a),
                            Endpoint.ARROW,
                        )
                    )

                    edge2 = graph.get_edge(node_c, node_b)
                    graph.remove_edge(edge2)
                    graph.add_edge(
                        Edge(
                            node_c,
                            node_b,
                            edge2.get_proximal_endpoint(node_c),
                            Endpoint.ARROW,
                        )
                    )

                    if verbose:
                        print(
                            "Orienting collider: "
                            + node_a.get_name()
                            + " *-> "
                            + node_b.get_name()
                            + " <-* "
                            + node_c.get_name()
                        )
