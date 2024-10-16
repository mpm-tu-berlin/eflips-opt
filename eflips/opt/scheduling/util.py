import json
from typing import List, Tuple, Dict

import networkx as nx  # type: ignore


def _validate_input_graph(graph: nx.Graph) -> nx.Graph:
    """
    Makes sure the input graph

    - is a directed acyclic graph
    - each node has an integer ID
    - each node has a tuple exactly 2 weights, which are either None or floats between 0 and 1
        - if the tuple is empty, it is filled with None
    - each edge has a single weight, which is an integer between 0 and 1e6

    :param graph: A directed acyclic graph as a networkx Graph
    :return: The validated graph. Raises an error if the graph is invalid
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The input graph is not a directed acyclic graph.")

    for node in graph.nodes:
        if not isinstance(node, int):
            raise ValueError("Each node must have an integer ID.")
        if not isinstance(graph.nodes[node]["weight"], tuple):
            raise ValueError("Each node must have a tuple of exactly two weights.")
        if len(graph.nodes[node]["weight"]) != 2:
            raise ValueError("Each node must have a tuple of exactly two weights.")

        for weight in graph.nodes[node]["weight"]:
            if not (
                isinstance(weight, int) or isinstance(weight, float) or weight is None
            ):
                raise ValueError(
                    "Each weight in the tuple must be an integer, float or None."
                )
            if weight is not None and (weight < 0 or weight > 1):
                raise ValueError("Each weight in the tuple must be between 0 and 1.")

    for edge in graph.edges:
        if not isinstance(graph.edges[edge]["weight"], int):
            raise ValueError(
                "Each edge must have a single weight, which is an integer between 0 and 1e6."
            )
        if graph.edges[edge]["weight"] < 0 or graph.edges[edge]["weight"] > 1e6:
            raise ValueError(
                "Each edge must have a single weight, which is an integer between 0 and 1e6."
            )

    return graph


def _subgraph_to_dict(
    graph: nx.Graph,
) -> Dict[str, List[Dict[str, int | Tuple[int | float | None, ...]]]]:
    """
    Convert a graph to a JSON object that can be used by the rust optimization model.

    The JSON output will have the following structure:
    {
        "nodes": [
            {
                "id": 0,
                "weight": (0.5, 0.3) # The weights of the node, (2-tuple, float (0-1) or None)
            },
            ...
        ],
        "edges": [
            {
                "source": 0,
                "target": 1,
                "weight": 300 # The edge weight (integer, 0-1e6)
            },
            ...
        ]
    }

    :param graph: A directed acyclic graph, containing the trips as nodes and the possible connections as edges.
    :return: A Dict that can be converted to JSON
    """
    nodes = []
    for node in graph.nodes:
        nodes.append(
            {
                "id": node,
                "weight": (
                    graph.nodes[node]["weight"][0],
                    graph.nodes[node]["weight"][1],
                ),
            }
        )
    edges = []
    for edge in graph.edges:
        edges.append(
            {
                "source": edge[0],
                "target": edge[1],
                "weight": graph.edges[edge]["weight"],
            }
        )
    return {"nodes": nodes, "edges": edges}


def _sort_graph_json(
    graph_json: List[Dict[str, List[Dict[str, int | Tuple[int | float | None, ...]]]]]
) -> List[Dict[str, List[Dict[str, int | Tuple[int | float | None, ...]]]]]:
    """
    For repeatability, sort the graph JSON
    :param graph_json: a list of dictionaries, each containing a 'nodes' and 'edges' key
    :return: a sorted list of dictionaries
    """

    # loaded is a list of dictionaries, with the following keys:
    # 'nodes': a list of 'id', 'weight' dicts
    # 'edges': a list of 'source', 'target', 'weight' dicts
    # We will sort each entrie's nodes by the id and edges by the (source, target) tuple
    for entry in graph_json:
        entry["nodes"] = sorted(entry["nodes"], key=lambda x: x["id"])
        entry["edges"] = sorted(
            entry["edges"], key=lambda x: (x["source"], x["target"])
        )

    # We will then sort the entries by size, descending
    graph_json = sorted(graph_json, key=lambda x: len(x["nodes"]), reverse=True)
    return graph_json


def _graph_to_json(graph: nx.Graph) -> str:
    """
    For historical reasons, we are passing the input to the solver as a slightly wonky JSON string. This function
    converts the input graph to that JSON string.
    :param graph:
    :return:
    """

    result = []
    for connected_component in nx.connected_components(graph.to_undirected()):
        subgraph = graph.subgraph(connected_component).copy()

        # filename is the number of nodes with leading zeros + a random UUID + .json
        the_dict = _subgraph_to_dict(subgraph)
        result.append(the_dict)
    result = _sort_graph_json(result)
    return json.dumps(result)
