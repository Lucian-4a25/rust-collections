use std::{collections::HashMap, hash::Hash};

use crate::graph::visit::{EdgeRef, GraphProp, IntoEdgeReferences, NodeCount, NodeIndexable};

use super::WeightMeasure;

#[derive(Debug)]
pub struct NegativeCycleErr;

#[allow(dead_code)]
/// floyd_warshall shortest path algorithm, this algorithm has N * N * N time complexity,
/// which means it scale badly.
pub fn floyd_warshall<G, F, V>(
    graph: G,
    edge_weight: F,
) -> Result<HashMap<(G::NodeId, G::NodeId), V>, NegativeCycleErr>
where
    G: IntoEdgeReferences + NodeIndexable + NodeCount + GraphProp,
    G::NodeId: Hash + Eq,
    F: Fn(G::EdgeRef) -> V,
    V: WeightMeasure,
{
    let node_count = graph.node_count();
    let mut distance = vec![vec![V::infinite(); node_count]; node_count];

    // we must be sure there is not duplicate edges between same pair nodes
    for edge_ref in graph.edge_references() {
        let (source, target) = (edge_ref.source(), edge_ref.target());
        let (source_idx, target_idx) = (graph.to_index(source), graph.to_index(target));
        let weight = edge_weight(edge_ref);

        distance[source_idx][target_idx] = weight;
        if !G::is_directed(graph) {
            distance[target_idx][source_idx] = weight;
        }
    }

    // the default value from node to self is zerod
    for i in 0..node_count {
        distance[i][i] = V::default();
    }

    // increasly add extra node to find the shortest path in every node pair
    for k in 1..node_count {
        for i in 0..node_count {
            for j in 0..node_count {
                let (combined_weight, overflowed) = distance[i][k].overflowing_add(distance[k][j]);
                if !overflowed && distance[i][j] > combined_weight {
                    distance[i][j] = combined_weight;
                }
            }
        }
    }

    for i in 0..node_count {
        if distance[i][i] < V::min() {
            return Err(NegativeCycleErr);
        }
    }

    let mut results = HashMap::new();
    for i in 0..node_count {
        for j in 0..node_count {
            results.insert((graph.from_index(i), graph.from_index(j)), distance[i][j]);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod test_floyd_warshall {
    use super::*;

    #[test]
    /// adopt pet_graph's test example
    fn test_basic() {
        use crate::graph::graph_adjacency_list::Graph;
        let mut graph: Graph<(), (), _> = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());

        for edge in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] {
            graph.add_edge(edge.0, edge.1, ());
        }

        let weight_map: HashMap<(_, _), i32> = [
            ((a, a), 0),
            ((a, b), 1),
            ((a, c), 4),
            ((a, d), 10),
            ((b, b), 0),
            ((b, c), 2),
            ((b, d), 2),
            ((c, c), 0),
            ((c, d), 2),
        ]
        .iter()
        .cloned()
        .collect();
        //     ----- b --------
        //    |      ^         | 2
        //    |    1 |    4    v
        //  2 |      a ------> c
        //    |   10 |         | 2
        //    |      v         v
        //     --->  d <-------

        let inf = std::i32::MAX;
        let expected_res: HashMap<(_, _), i32> = [
            ((a, a), 0),
            ((a, b), 1),
            ((a, c), 3),
            ((a, d), 3),
            ((b, a), inf),
            ((b, b), 0),
            ((b, c), 2),
            ((b, d), 2),
            ((c, a), inf),
            ((c, b), inf),
            ((c, c), 0),
            ((c, d), 2),
            ((d, a), inf),
            ((d, b), inf),
            ((d, c), inf),
            ((d, d), 0),
        ]
        .iter()
        .cloned()
        .collect();

        let res = floyd_warshall(&graph, |edge| {
            if let Some(weight) = weight_map.get(&(edge.source(), edge.target())) {
                *weight
            } else {
                inf
            }
        })
        .unwrap();

        let nodes = [a, b, c, d];
        for node1 in &nodes {
            for node2 in &nodes {
                assert_eq!(
                    res.get(&(*node1, *node2)).unwrap(),
                    expected_res.get(&(*node1, *node2)).unwrap()
                );
            }
        }
    }
}
