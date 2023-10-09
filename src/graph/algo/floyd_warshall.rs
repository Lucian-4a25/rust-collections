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
