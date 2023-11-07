use crate::graph::{
    visit::{IntoNeiborghbors, IntoNeighborsDirected, NodeCount},
    Direction,
};
use indexmap::IndexSet;
use std::{hash::Hash, iter::from_fn};

// simple_paths, see: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html
pub fn all_simple_paths<G, P>(
    graph: G,
    from: G::NodeId,
    to: G::NodeId,
    min_intermediate_nodes: usize,
    max_intermediate_nodes: Option<usize>,
) -> impl Iterator<Item = P>
where
    G: IntoNeighborsDirected + NodeCount,
    G::NodeId: Hash + Eq,
    P: FromIterator<G::NodeId>,
{
    let max_len = if let Some(m) = max_intermediate_nodes {
        m + 1
    } else {
        graph.node_count() - 1
    };
    let mut stack = vec![graph.neighbors_directed(from, Direction::Outcoming)];
    let mut visited: IndexSet<G::NodeId> = IndexSet::from_iter(Some(from));

    from_fn(move || {
        while let Some(children) = stack.last_mut() {
            if visited.len() <= max_len {
                if let Some(child) = children.next() {
                    if child == to && visited.len() >= min_intermediate_nodes {
                        let r = visited.iter().cloned().chain(Some(child)).collect::<P>();
                        stack.pop();
                        visited.pop();
                        return Some(r);
                    } else if !visited.contains(&child) && visited.len() < max_len {
                        visited.insert(child);
                        stack.push(graph.neighbors_directed(child, Direction::Outcoming));
                        continue;
                    }
                }
            }

            stack.pop();
            visited.pop();
        }

        None
    })
}
