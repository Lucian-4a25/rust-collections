mod astar;
mod bellman_ford;
mod dijkstra;
mod feedback_arc_set;
mod floyd_warshall;
mod isomorphism;
mod k_shortest_path;
mod matching;

use core::ops::Add;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::Hash,
};

use super::visit::{IntoNeighborsDirected, IntoNodeIdentifiers, Topological, VisitMap, Visitable};

/// A struct to record the edge weight in reverse order
pub struct MinScore<T: PartialOrd, N>(T, N);

impl<T: PartialOrd, N> PartialOrd for MinScore<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl<T: PartialOrd, N> PartialEq for MinScore<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialOrd, N> Ord for MinScore<T, N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match other.0.partial_cmp(&self.0) {
            Some(r) => r,
            None => {
                if other.0.ne(&other.0) && self.0.ne(&self.0) {
                    return Ordering::Equal;
                }
                if self.0.ne(&self.0) {
                    return Ordering::Less;
                }
                return Ordering::Greater;
            }
        }
    }
}

impl<T: PartialOrd, N> Eq for MinScore<T, N> {}

pub trait Measure: PartialOrd + Default + Add<Output = Self> + Copy {}

impl<T> Measure for T where T: PartialOrd + Default + Add<Output = Self> + Copy {}

// used to measure the max and min value
pub trait WeightMeasure: Measure {
    fn min() -> Self;

    fn infinite() -> Self;

    fn overflowing_add(self, rhs: Self) -> (Self, bool);
}

impl WeightMeasure for usize {
    fn infinite() -> Self {
        usize::MAX
    }

    fn min() -> Self {
        usize::MIN
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }
}

impl WeightMeasure for isize {
    fn infinite() -> Self {
        isize::MAX
    }

    fn min() -> Self {
        isize::MIN
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }
}

impl WeightMeasure for i32 {
    fn infinite() -> Self {
        i32::MAX
    }

    fn min() -> Self {
        i32::MIN
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }
}

impl WeightMeasure for i64 {
    fn infinite() -> Self {
        i64::MAX
    }

    fn min() -> Self {
        i64::MIN
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }
}

impl WeightMeasure for f64 {
    fn min() -> Self {
        f64::MIN
    }

    fn infinite() -> Self {
        f64::MAX
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        // for an overflow: a + b > max: both values need to be positive and a > max - b must be satisfied
        let overflow =
            self > Self::default() && rhs > Self::default() && self > Self::infinite() - rhs;

        // for an underflow: a + b < min: overflow can not happen and both values must be negative and a < min - b must be satisfied
        let underflow =
            !overflow && self < Self::default() && rhs < Self::default() && self < Self::MIN - rhs;

        (self + rhs, overflow || underflow)
    }
}

impl WeightMeasure for f32 {
    fn min() -> Self {
        f32::MIN
    }
    fn infinite() -> Self {
        f32::MAX
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        // for an overflow: a + b > max: both values need to be positive and a > max - b must be satisfied
        let overflow =
            self > Self::default() && rhs > Self::default() && self > Self::infinite() - rhs;

        // for an underflow: a + b < min: overflow can not happen and both values must be negative and a < min - b must be satisfied
        let underflow =
            !overflow && self < Self::default() && rhs < Self::default() && self < Self::MIN - rhs;

        (self + rhs, overflow || underflow)
    }
}

/// check if graph has a cycle use depth-frist-search algorithm
#[allow(dead_code)]
pub fn is_graph_cyclic<G>(graph: G) -> bool
where
    G: Visitable + IntoNeighborsDirected + IntoNodeIdentifiers,
{
    let size = graph.node_identifiers().size_hint().0;
    let mut topo = Topological::new(graph);

    let mut counter = 0;
    while let Some(_) = topo.next(graph) {
        counter += 1;
    }
    // println!("size of graph: {size}, the counter: {counter}");

    counter != size
}

#[cfg(test)]
mod test {
    use crate::graph::Directed;

    use super::*;

    #[test]
    fn test_is_graph_cyclic() {
        use crate::graph::graph_adjacency_list::Graph;
        let mut g: Graph<(), (), Directed> = Graph::new();
        for _ in 0..3 {
            g.add_node(());
        }
        g.add_edge(0, 1, ());
        g.add_edge(1, 2, ());
        g.add_edge(2, 0, ());

        assert_eq!(is_graph_cyclic(&g), true);

        g.remove_edge(2);

        assert_eq!(is_graph_cyclic(&g), false);
    }
}
