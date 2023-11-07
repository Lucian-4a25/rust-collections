use itertools::assert_equal;
use rust_collections::graph::graph_adjacency_list::Graph;
use rust_collections::graph::{algo::simple_paths::all_simple_paths, Directed};
use std::{collections::HashSet, iter::FromIterator};

type DiGraph<N, E> = Graph<N, E, Directed>;

#[test]
fn test_all_simple_paths() {
    let graph = DiGraph::<i32, i32>::from_edges([
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 4),
        (4, 2),
        (4, 5),
        (5, 2),
        (5, 3),
    ]);

    let expexted_simple_paths_0_to_5 = vec![
        vec![0usize, 1, 2, 3, 4, 5],
        vec![0, 1, 2, 4, 5],
        vec![0, 1, 3, 2, 4, 5],
        vec![0, 1, 3, 4, 5],
        vec![0, 2, 3, 4, 5],
        vec![0, 2, 4, 5],
        vec![0, 3, 2, 4, 5],
        vec![0, 3, 4, 5],
    ];

    let actual_simple_paths_0_to_5: HashSet<Vec<_>> = all_simple_paths(&graph, 0, 5, 0, None)
        .map(|v: Vec<_>| v.into_iter().map(|i| i).collect())
        .collect();
    assert_eq!(actual_simple_paths_0_to_5.len(), 8);
    assert_eq!(
        HashSet::from_iter(expexted_simple_paths_0_to_5),
        actual_simple_paths_0_to_5
    );
}

#[test]
fn test_one_simple_path() {
    let graph = DiGraph::<i32, i32>::from_edges([(0, 1), (2, 1)]);

    let expexted_simple_paths_0_to_1 = &[vec![0usize, 1]];
    let actual_simple_paths_0_to_1: Vec<Vec<_>> = all_simple_paths(&graph, 0, 1, 0, None)
        .map(|v: Vec<_>| v.into_iter().map(|i| i).collect())
        .collect();

    assert_eq!(actual_simple_paths_0_to_1.len(), 1);
    assert_equal(expexted_simple_paths_0_to_1, &actual_simple_paths_0_to_1);
}

#[test]
fn test_no_simple_paths() {
    let graph = DiGraph::<i32, i32>::from_edges([(0, 1), (2, 1)]);

    let actual_simple_paths_0_to_2: Vec<Vec<_>> = all_simple_paths(&graph, 0, 2, 0, None)
        .map(|v: Vec<_>| v.into_iter().map(|i| i).collect())
        .collect();

    assert_eq!(actual_simple_paths_0_to_2.len(), 0);
}
