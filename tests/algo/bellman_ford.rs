use rust_collections::graph::algo::bellman_ford::{bellman_ford, find_negative_cycle};
use rust_collections::graph::graph_adjacency_list::Graph;

/// copy the test suit from pet_graph
#[test]
fn test_find_negative_cycle() {
    let mut g: Graph<usize, f64> = Graph::new();
    for i in 0..5 {
        g.add_node(i);
    }
    for edge in [
        (0, 1, 1.),
        (0, 2, 1.),
        (0, 3, 1.),
        (1, 3, 1.),
        (2, 1, 1.),
        (3, 2, -3.),
    ] {
        g.add_edge(edge.0, edge.1, edge.2);
    }

    let path = find_negative_cycle(&g, 0);
    assert_eq!(path, Some([1, 3, 2].to_vec()));
}

/// copy the test suit from pet_graph
#[test]
#[allow(unused_variables)]
fn test_bellman_fords() {
    let mut g = Graph::new();
    let a = g.add_node(()); // node with no weight
    let b = g.add_node(());
    let c = g.add_node(());
    let d = g.add_node(());
    let e = g.add_node(());
    let f = g.add_node(());
    for e in [
        (0, 1, 2.0),
        (0, 3, 4.0),
        (1, 2, 1.0),
        (1, 5, 7.0),
        (2, 4, 5.0),
        (4, 5, 1.0),
        (3, 4, 1.0),
    ] {
        g.add_edge(e.0, e.1, e.2);
    }

    // Graph represented with the weight of each edge
    //
    //     2       1
    // a ----- b ----- c
    // | 4     | 7     |
    // d       f       | 5
    // | 1     | 1     |
    // \------ e ------/

    let path = bellman_ford(&g, a);
    assert!(path.is_ok());
    let path = path.unwrap();
    assert_eq!(path.distances, vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        path.predecessors,
        vec![None, Some(a), Some(b), Some(a), Some(d), Some(e)]
    );

    // Node f (indice 5) can be reach from a with a path costing 6.
    // Predecessor of f is Some(e) which predecessor is Some(d) which predecessor is Some(a).
    // Thus the path from a to f is a <-> d <-> e <-> f

    let mut ng = Graph::new_undirected();

    for _ in 0..7 {
        ng.add_node(());
    }

    for e in [
        (0, 1, -2.0),
        (0, 3, -4.0),
        (1, 2, -1.0),
        (1, 5, -25.0),
        (2, 4, -5.0),
        (4, 5, -25.0),
        (3, 4, -1.0),
    ] {
        ng.add_edge(e.0, e.1, e.2);
    }
    assert!(bellman_ford(&ng, 0).is_err());
}

#[test]
#[allow(dead_code, unused_variables)]
fn test_bellman_fords_with_integer() {
    let mut g = Graph::new();
    let a = g.add_node(()); // node with no weight
    let b = g.add_node(());
    let c = g.add_node(());
    let d = g.add_node(());
    let e = g.add_node(());
    let f = g.add_node(());
    for e in [
        (0, 1, 2),
        (0, 3, 4),
        (1, 2, 1),
        (1, 5, 7),
        (2, 4, 5),
        (4, 5, 1),
        (3, 4, 1),
    ] {
        g.add_edge(e.0, e.1, e.2);
    }

    // Graph represented with the weight of each edge
    //
    //     2       1
    // a ----- b ----- c
    // | 4     | 7     |
    // d       f       | 5
    // | 1     | 1     |
    // \------ e ------/

    let path = bellman_ford(&g, a);
    assert!(path.is_ok());
    let path = path.unwrap();
    assert_eq!(path.distances, vec![0, 2, 3, 4, 5, 6]);
    assert_eq!(
        path.predecessors,
        vec![None, Some(a), Some(b), Some(a), Some(d), Some(e)]
    );
}
