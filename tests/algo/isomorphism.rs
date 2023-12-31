use rust_collections::graph::algo::isomorphism::{
    is_isomorphism_matching, is_isomorphism_semantic_matching, isomorphism_semantic_matching_iter,
    vf2pp_is_isomorphism_matching, vf2pp_is_isomorphism_semantic_matching,
    vf2pp_isomorphism_semantic_matching_iter,
};
use rust_collections::graph::graph_adjacency_list::Graph;
use rust_collections::graph::visit::IntoNeighborsDirected;
use rust_collections::graph::{Directed, Direction, GraphType, UnDirected};
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;

/// Petersen A and B are isomorphic
///
/// http://www.dharwadker.org/tevet/isomorphism/
const PETERSEN_A: &str = "
 0 1 0 0 1 0 1 0 0 0
 1 0 1 0 0 0 0 1 0 0
 0 1 0 1 0 0 0 0 1 0
 0 0 1 0 1 0 0 0 0 1
 1 0 0 1 0 1 0 0 0 0
 0 0 0 0 1 0 0 1 1 0
 1 0 0 0 0 0 0 0 1 1
 0 1 0 0 0 1 0 0 0 1
 0 0 1 0 0 1 1 0 0 0
 0 0 0 1 0 0 1 1 0 0
";

const PETERSEN_B: &str = "
 0 0 0 1 0 1 0 0 0 1
 0 0 0 1 1 0 1 0 0 0
 0 0 0 0 0 0 1 1 0 1
 1 1 0 0 0 0 0 1 0 0
 0 1 0 0 0 0 0 0 1 1
 1 0 0 0 0 0 1 0 1 0
 0 1 1 0 0 1 0 0 0 0
 0 0 1 1 0 0 0 0 1 0
 0 0 0 0 1 1 0 1 0 0
 1 0 1 0 1 0 0 0 0 0
";

/// An almost full set, isomorphic
const FULL_A: &str = "
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 0 1 1 1 0 1
 1 1 1 1 1 1 1 1 1 1
";

const FULL_B: &str = "
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 0 1 1 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1
";

/// Praust A and B are not isomorphic
const PRAUST_A: &str = "
 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0
 1 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0
 1 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0
 0 1 0 0 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 1 0 1 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0
 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0
 0 1 0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 1 0 0
 0 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0
 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1
 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0
 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 1 0 0 0
 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 1 0 0 0 1
 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 1 0
 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 1 1
 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 1 0 1
 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 1 0
";

const PRAUST_B: &str = "
 0 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 1 1 0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0
 1 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0
 1 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0
 0 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 1 0 1 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0
 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0
 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0
 0 1 0 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 0 0
 0 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0
 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0
 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 0 1
 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 0
 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 1 0 1
 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 1 0
 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 1 0
 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 0 0 1
 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 1 0 0 1
 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 1 1 0
";

const G1U: &str = "
0 1 1 0 1
1 0 1 0 0
1 1 0 0 0
0 0 0 0 0
1 0 0 0 0
";

const G2U: &str = "
0 1 0 1 0
1 0 0 1 1
0 0 0 0 0
1 1 0 0 0
0 1 0 0 0
";

const G4U: &str = "
0 1 1 0 1
1 0 0 1 0
1 0 0 0 0
0 1 0 0 0
1 0 0 0 0
";

const G1D: &str = "
0 1 1 0 1
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
";

const G4D: &str = "
0 1 1 0 1
0 0 0 1 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
";

// G8 1,2 are not iso
const G8_1: &str = "
0 1 1 0 0 1 1 1
1 0 1 0 1 0 1 1
1 1 0 1 0 0 1 1
0 0 1 0 1 1 1 1
0 1 0 1 0 1 1 1
1 0 0 1 1 0 1 1
1 1 1 1 1 1 0 1
1 1 1 1 1 1 1 0
";

const G8_2: &str = "
0 1 0 1 0 1 1 1
1 0 1 0 1 0 1 1
0 1 0 1 0 1 1 1
1 0 1 0 1 0 1 1
0 1 0 1 0 1 1 1
1 0 1 0 1 0 1 1
1 1 1 1 1 1 0 1
1 1 1 1 1 1 1 0
";

// G3 1,2 are not iso
const G3_1: &str = "
0 1 0
1 0 1
0 1 0
";
const G3_2: &str = "
0 1 1
1 0 1
1 1 0
";

// Non-isomorphic due to selfloop difference
const S1: &str = "
1 1 1
1 0 1
1 0 0
";
const S2: &str = "
1 1 1
0 1 1
1 0 0
";

/// Parse a text adjacency matrix format into a directed graph
fn parse_graph<Ty: GraphType>(s: &str) -> Graph<(), (), Ty> {
    let mut gr = Graph::with_capacity((0, 0));
    let s = s.trim();
    let lines = s.lines().filter(|l| !l.is_empty());
    for (row, line) in lines.enumerate() {
        for (col, word) in line.split(' ').filter(|s| !s.is_empty()).enumerate() {
            let has_edge = word.parse::<i32>().unwrap();
            assert!(has_edge == 0 || has_edge == 1);
            if has_edge == 0 {
                continue;
            }
            while col >= gr.node_count() || row >= gr.node_count() {
                gr.add_node(());
            }
            gr.add_edge(row, col, ());
        }
    }
    gr
}

fn str_to_graph(s: &str) -> Graph<(), (), UnDirected> {
    parse_graph(s)
}

fn str_to_digraph(s: &str) -> Graph<(), (), Directed> {
    parse_graph(s)
}

/// Parse a file in adjacency matrix format into a directed graph
fn graph_from_file(path: &str) -> Graph<(), (), Directed> {
    let mut f = File::open(path).expect("file not found");
    let mut contents = String::new();
    f.read_to_string(&mut contents)
        .expect("failed to read from file");
    parse_graph(&contents)
}

/*
fn graph_to_ad_matrix<N, E, Ty: EdgeType>(g: &Graph<N,E,Ty>)
{
    let n = g.node_count();
    for i in (0..n) {
        for j in (0..n) {
            let ix = NodeIndex::new(i);
            let jx = NodeIndex::new(j);
            let out = match g.find_edge(ix, jx) {
                None => "0",
                Some(_) => "1",
            };
            print!("{} ", out);
        }
        println!("");
    }
}
*/

#[test]
fn petersen_iso() {
    // The correct isomorphism is
    // 0 => 0, 1 => 3, 2 => 1, 3 => 4, 5 => 2, 6 => 5, 7 => 7, 8 => 6, 9 => 8, 4 => 9
    let peta = str_to_digraph(PETERSEN_A);
    let petb = str_to_digraph(PETERSEN_B);
    /*
    println!("{:?}", peta);
    graph_to_ad_matrix(&peta);
    println!("");
    graph_to_ad_matrix(&petb);
    */

    assert!(is_isomorphism_matching(&peta, &petb, false));
    assert!(vf2pp_is_isomorphism_matching(&peta, &petb, false));
}

#[test]
fn petersen_undir_iso() {
    // The correct isomorphism is
    // 0 => 0, 1 => 3, 2 => 1, 3 => 4, 5 => 2, 6 => 5, 7 => 7, 8 => 6, 9 => 8, 4 => 9
    let peta = str_to_digraph(PETERSEN_A);
    let petb = str_to_digraph(PETERSEN_B);

    assert!(is_isomorphism_matching(&peta, &petb, false));
    assert!(vf2pp_is_isomorphism_matching(&peta, &petb, false));
}

#[test]
fn full_iso() {
    let a = str_to_graph(FULL_A);
    let b = str_to_graph(FULL_B);

    assert!(is_isomorphism_matching(&a, &b, false));
    assert!(vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
#[cfg_attr(miri, ignore = "Takes too long to run in Miri")]
fn praust_dir_no_iso() {
    let a = str_to_digraph(PRAUST_A);
    let b = str_to_digraph(PRAUST_B);

    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
#[cfg_attr(miri, ignore = "Takes too long to run in Miri")]
fn praust_undir_no_iso() {
    let a = str_to_graph(PRAUST_A);
    let b = str_to_graph(PRAUST_B);

    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn coxeter_di_iso() {
    // The correct isomorphism is
    let a = str_to_digraph(COXETER_A);
    let b = str_to_digraph(COXETER_B);
    assert!(is_isomorphism_matching(&a, &b, false));
    assert!(vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn coxeter_undi_iso() {
    // The correct isomorphism is
    let a = str_to_graph(COXETER_A);
    let b = str_to_graph(COXETER_B);
    assert!(is_isomorphism_matching(&a, &b, false));
    assert!(vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn g14_dir_not_iso() {
    let a = str_to_digraph(G1D);
    let b = str_to_digraph(G4D);
    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn g14_undir_not_iso() {
    let a = str_to_digraph(G1U);
    let b = str_to_digraph(G4U);
    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn g12_undir_iso() {
    let a = str_to_digraph(G1U);
    let b = str_to_digraph(G2U);
    assert!(is_isomorphism_matching(&a, &b, false));
    assert!(vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn g3_not_iso() {
    let a = str_to_digraph(G3_1);
    let b = str_to_digraph(G3_2);
    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn g8_not_iso() {
    let a = str_to_digraph(G8_1);
    let b = str_to_digraph(G8_2);
    assert_eq!(a.edge_count(), b.edge_count());
    assert_eq!(a.node_count(), b.node_count());
    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn s12_not_iso() {
    let a = str_to_digraph(S1);
    let b = str_to_digraph(S2);
    assert_eq!(a.edge_count(), b.edge_count());
    assert_eq!(a.node_count(), b.node_count());
    assert!(!is_isomorphism_matching(&a, &b, false));
    assert!(!vf2pp_is_isomorphism_matching(&a, &b, false));
}

#[test]
fn iso1() {
    let mut g0 = Graph::<_, ()>::new();
    let mut g1 = Graph::<_, ()>::new();
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));

    // very simple cases
    let a0 = g0.add_node(0);
    let a1 = g1.add_node(0);
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    let b0 = g0.add_node(1);
    let b1 = g1.add_node(1);
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    let _ = g0.add_node(2);
    assert!(!is_isomorphism_matching(&g0, &g1, false));
    assert!(!vf2pp_is_isomorphism_matching(&g0, &g1, false));
    let _ = g1.add_node(2);
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    g0.add_edge(a0, b0, ());
    assert!(!is_isomorphism_matching(&g0, &g1, false));
    assert!(!vf2pp_is_isomorphism_matching(&g0, &g1, false));
    g1.add_edge(a1, b1, ());
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
}

#[test]
fn iso2() {
    let mut g0 = Graph::<_, ()>::new();
    let mut g1 = Graph::<_, ()>::new();

    let a0 = g0.add_node(0);
    let a1 = g1.add_node(0);
    let b0 = g0.add_node(1);
    let b1 = g1.add_node(1);
    let c0 = g0.add_node(2);
    let c1 = g1.add_node(2);
    g0.add_edge(a0, b0, ());
    g1.add_edge(c1, b1, ());
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    // a -> b
    // a -> c
    // vs.
    // c -> b
    // c -> a
    g0.add_edge(a0, c0, ());
    g1.add_edge(c1, a1, ());
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));

    // add
    // b -> c
    // vs
    // b -> a

    let _ = g0.add_edge(b0, c0, ());
    let _ = g1.add_edge(b1, a1, ());
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    let d0 = g0.add_node(3);
    let d1 = g1.add_node(3);
    let e0 = g0.add_node(4);
    let e1 = g1.add_node(4);
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
    // add
    // b -> e -> d
    // vs
    // b -> d -> e
    g0.add_edge(b0, e0, ());
    g0.add_edge(e0, d0, ());
    g1.add_edge(b1, d1, ());
    g1.add_edge(d1, e1, ());
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
}

#[test]
fn iso_matching() {
    let g0 = Graph::<(), _>::from_edges([(0, 0, 1), (0, 1, 2), (0, 2, 3), (1, 2, 4)]);

    let mut g1 = g0.clone();
    *g1.edge_weight_mut(0).unwrap() = 0;
    assert!(!is_isomorphism_semantic_matching(
        &g0,
        &g1,
        |x, y| x == y,
        |x, y| x == y,
        false,
    ));
    assert!(!vf2pp_is_isomorphism_semantic_matching(
        &g0,
        &g1,
        |x, y| x == y,
        |x, y| x == y,
        false,
    ));

    let mut g2 = g0.clone();
    *g2.edge_weight_mut(1).unwrap() = 0;
    assert!(!is_isomorphism_semantic_matching(
        &g0,
        &g2,
        |x, y| x == y,
        |x, y| x == y,
        false
    ));
    assert!(!vf2pp_is_isomorphism_semantic_matching(
        &g0,
        &g2,
        |x, y| x == y,
        |x, y| x == y,
        false
    ));
}

#[test]
fn iso_100n_100e() {
    let g0 = str_to_digraph(include_str!("res/graph_100n_100e.txt"));
    let g1 = str_to_digraph(include_str!("res/graph_100n_100e_iso.txt"));
    assert!(is_isomorphism_matching(&g0, &g1, false));
    println!("-------------------");
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
}

#[test]
#[cfg_attr(miri, ignore = "Too large for Miri")]
fn iso_large() {
    let g0 = graph_from_file("tests/algo/res/graph_1000n_1000e.txt");
    let g1 = graph_from_file("tests/algo/res/graph_1000n_1000e.txt");
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
}

// isomorphism isn't correct for multigraphs.
// Keep this testcase to document how
#[should_panic]
#[test]
fn iso_multigraph_failure() {
    let g0 = Graph::<(), ()>::from_edges([(0, 0), (0, 0), (0, 1), (1, 1), (1, 1), (1, 0)]);

    let g1 = Graph::<(), ()>::from_edges([(0, 0), (0, 1), (0, 1), (1, 1), (1, 0), (1, 0)]);
    assert!(is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, false));
}

#[test]
#[cfg_attr(miri, ignore = "Takes too long to run in Miri")]
fn iso_subgraph() {
    let g0 = Graph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 0)]);
    let g1 = Graph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (0, 4)]);
    assert!(!is_isomorphism_matching(&g0, &g1, false));
    assert!(is_isomorphism_matching(&g0, &g1, true));
    assert!(!vf2pp_is_isomorphism_matching(&g0, &g1, false));
    assert!(vf2pp_is_isomorphism_matching(&g0, &g1, true));
}

#[test]
// #[cfg_attr(miri, ignore = "Takes too long to run in Miri")]
fn iter_subgraph() {
    let a = Graph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 0)]);
    let b = Graph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (0, 4)]);
    let a_ref = &a;
    let b_ref = &b;
    let node_match = { |x: &(), y: &()| x == y };
    let edge_match = { |x: &(), y: &()| x == y };

    let mappings =
        isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true).unwrap();
    let mapping_2 =
        vf2pp_isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true);

    // Verify the iterator returns the expected mappings
    let expected_mappings: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];
    for mapping in mappings {
        assert!(expected_mappings.contains(&mapping))
    }
    for mapping in mapping_2 {
        assert!(expected_mappings.contains(&mapping))
    }

    // Verify all the mappings from the iterator are different
    let a = str_to_digraph(COXETER_A);
    let b = str_to_digraph(COXETER_B);
    let a_ref = &a;
    let b_ref = &b;

    let mut unique = HashSet::new();
    assert!(
        isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true)
            .unwrap()
            .all(|x| { unique.insert(x) })
    );
    let mut unique2 = HashSet::new();
    assert!(
        vf2pp_isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true)
            .all(|x| { unique2.insert(x) })
    );

    // The iterator should return None for graphs that are not isomorphic
    let a = str_to_digraph(G8_1);
    let b = str_to_digraph(G8_2);
    let a_ref = &a;
    let b_ref = &b;

    assert!(
        isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true)
            .unwrap()
            .next()
            .is_none()
    );
    assert!(
        vf2pp_isomorphism_semantic_matching_iter(a_ref, b_ref, node_match, edge_match, true)
            .next()
            .is_none()
    );

    // https://github.com/petgraph/petgraph/issues/534
    let mut g = Graph::<String, ()>::new();
    let e1 = g.add_node("l1".to_string());
    let e2 = g.add_node("l2".to_string());
    g.add_edge(e1, e2, ());
    let e3 = g.add_node("l3".to_string());
    g.add_edge(e2, e3, ());
    let e4 = g.add_node("l4".to_string());
    g.add_edge(e3, e4, ());

    let mut sub = Graph::<String, ()>::new();
    let e3 = sub.add_node("l3".to_string());
    let e4 = sub.add_node("l4".to_string());
    sub.add_edge(e3, e4, ());

    let node_match = { |x: &String, y: &String| x == y };
    let edge_match = { |x: &(), y: &()| x == y };
    assert_eq!(
        isomorphism_semantic_matching_iter(&sub, &g, node_match, edge_match, true)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![vec![2, 3]]
    );
    assert_eq!(
        vf2pp_isomorphism_semantic_matching_iter(&sub, &g, node_match, edge_match, true)
            .collect::<Vec<_>>(),
        vec![vec![2, 3]]
    );
}

/// Isomorphic pair
const COXETER_A: &str = "
 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1
 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0
 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0
 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0
 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
";

const COXETER_B: &str = "
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0
 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0
 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1
 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1
 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0
 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 1 0
";
