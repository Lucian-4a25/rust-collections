use super::*;
use indexmap::{map::Entry, IndexMap};
use std::{hash::Hash, marker::PhantomData};

pub trait NodeTrait: Hash + Eq + Ord + Copy {}
impl<N> NodeTrait for N where N: Hash + Eq + Ord + Copy {}

/// Sparse matrix with Diction of Key implemention
pub struct DokGraph<N, E, D: GraphType> {
    nodes: IndexMap<N, Vec<(N, Direction)>>,
    // a map of (N1, N2) to their edge, None if there isn't edge between n1 and n2
    edges: IndexMap<(N, N), E>,
    phantomdata: PhantomData<D>,
}

// Question: Why we don't use Vec<N> to store the N and identify the node with index?
#[allow(dead_code)]
impl<N, E> DokGraph<N, E, Directed> {
    fn new() -> Self {
        Self {
            nodes: IndexMap::new(),
            edges: IndexMap::new(),
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N: NodeTrait, E, D: GraphType> DokGraph<N, E, D> {
    pub fn add_node(&mut self, node: N) -> N {
        self.nodes.entry(node).or_insert(Default::default());
        node
    }

    pub fn remove_node(&mut self, n: N) -> bool {
        match self.nodes.entry(n) {
            Entry::Occupied(entry) => {
                let v = entry.remove();
                // remove all edges related with n
                for (n2, d) in v.into_iter() {
                    let d2 = Direction::oppsite(&d);
                    let n2_mut = self
                        .nodes
                        .get_mut(&n2)
                        .expect("the related node must have edges");
                    let pos = n2_mut
                        .iter()
                        .position(|(node, d)| node == &n && d == &d2)
                        .expect("the related node must have edges");
                    n2_mut.swap_remove(pos);
                    self.remove_edge(n, n2);
                }
                true
            }
            Entry::Vacant(_) => false,
        }
    }

    /// insert an edge between a and b
    pub fn add_edge(&mut self, a: N, b: N, edge: E) -> Option<E> {
        let a_mut = self.nodes.entry(a).or_insert(Default::default());
        if !a_mut.contains(&(b, Direction::Outcoming)) {
            a_mut.push((b, Direction::Outcoming));
        }
        let b_mut = self.nodes.entry(b).or_insert(Default::default());
        if !b_mut.contains(&(a, Direction::Incoming)) {
            b_mut.push((a, Direction::Incoming));
        }

        self.edges.insert((a, b), edge)
    }

    pub fn remove_edge(&mut self, a: N, b: N) -> Option<E> {
        self.nodes.entry(a).and_modify(|a_mut| {
            a_mut
                .iter()
                .position(|v| v == &(b, Direction::Outcoming))
                .map(|p| {
                    a_mut.swap_remove(p);
                });
        });
        self.nodes.entry(b).and_modify(|b_mut| {
            b_mut
                .iter()
                .position(|v| v == &(a, Direction::Incoming))
                .map(|p| {
                    b_mut.swap_remove(p);
                });
        });

        self.edges.remove(&(a, b))
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn contains_edge(&self, a: N, b: N) -> bool {
        self.edges.contains_key(&(a, b))
    }

    pub fn contains_node(&self, n: N) -> bool {
        self.nodes.contains_key(&n)
    }

    pub fn is_directed(&self) -> bool {
        D::is_directed()
    }

    /// return a iterator of nodes with an edge starts from 'n'
    pub fn neighbors(&self, n: N) -> Neighbors<'_, N, D> {
        Neighbors {
            start_node: n,
            direction: Direction::Outcoming,
            nodes: self
                .nodes
                .get(&n)
                .map_or([].iter(), |siblings| siblings.iter()),
            phantomdata: PhantomData,
        }
    }

    pub fn neighbors_directed(&self, n: N, d: Direction) -> Neighbors<'_, N, D> {
        Neighbors {
            start_node: n,
            direction: d,
            nodes: self
                .nodes
                .get(&n)
                .map_or([].iter(), |siblings| siblings.iter()),
            phantomdata: PhantomData,
        }
    }

    /// return a iterators of edges and nodes starts from node 'n'
    pub fn edges(&self, n: N) -> Edges<'_, N, E, D> {
        Edges {
            start_node: n,
            direction: Direction::Outcoming,
            nodes: self
                .nodes
                .get(&n)
                .map_or([].iter(), |sibling| sibling.iter()),
            edges: &self.edges,
            phantomdata: PhantomData,
        }
    }

    pub fn edges_directed(&self, n: N, d: Direction) -> Edges<'_, N, E, D> {
        Edges {
            start_node: n,
            direction: d,
            nodes: self
                .nodes
                .get(&n)
                .map_or([].iter(), |sibling| sibling.iter()),
            edges: &self.edges,
            phantomdata: PhantomData,
        }
    }
}

pub struct Edges<'a, N: NodeTrait, E, Ty: GraphType> {
    start_node: N,
    direction: Direction,
    nodes: std::slice::Iter<'a, (N, Direction)>,
    edges: &'a IndexMap<(N, N), E>,
    phantomdata: PhantomData<Ty>,
}

impl<'a, N: NodeTrait, E, Ty: GraphType> Iterator for Edges<'a, N, E, Ty> {
    type Item = (N, N, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        if Ty::is_directed() {
            self.nodes.find_map(|n| {
                if n.1 == self.direction || n.0 == self.start_node {
                    let k = match n.1 {
                        Direction::Outcoming => (self.start_node, n.0),
                        Direction::Incoming => (n.0, self.start_node),
                    };
                    self.edges.get(&k).map(|e| (k.0, k.1, e))
                } else {
                    None
                }
            })
        } else {
            self.nodes.next().map(|n| {
                let k = match n.1 {
                    Direction::Outcoming => (self.start_node, n.0),
                    Direction::Incoming => (n.0, self.start_node),
                };
                self.edges.get(&k).map(|e| (k.0, k.1, e)).unwrap()
            })
        }
    }
}

pub struct Neighbors<'a, N, Ty: GraphType> {
    start_node: N,
    direction: Direction,
    nodes: std::slice::Iter<'a, (N, Direction)>,
    phantomdata: PhantomData<Ty>,
}

impl<'a, N: NodeTrait, Ty: GraphType> Iterator for Neighbors<'a, N, Ty> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        if Ty::is_directed() {
            self.nodes.find_map(|n| {
                if n.1 == self.direction || n.0 == self.start_node {
                    Some(n.0)
                } else {
                    None
                }
            })
        } else {
            self.nodes.next().map(|n| n.0)
        }
    }
}

#[cfg(test)]
mod test_graphmap {
    use super::*;

    #[test]
    fn test_basic() {
        let mut g: _ = DokGraph::new();
        g.add_node(5);
        g.add_node(4);
        g.add_node(3);
        g.add_node(2);
        g.add_node(1);
        g.add_node(6);
        g.add_edge(1, 2, "1-2");
        g.add_edge(2, 3, "2-3");
        g.add_edge(3, 4, "3-4");
        g.add_edge(4, 5, "4-5");
        g.add_edge(2, 6, "2-6");
        g.add_edge(2, 5, "2-5");
        g.add_edge(1, 4, "1-4");

        for (offset, n) in g.neighbors(2).enumerate() {
            assert_eq!(
                if offset == 0 {
                    3
                } else if offset == 1 {
                    6
                } else {
                    5
                },
                n
            )
        }

        for (offset, n) in g.neighbors_directed(4, Direction::Incoming).enumerate() {
            assert_eq!(if offset == 0 { 3 } else { 1 }, n);
        }

        for (offset, n) in g.neighbors_directed(1, Direction::Outcoming).enumerate() {
            assert_eq!(if offset == 0 { 2 } else { 4 }, n);
        }

        for (offset, n) in g.edges(2).enumerate() {
            if offset == 0 {
                assert_eq!(n, (2, 3, &"2-3"));
            } else if offset == 1 {
                assert_eq!(n, (2, 6, &"2-6"));
            } else {
                assert_eq!(n, (2, 5, &"2-5"));
            };
        }

        for (_, n) in g.edges_directed(3, Direction::Incoming).enumerate() {
            assert_eq!(n, (2, 3, &"2-3"));
        }

        for (_, n) in g.edges_directed(4, Direction::Outcoming).enumerate() {
            assert_eq!(n, (4, 5, &"4-5"));
        }
    }
}
