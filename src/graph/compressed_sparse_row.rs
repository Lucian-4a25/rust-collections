use super::{Directed, GraphType, UnDirected};
use std::{cmp::Ordering, marker::PhantomData, ops::Range};

/// The compressed sparsed row implemention of graph
/// https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
/// NB: for csr graoh, the undirected graph mean we need to insert two edges in the edges, so that we could get the
/// edges for both of nodes; but for directed map, we only need to insert one edge, so there is no way to know a node's incoming edges.
pub struct Csr<N, E, T: GraphType> {
    nodes: Vec<N>,
    // Q: if we could store a (Edge, Direction) in the edges, so that we could know all edges of specified node,
    // and for undirected graph, if we want to know edge, we could sort the idx, and only store the (large, small) position
    // A: if we store two edges for directed graph, then we have to shift the cols and edges every time we insert. In order to
    // ensure the insertion performance, we only could insert at the end of edges and cols in most of cases. Maybe this is the cost
    // of consecutive edges for node. That's why we insert one edge for directed graph.
    edges: Vec<E>,
    cols: Vec<usize>,
    edge_count: usize,
    /// the number of edges for nodes
    rows: Vec<usize>,
    phantomdata: PhantomData<T>,
}

static BINARY_SEARCH_SIZE: usize = 16;

#[allow(dead_code)]
impl<N, E> Csr<N, E, Directed> {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
            cols: Default::default(),
            edge_count: 0,
            rows: vec![0],
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E> Csr<N, E, UnDirected> {
    pub fn new_undirected() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
            cols: Default::default(),
            edge_count: 0,
            rows: vec![0],
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E, T: GraphType> Csr<N, E, T> {
    pub fn is_directed(&self) -> bool {
        T::is_directed()
    }

    pub fn add_node(&mut self, n: N) -> usize {
        let pos = self.nodes.len();
        self.nodes.push(n);
        self.rows.push(self.cols.len());
        pos
    }

    pub fn add_edge(&mut self, a: usize, b: usize, e: E) -> Option<E>
    where
        E: Clone,
    {
        let old_v = self.insert_single_edge(a, b, e.clone());
        // for undirected graph, we need to insert in the oppsite edge
        if !self.is_directed() && a != b {
            self.insert_single_edge(b, a, e);
        }

        old_v
    }

    fn insert_single_edge(&mut self, a: usize, b: usize, e: E) -> Option<E>
    where
        E: Clone,
    {
        let range = self.get_node_range(a);
        let pos = self.find_insertion_pos(range, b);
        let mut old_v = None;
        match pos {
            Ok(pos) => {
                self.cols.insert(pos, b);
                self.edges.insert(pos, e);
            }
            Err(pos) => {
                old_v = Some(std::mem::replace(&mut self.edges[pos], e));
            }
        }

        for r in &mut self.rows[a + 1..] {
            *r += 1;
        }

        old_v
    }

    /// find the insertion position for edge, return Err(usize) if the edge already exists
    fn find_insertion_pos(&self, range: Range<usize>, b: usize) -> Result<usize, usize> {
        let end = range.end;
        // we could use binary search in a ordered array
        if range.size_hint().0 < BINARY_SEARCH_SIZE {
            for i in range {
                match self.cols[i].cmp(&b) {
                    Ordering::Equal => {
                        return Err(i);
                    }
                    Ordering::Greater => {
                        return Ok(i);
                    }
                    Ordering::Less => {
                        continue;
                    }
                }
            }
            return Ok(end);
        } else {
            match self.cols[range].binary_search(&b) {
                Ok(r) => Err(r),
                Err(e) => Ok(e),
            }
        }
    }

    fn get_node_range(&self, n: usize) -> Range<usize> {
        self.rows[n]..self.rows[n + 1]
    }

    pub fn clear_edges(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.cols.clear();
        self.rows = vec![0; 1];
        self.edge_count = 0;
    }

    pub fn contains_edge(&self, a: usize, b: usize) -> bool {
        let range = self.get_node_range(a);
        self.find_insertion_pos(range, b).is_err()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        if self.is_directed() {
            self.cols.len()
        } else {
            self.edge_count
        }
    }

    /// return a iterator of all edges of a
    pub fn edges(&self, n: usize) -> Edges<E> {
        Edges {
            range: self.get_node_range(n),
            edges: &self.edges,
        }
    }

    pub fn neighbors(&self, n: usize) -> Neighbors<'_> {
        Neighbors {
            range: self.get_node_range(n),
            nodes: &self.cols,
        }
    }
}

pub struct Edges<'a, E> {
    range: Range<usize>,
    edges: &'a [E],
}

impl<'a, E> Iterator for Edges<'a, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.range.next() {
            Some(&self.edges[idx])
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

pub struct Neighbors<'a> {
    range: Range<usize>,
    nodes: &'a [usize],
}

impl<'a> Iterator for Neighbors<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.range.next() {
            Some(self.nodes[idx])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test_csr {
    use super::*;

    #[test]
    fn test_basic_directed() {
        let mut graph: _ = Csr::new();
        let famous_physicist = [
            "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
        ];
        for name in famous_physicist.clone() {
            graph.add_node(name);
        }

        graph.add_edge(0, 1, 7);
        graph.add_edge(1, 2, 1);
        graph.add_edge(2, 3, 2);
        graph.add_edge(3, 4, 3);
        graph.add_edge(0, 3, 4);
        graph.add_edge(1, 4, 5);
        graph.add_edge(1, 5, 6);

        // contains_edge api
        assert!(graph.contains_edge(2, 3));
        assert!(graph.contains_edge(1, 5));

        // edges api
        let mut counter = 0;
        for e in graph.edges(0) {
            assert_eq!(*e, if counter == 0 { 7 } else { 4 });
            counter += 1;
        }
        assert_eq!(counter, 2);

        let mut counter = 0;
        for e in graph.edges(1) {
            assert_eq!(
                *e,
                if counter == 0 {
                    1
                } else if counter == 1 {
                    5
                } else {
                    6
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        // test neighbors api
        let mut counter = 0;
        for i in graph.neighbors(1) {
            assert_eq!(
                i,
                if counter == 0 {
                    2
                } else if counter == 1 {
                    4
                } else {
                    5
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        let mut counter = 0;
        for i in graph.neighbors(3) {
            assert_eq!(i, 4);
            counter += 1;
        }
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_basic_undirected() {
        let mut graph: _ = Csr::new_undirected();
        let famous_physicist = [
            "Galileo", "Newton", "Faraday", "Maxwell", "Einstein", "Planck",
        ];
        for name in famous_physicist.clone() {
            graph.add_node(name);
        }

        graph.add_edge(0, 1, 7);
        graph.add_edge(1, 2, 1);
        graph.add_edge(2, 3, 2);
        graph.add_edge(3, 4, 3);
        graph.add_edge(0, 3, 4);
        graph.add_edge(1, 4, 5);
        graph.add_edge(1, 5, 6);

        // contains_edge api
        assert!(graph.contains_edge(2, 3));
        assert!(graph.contains_edge(1, 5));

        // edges api
        let mut counter = 0;
        for e in graph.edges(0) {
            assert_eq!(*e, if counter == 0 { 7 } else { 4 });
            counter += 1;
        }
        assert_eq!(counter, 2);

        let mut counter = 0;
        for e in graph.edges(1) {
            assert_eq!(
                *e,
                if counter == 0 {
                    7
                } else if counter == 1 {
                    1
                } else if counter == 2 {
                    5
                } else {
                    6
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 4);

        // test neighbors api
        let mut counter = 0;
        for i in graph.neighbors(1) {
            assert_eq!(
                i,
                if counter == 0 {
                    0
                } else if counter == 1 {
                    2
                } else if counter == 2 {
                    4
                } else {
                    5
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 4);

        let mut counter = 0;
        for i in graph.neighbors(3) {
            assert_eq!(
                i,
                if counter == 0 {
                    0
                } else if counter == 1 {
                    2
                } else {
                    4
                }
            );
            counter += 1;
        }
        assert_eq!(counter, 3);
    }
}
