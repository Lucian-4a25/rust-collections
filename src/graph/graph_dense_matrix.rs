use super::{stable_vec::StableVec, Directed, Direction, GraphType, UnDirected};
use crate::vecdeque_cus::VecDeque;
use std::marker::PhantomData;

struct MatrixGraph<N, E, T: GraphType> {
    nodes: StableVec<N>,
    /// a flat array format of matrix,
    /// others format may be considered, such as Vec<Vec<E>>, but more space will be wasted
    /// and the memory layout is not consective, maybe not so cache friendly.
    /// NB: Vec<Vec<E>> is a implemention of sparse matrix, not dense matrix implemention
    edges: Vec<Option<E>>,
    edge_count: usize,
    phantomdata: PhantomData<T>,
}

/// return the edge pos from a to b
fn get_edge_pos(row: usize, column: usize, width: usize, directed: bool) -> usize {
    if directed {
        row * width + column
    } else {
        let (r, c) = if row < column {
            (column, row)
        } else {
            (row, column)
        };
        // 等差数列求和
        (r + 1) * r / 2 + c
    }
}

#[allow(dead_code)]
impl<N, E> MatrixGraph<N, E, Directed> {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
            edge_count: 0,
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E> MatrixGraph<N, E, UnDirected> {
    pub fn new_undirected() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
            edge_count: 0,
            phantomdata: PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<N, E, T: GraphType> MatrixGraph<N, E, T> {
    pub fn is_directed(&self) -> bool {
        T::is_directed()
    }

    pub fn add_node(&mut self, n: N) -> usize {
        let old_group_width = self.group_width();
        let inserted_pos = self.nodes.insert(n);
        // insert into a removed place, we don't need to move or realloc
        if inserted_pos >= old_group_width {
            self.reserve(old_group_width, 1);
            // println!(
            //     "do reserve in edges: cap {}, len {}",
            //     self.edges.capacity(),
            //     self.edges.len()
            // );
        }
        inserted_pos
    }

    pub fn remove_node(&mut self, n: usize) -> Option<N> {
        let removed_node = self.nodes.remove(n);
        // we need to delete all edges of n
        if removed_node.is_some() {
            let group_width = self.group_width();
            let directed = self.is_directed();
            // remove all outcoming edges from n
            let start_edge_from = get_edge_pos(n, 0, group_width, directed);
            let end_outcoming_pos = if directed { group_width } else { n + 1 };
            for i in 0..end_outcoming_pos {
                let e = self.edges[start_edge_from + i].take();
                // println!("removed pos: {}", start_edge_from + i);
                if e.is_some() {
                    self.edge_count -= 1;
                }
            }

            // remove all incoming edge to n
            let start_pos = if directed { 0 } else { n + 1 };
            for i in start_pos..group_width {
                if directed && i == n {
                    continue;
                }
                let e = self.edges[get_edge_pos(i, n, group_width, directed)].take();
                if e.is_some() {
                    self.edge_count -= 1;
                }
            }
        }

        removed_node
    }

    /// add an edge into the edges, return Option<E> if there is already an existing edge
    pub fn add_edge(&mut self, a: usize, b: usize, e: E) -> Option<E> {
        let m = std::cmp::max(a, b);
        if m >= self.group_width() {
            panic!("node index out of bound");
        }
        let width = self.group_width();
        let directed = self.is_directed();
        let old_edge = self.edges[get_edge_pos(a, b, width, directed)].replace(e);
        if old_edge.is_none() {
            self.edge_count += 1;
        }
        old_edge
    }

    /// make more space for addition if neccessary
    pub fn reserve(&mut self, old_group_width: usize, addition: usize) {
        let new_node_num = old_group_width + addition;
        let directed = self.is_directed();
        let needed_space = if directed {
            new_node_num * new_node_num
        } else {
            (new_node_num + 1) * new_node_num / 2
        };

        // we may need more memory space
        // println!("needed space: {needed_space}");
        self.edges.resize_with(needed_space, || None);
        // we need to move elements to fit the new position
        if T::is_directed() {
            unsafe {
                let p = self.edges.as_mut_ptr();
                for i in (1..new_node_num - 1).rev() {
                    let old_group_start = p.add(i * (new_node_num - 1));
                    let new_group_start = p.add(i * new_node_num);
                    if i >= old_group_width {
                        std::ptr::swap_nonoverlapping(
                            old_group_start,
                            new_group_start,
                            old_group_width,
                        );
                    } else {
                        for i in 0..old_group_width {
                            std::ptr::swap(old_group_start.add(i), new_group_start.add(i));
                        }
                    }
                }
            }
        }
    }

    pub fn remove_edge(&mut self, a: usize, b: usize) -> Option<E> {
        let edge_pos = get_edge_pos(a, b, self.group_width(), self.is_directed());
        let removed_edge = self.edges[edge_pos].take();
        if removed_edge.is_some() {
            self.edge_count -= 1;
        }
        removed_edge
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.edge_count = 0;
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn node_count(&self) -> usize {
        self.nodes.node_count()
    }

    /// current length of matrix
    fn group_width(&self) -> usize {
        self.nodes.max_pos()
    }

    fn contains_node(&self, n: usize) -> bool {
        self.nodes.contains(n)
    }

    /// return a iterator of all edges of a
    /// - Outcoming edges for Directed Graph
    /// - All edges for Undirected Graph
    pub fn edges(&self, a: usize) -> Edges<'_, E> {
        let mut idxs = VecDeque::new();
        for i in 0..self.group_width() {
            if self.contains_node(i) {
                idxs.push_back((a, i));
            }
        }

        Edges {
            edges: &self.edges,
            idxs,
            width: self.group_width(),
            directed: self.is_directed(),
        }
    }

    pub fn edges_directed(&self, a: usize, d: Direction) -> Edges<'_, E> {
        let mut idxs = VecDeque::new();
        if matches!(d, Direction::Outcoming) || !self.is_directed() {
            for i in 0..self.group_width() {
                if self.contains_node(i) {
                    idxs.push_back((a, i));
                }
            }
        } else {
            for i in 0..self.group_width() {
                if self.contains_node(i) {
                    idxs.push_back((i, a));
                }
            }
        }
        Edges {
            idxs,
            edges: &self.edges,
            width: self.group_width(),
            directed: self.is_directed(),
        }
    }

    /// return all neighbor nodes of a
    pub fn neighbors(&self, a: usize) -> Neighbors<'_, E> {
        let mut idxs = VecDeque::new();
        for i in 0..self.group_width() {
            if self.contains_node(i) {
                idxs.push_back((a, i));
            }
        }

        Neighbors {
            start: a,
            edges: &self.edges,
            directed: self.is_directed(),
            width: self.group_width(),
            idxs,
        }
    }

    pub fn neighbors_directed(&self, a: usize, d: Direction) -> Neighbors<'_, E> {
        let mut idxs = VecDeque::new();
        if matches!(d, Direction::Outcoming) || !self.is_directed() {
            for i in 0..self.group_width() {
                if self.contains_node(i) {
                    idxs.push_back((a, i));
                }
            }
        } else {
            for i in 0..self.group_width() {
                if self.contains_node(i) {
                    idxs.push_back((i, a));
                }
            }
        }

        Neighbors {
            directed: self.is_directed(),
            idxs,
            start: a,
            width: self.group_width(),
            edges: &self.edges,
        }
    }
}

pub struct Edges<'a, E> {
    width: usize,
    directed: bool,
    edges: &'a [Option<E>],
    idxs: VecDeque<(usize, usize)>,
}

impl<'a, E> Iterator for Edges<'a, E> {
    type Item = (usize, usize, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((row, col)) = self.idxs.pop_front() {
            let pos = get_edge_pos(row, col, self.width, self.directed);
            // println!("pos in edge: {}", pos);
            if let Some(edge) = self.edges[pos].as_ref() {
                return Some((row, col, edge));
            }
        }
        None
    }
}

pub struct Neighbors<'a, E> {
    start: usize,
    edges: &'a [Option<E>],
    idxs: VecDeque<(usize, usize)>,
    width: usize,
    directed: bool,
}

impl<'a, E> Iterator for Neighbors<'a, E> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((row, col)) = self.idxs.pop_front() {
            let pos = get_edge_pos(row, col, self.width, self.directed);
            if let Some(_) = self.edges[pos].as_ref() {
                return Some(if col != self.start { col } else { row });
            }
        }
        None
    }
}

#[cfg(test)]
mod test_matrix_graph {
    use super::*;

    #[test]
    fn test_basic_directed() {
        let mut graph = MatrixGraph::new();

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

        let iter = graph.neighbors(3);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, 4);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.neighbors(1);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(
                neighbor_id,
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

        let iter: Neighbors<'_, i32> = graph.neighbors_directed(3, Direction::Incoming);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, if counter == 0 { 0 } else { 2 });
            counter += 1;
        }
        assert_eq!(counter, 2);

        let iter: Neighbors<'_, i32> = graph.neighbors_directed(4, Direction::Incoming);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, if counter == 0 { 1 } else { 3 });
            counter += 1;
        }
        assert_eq!(counter, 2);

        let iter = graph.edges(1);
        let mut counter = 0;
        for (offset, e) in iter.enumerate() {
            assert_eq!(
                if offset == 0 {
                    1
                } else if offset == 1 {
                    5
                } else {
                    6
                },
                *e.2
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        let iter = graph.edges_directed(4, Direction::Incoming);
        let mut counter = 0;
        for (offset, e) in iter.enumerate() {
            assert_eq!(if offset == 0 { 5 } else { 3 }, *e.2);
            counter += 1;
        }
        assert_eq!(counter, 2);

        // --------------- check after remove ---------------
        assert_eq!(graph.remove_node(2), Some("Faraday"));

        let iter = graph.neighbors(1);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, if counter == 0 { 4 } else { 5 });
            counter += 1;
        }
        assert_eq!(counter, 2);

        let iter: Neighbors<'_, i32> = graph.neighbors_directed(3, Direction::Incoming);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, 0);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.edges(1);
        let mut counter = 0;
        for (offset, e) in iter.enumerate() {
            assert_eq!(if offset == 0 { 5 } else { 6 }, *e.2);
            counter += 1;
        }
        assert_eq!(counter, 2);

        let iter = graph.edges_directed(3, Direction::Incoming);
        let mut counter = 0;
        for e in iter {
            assert_eq!(4, *e.2);
            counter += 1;
        }
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_basic_undirected() {
        let mut graph = MatrixGraph::new_undirected();

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

        let iter = graph.neighbors(3);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(
                neighbor_id,
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

        let iter = graph.neighbors_directed(1, Direction::Outcoming);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(
                neighbor_id,
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

        let iter = graph.edges(3);
        let mut counter = 0;
        for edge in iter {
            // println!("edge: {}", edge.2);
            assert_eq!(
                if counter == 0 {
                    4
                } else if counter == 1 {
                    2
                } else {
                    3
                },
                *edge.2
            );
            counter += 1;
        }
        assert_eq!(counter, 3);

        let iter = graph.edges_directed(2, Direction::Incoming);
        let mut counter = 0;
        for edge in iter {
            assert_eq!(if counter == 0 { 1 } else { 2 }, *edge.2);
            counter += 1;
        }
        assert_eq!(counter, 2);

        // ---------------- remove element to check ---------------
        assert_eq!(graph.remove_node(3), Some("Maxwell"));
        assert_eq!(graph.edge_count(), 4);
        let iter = graph.neighbors(2);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, 1);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.neighbors_directed(0, Direction::Outcoming);
        let mut counter = 0;
        for neighbor_id in iter {
            assert_eq!(neighbor_id, 1);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.edges(0);
        let mut counter = 0;
        for edge in iter {
            assert_eq!(7, *edge.2);
            counter += 1;
        }
        assert_eq!(counter, 1);

        let iter = graph.edges_directed(1, Direction::Incoming);
        let mut counter = 0;
        for edge in iter {
            assert_eq!(
                if counter == 0 {
                    7
                } else if counter == 1 {
                    1
                } else if counter == 2 {
                    5
                } else {
                    6
                },
                *edge.2
            );
            counter += 1;
        }
        assert_eq!(counter, 4);

        let iter = graph.edges_directed(4, Direction::Incoming);
        let mut counter = 0;
        for edge in iter {
            assert_eq!(5, *edge.2);
            counter += 1;
        }
        assert_eq!(counter, 1);
    }
}
