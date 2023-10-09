pub use common::*;

mod algo;
mod common;
mod compressed_sparse_row;
mod graph_adjacency_list;
mod graph_dense_matrix;
mod graph_sparse_dok;
mod stable_vec;
mod visit;

pub trait IntoWeightedEdge<E> {
    type NodeId;

    fn into_weighted_edge(self) -> (Self::NodeId, Self::NodeId, E);
}

impl<N, E> IntoWeightedEdge<E> for (N, N, E) {
    type NodeId = N;

    fn into_weighted_edge(self) -> (Self::NodeId, Self::NodeId, E) {
        self
    }
}

impl<N, E> IntoWeightedEdge<E> for (N, N)
where
    E: Default,
{
    type NodeId = N;

    fn into_weighted_edge(self) -> (Self::NodeId, Self::NodeId, E) {
        (self.0, self.1, E::default())
    }
}

// See: https://doc.rust-lang.org/error_codes/E0207.html
