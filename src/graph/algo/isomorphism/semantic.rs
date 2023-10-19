use crate::graph::{
    visit::{GraphBase, GraphDataAccess, IntoEdgeDirected},
    Direction,
};

pub trait NodeMatcher<G0: GraphBase, G1: GraphBase> {
    fn enabled() -> bool;

    fn eq(&mut self, g0: G0, g1: G1, node_id_0: G0::NodeId, node_id_1: G1::NodeId) -> bool;
}

pub trait EdgeMatcher<G0: GraphBase, G1: GraphBase> {
    fn enabled() -> bool;

    fn eq(
        &mut self,
        g0: G0,
        g1: G1,
        g0_nodes: (G0::NodeId, G0::NodeId),
        g1_nodes: (G1::NodeId, G1::NodeId),
    ) -> bool;
}

pub struct NoSemanticMatch;

impl<G0: GraphBase, G1: GraphBase> NodeMatcher<G0, G1> for NoSemanticMatch {
    fn enabled() -> bool {
        false
    }

    fn eq(
        &mut self,
        _g0: G0,
        _g1: G1,
        _node_id_0: <G0 as GraphBase>::NodeId,
        _node_id_1: <G1 as GraphBase>::NodeId,
    ) -> bool {
        false
    }
}

impl<G0: GraphBase, G1: GraphBase> EdgeMatcher<G0, G1> for NoSemanticMatch {
    fn enabled() -> bool {
        false
    }

    fn eq(
        &mut self,
        _g0: G0,
        _g1: G1,
        _: (G0::NodeId, G0::NodeId),
        _: (G1::NodeId, G1::NodeId),
    ) -> bool {
        false
    }
}

impl<G0, G1, F> NodeMatcher<G0, G1> for F
where
    G0: GraphDataAccess + IntoEdgeDirected,
    G1: GraphDataAccess + IntoEdgeDirected,
    F: FnMut(G0::NodeWeight, G1::NodeWeight) -> bool,
{
    fn enabled() -> bool {
        true
    }

    fn eq(
        &mut self,
        g0: G0,
        g1: G1,
        node_id_0: <G0 as GraphBase>::NodeId,
        node_id_1: <G1 as GraphBase>::NodeId,
    ) -> bool {
        if let (Some(n0), Some(n1)) = (
            G0::node_weight(g0, node_id_0),
            G1::node_weight(g1, node_id_1),
        ) {
            self(n0, n1)
        } else {
            false
        }
    }
}

impl<G0, G1, F> EdgeMatcher<G0, G1> for F
where
    G0: GraphDataAccess + IntoEdgeDirected + Copy,
    G1: GraphDataAccess + IntoEdgeDirected + Copy,
    F: FnMut(G0::EdgeWeight, G1::EdgeWeight) -> bool,
{
    fn enabled() -> bool {
        true
    }

    fn eq(
        &mut self,
        g0: G0,
        g1: G1,
        (g0_node_id_0, g0_node_id_1): (<G0 as GraphBase>::NodeId, <G0 as GraphBase>::NodeId),
        (g1_node_id_0, g1_node_id_1): (<G1 as GraphBase>::NodeId, <G1 as GraphBase>::NodeId),
    ) -> bool {
        let edge_0: Option<G0::EdgeId> =
            g0.edge_directed(g0_node_id_0, g0_node_id_1, Direction::Outcoming);
        let edge_1: Option<G1::EdgeId> =
            g1.edge_directed(g1_node_id_0, g1_node_id_1, Direction::Outcoming);

        if edge_0.is_none() && edge_1.is_none() {
            return false;
        }

        if let (Some(n0), Some(n1)) = (
            G0::edge_weight(g0, edge_0.unwrap()),
            G1::edge_weight(g1, edge_1.unwrap()),
        ) {
            self(n0, n1)
        } else {
            false
        }
    }
}
