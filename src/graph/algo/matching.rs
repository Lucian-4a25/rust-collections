use crate::{
    graph::visit::{
        EdgeRef, GraphBase, IntoEdges, IntoNeiborghbors, IntoNodeIdentifiers, NodeCount,
        NodeIndexable, VisitMap, Visitable,
    },
    vecdeque_cus::VecDeque,
};

/// compute graph matching using greedy heuristic method.
pub fn greedy_matching<G>(g: G) -> Matching<G>
where
    G: IntoNodeIdentifiers + Visitable + NodeIndexable + IntoNeiborghbors,
{
    let mut visited = g.visit_map();
    let mut mate = vec![None; g.node_bound()];
    let mut edges_num = 0usize;
    for node in g.node_identifiers() {
        let mut first = Some(node);
        dfs_non_backtracing(g, node, &mut visited, |another| {
            if let Some(first) = first.take() {
                mate[g.to_index(first)] = Some(g.to_index(another));
                mate[g.to_index(another)] = Some(g.to_index(first));
                edges_num += 1;
            } else {
                first = Some(another);
            }
        });
    }

    Matching {
        mate,
        edges_num,
        graph: g,
    }
}

#[inline]
fn dfs_non_backtracing<G, F>(g: G, node: G::NodeId, visited: &mut G::Map, mut add_mate: F)
where
    G: IntoNeiborghbors + Visitable,
    F: FnMut(G::NodeId),
{
    if visited.visit(node) {
        for neighbor in g.neighbors(node) {
            if visited.visit(neighbor) {
                add_mate(neighbor);
                dfs_non_backtracing(g, neighbor, visited, add_mate);
                break;
            }
        }
    }
}

/// compute graph maximum matching using Gabow's algorithm, https://dl.acm.org/doi/10.1145/321941.321942
pub fn maximum_matching<G>(g: G) -> Matching<G>
where
    G: IntoNeiborghbors + IntoNodeIdentifiers + NodeIndexable + IntoEdges + Visitable,
{
    // increase max len with 1 to place dummy value
    let max_len = g.node_bound() + 1;
    let dummy_idx = max_len - 1;
    // record mate of each node pair
    let mut mate: _ = vec![None; max_len];
    let mut edges_num = 0usize;
    let mut label = vec![Label::None; max_len];
    let mut first = vec![usize::MAX; max_len];
    let mut visited = g.visit_map();

    // E0: iterate all nodes in graph once
    for node in g.node_identifiers() {
        let start_node_idx = g.to_index(node);
        // E1: find unmatched node
        if mate[start_node_idx].is_none() {
            // try to find a augumention path for current node
            label[start_node_idx] = Label::First;
            first[start_node_idx] = dummy_idx;

            let mut queue = VecDeque::new();
            queue.push_back(start_node_idx);
            // E2: choose an edge, use BFS to iterate all outer label until find a unmatched node
            'outer: while let Some(outer_idx) = queue.pop_front() {
                for edge in g.edges(g.from_index(outer_idx)) {
                    let target = edge.target();
                    let target_idx = g.to_index(target);
                    // ignore self-loop edge
                    if target_idx == outer_idx {
                        continue;
                    }

                    // E3: y is unmatched node, a augumention path is found
                    if mate[target_idx].is_none() && target_idx != start_node_idx {
                        mate[target_idx] = Some(outer_idx);
                        augment_path(outer_idx, target_idx, &mut mate, &label);
                        edges_num += 1;

                        break 'outer;
                    }
                    // E4: y is a outer node, we need to assign edge label in function `l`, y maybe
                    // the start node
                    else if label[target_idx].is_outer() {
                        assign_edge_lable(
                            g,
                            edge.source(),
                            edge.target(),
                            &mut label,
                            &mate,
                            &mut first,
                            dummy_idx,
                        );
                    }
                    // E5: Assign a vertex label
                    else {
                        // it's impossible target node has no mate and it's not outer lable.
                        let target_mate_idx = mate[target_idx].unwrap();

                        if label[target_mate_idx].is_nonouter() {
                            label[target_mate_idx] = Label::Vertex(outer_idx);
                            first[target_mate_idx] = target_idx;
                            // add new outer label to queue if it's not visited
                            if visited.visit(g.from_index(target_mate_idx)) {
                                queue.push_back(target_mate_idx);
                            }
                        }
                    }
                }
            }

            // reset Label state
            for l in label.iter_mut() {
                *l = Label::None;
            }
        }
    }

    mate.pop();

    Matching {
        mate,
        edges_num,
        graph: g,
    }
}

/// assign edge label
fn assign_edge_lable<G>(
    graph: G,
    x: G::NodeId,
    y: G::NodeId,
    label: &mut Vec<Label>,
    mate: &Vec<Option<usize>>,
    first: &mut Vec<usize>,
    dummy_idx: usize,
) where
    G: NodeIndexable,
{
    let mut labeled = vec![];
    let (source, target) = (graph.to_index(x), graph.to_index(y));

    // L0: label r and s until r = s
    let mut r = first[source];
    let mut s = first[target];
    labeled.push(r);
    labeled.push(s);

    let join = loop {
        // L1: interchange r and s if s is not dummy
        if s != dummy_idx {
            std::mem::swap(&mut r, &mut s);
        }
        // L2: Next nonouter vertex.
        let r_mate = mate[r].unwrap();
        let next_outer = label[r_mate].try_next_outer().unwrap();
        r = first[next_outer];

        if !labeled.contains(&r) {
            labeled.push(r);
        } else {
            break r;
        }
    };

    // L3: Label vertices in P(x), P(y).
    for mut v in [first[source], first[target]] {
        // L4: if v != join, set Label(v) = n(xy), First(v) = join, v = next_nonouter
        while v != join {
            label[v] = Label::Edge(source, target);
            first[v] = join;
            let v_mate = mate[v].unwrap();
            // Do label[v_mate] must be Vertex label?
            // Yes, casue 'v' must be a nonouter node, which has a vertex label mate.
            let next_outer = label[v_mate].try_next_outer().unwrap();
            v = first[next_outer];
        }
    }

    // L5: update first
    for (vertex_idx, l) in label.iter().enumerate() {
        if l.is_outer() && label[first[vertex_idx]].is_outer() {
            first[vertex_idx] = join;
        }
    }
}

/// augment matching path
#[inline]
fn augment_path(outer: usize, target: usize, mate: &mut Vec<Option<usize>>, label: &Vec<Label>) {
    // R1: set t = MATE(v), MATE(v) = w, if MATE(t) != Some(outer), outer is the first Outer label, return
    let t = mate[outer];
    mate[outer] = Some(target);
    // R1: If MATE(t) != v, return. (This is the core of E algo.)
    // There are two cases:
    // 1. when outer has no mate, it mean we already reach the root node
    // 2. when t's (t is outer's mate) mate is not outer, it indicate t is already
    // matched with another node outside the circle, which means we don't need to check it again.
    if t != Some(outer) {
        return;
    }

    let t = t.unwrap();

    // R2: If v has a vertex label, set MATE(t) = LABEL(v), call R(LABEL(v), t)
    if let Some(next_outer) = label[outer].try_next_outer() {
        mate[t] = Some(next_outer);
        augment_path(next_outer, t, mate, label);
        return;
    }

    // R3:  (Vertex v has an edge label ) Set x, y to vertices so LABEL(v) = n(xy),
    // call R(x, y) recurslvely, call R(y, x) recurslvely
    // Q: Did this really do the correct augment?
    if let Some((x, y)) = label[outer].try_edge_outer() {
        augment_path(x, y, mate, label);
        augment_path(y, x, mate, label);
        return;
    }

    panic!("Unexpected label when augmenting path");
}

#[derive(Clone, Copy)]
enum Label {
    None,
    First,
    Vertex(usize),
    Edge(usize, usize),
    NonOuter,
}

impl Label {
    fn is_outer(self) -> bool {
        match self {
            Label::Edge(_, _) | Label::First | Label::Vertex(_) => true,
            _ => false,
        }
    }

    fn is_nonouter(self) -> bool {
        !Self::is_outer(self)
    }

    fn try_next_outer(self) -> Option<usize> {
        match self {
            Label::Vertex(v) => Some(v),
            _ => None,
        }
    }

    fn try_edge_outer(self) -> Option<(usize, usize)> {
        match self {
            Label::Edge(a, b) => Some((a, b)),
            _ => None,
        }
    }
}

pub struct Matching<G>
where
    G: GraphBase,
{
    graph: G,
    mate: Vec<Option<usize>>,
    edges_num: usize,
}

impl<G> Matching<G>
where
    G: NodeCount + GraphBase + Copy,
{
    pub fn is_perfect(&self) -> bool {
        let node_count = self.graph.node_count();
        node_count % 2 == 0 && node_count / 2 == self.edges_num
    }
}

impl<G> Matching<G>
where
    G: NodeIndexable + Copy,
{
    pub fn mate(&self, node: G::NodeId) -> Option<G::NodeId> {
        self.mate
            .get(self.graph.to_index(node))
            .and_then(|&n| n.map(|n_id| self.graph.from_index(n_id)))
        // .map_or(None, |v| v.map_or(None, |n| Some(self.graph.from_index(n))))
    }

    pub fn edges(&self) -> MatchedEdges<'_, G> {
        MatchedEdges {
            graph: self.graph,
            pos: 0,
            mate: self.mate.iter(),
        }
    }

    pub fn nodes(&self) -> MatchedNodes<'_, G> {
        MatchedNodes {
            graph: self.graph,
            mate: self.mate.iter(),
            pos: 0,
        }
    }

    pub fn contains_edge(&self, n: G::NodeId, m: G::NodeId) -> bool {
        match self.mate[self.graph.to_index(n)] {
            Some(im) => im == self.graph.to_index(m),
            None => false,
        }
    }

    pub fn contains_node(&self, n: G::NodeId) -> bool {
        self.mate[self.graph.to_index(n)].is_some()
    }

    pub fn len(&self) -> usize {
        self.edges_num
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct MatchedEdges<'a, G>
where
    G: NodeIndexable,
{
    graph: G,
    mate: std::slice::Iter<'a, Option<usize>>,
    pos: usize,
}

impl<'a, G: NodeIndexable> Iterator for MatchedEdges<'a, G> {
    type Item = (G::NodeId, G::NodeId);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(m_op) = self.mate.next() {
            if let Some(m) = m_op {
                // ignore repeated result
                let result = if *m > self.pos {
                    Some((self.graph.from_index(self.pos), self.graph.from_index(*m)))
                } else {
                    None
                };
                self.pos += 1;

                if result.is_some() {
                    return result;
                }
            } else {
                self.pos += 1;
            }
        }

        None
    }
}

pub struct MatchedNodes<'a, G: NodeIndexable> {
    graph: G,
    mate: std::slice::Iter<'a, Option<usize>>,
    pos: usize,
}

impl<'a, G: NodeIndexable> Iterator for MatchedNodes<'a, G> {
    type Item = G::NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(m_op) = self.mate.next() {
            if m_op.is_some() {
                let res = self.graph.from_index(self.pos);
                self.pos += 1;
                return Some(res);
            }
            self.pos += 1;
        }

        None
    }
}
