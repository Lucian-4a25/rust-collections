// implement a simple version of b-tree map
use std::{
    alloc::Layout, cmp::Ordering, collections::VecDeque, fmt::Debug, mem::MaybeUninit, ops::Range,
    ptr::NonNull,
};

const B: usize = 2;
const CAPACITY: usize = 2 * B - 1;
const MIN_CAP: usize = B - 1;
const MAX_EDGES: usize = 2 * B;

// Split related.
const EDGE_IDX_LEFT_OF_CENTER: usize = B - 1;
const KV_IDX_CENTER: usize = B - 1;
const EDGE_IDX_RIGHT_OF_CENTER: usize = B;

// A node is root node only when it's the value of field of root.
pub struct BTreeMap<K, V>
where
    K: Ord + Debug,
{
    root: Option<NonNull<Node<K, V>>>,
    len: usize,
}

struct Node<K, V> {
    node_type: NodeType,
    parent: Option<NonNull<Node<K, V>>>,
    // the index in the parent edges slice, which is one greater than the index in the keys slice.
    parent_idx: Option<usize>,
    // current length of bucket
    len: usize,
    keys: [MaybeUninit<K>; CAPACITY],
    vals: [MaybeUninit<V>; CAPACITY],
    // TODO: Option type will lead to use double memory space of [NonNull<Node<K, V>>; MAX_EDGES],
    // maybe could use better data structure to optimize
    edges: Option<[NonNull<Node<K, V>>; MAX_EDGES]>,
}

enum NodeType {
    InternalNode,
    Leaf,
}

/// When the bucket is full, we must split current node.
struct SplitResult<K, V> {
    kv: (K, V),
    split_node: NonNull<Node<K, V>>,
}

enum LeftOrRight {
    Left(usize),
    Right(usize),
}

/// Given an edge index where we want to insert into a node filled to capacity,
/// computes a sensible KV index of a split point and where to perform the insertion.
/// The goal of the split point is for its key and value to end up in a parent node;
/// the keys, values and edges to the left of the split point become the left child;
/// the keys, values and edges to the right of the split point become the right child.
fn splitpoint(edge_idx: usize) -> (usize, LeftOrRight) {
    debug_assert!(edge_idx <= CAPACITY);
    // Rust issue #74834 tries to explain these symmetric rules.
    match edge_idx {
        0..EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER - 1, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_LEFT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Left(edge_idx)),
        EDGE_IDX_RIGHT_OF_CENTER => (KV_IDX_CENTER, LeftOrRight::Right(0)),
        _ => (
            KV_IDX_CENTER + 1,
            LeftOrRight::Right(edge_idx - (KV_IDX_CENTER + 1 + 1)),
        ),
    }
}

enum SiblingNode<K, V> {
    Left(NonNull<Node<K, V>>),
    Right(NonNull<Node<K, V>>),
}

#[allow(dead_code)]
impl<K, V> Node<K, V>
where
    K: Ord,
{
    fn new_leaf() -> NonNull<Self> {
        unsafe {
            NonNull::from(Box::leak(Box::new(Self {
                node_type: NodeType::Leaf,
                parent: None,
                parent_idx: None,
                len: 0,
                keys: MaybeUninit::uninit().assume_init(),
                vals: MaybeUninit::uninit().assume_init(),
                edges: None,
            })))
        }
    }

    fn new_internal_node() -> NonNull<Self> {
        unsafe {
            NonNull::from(Box::leak(Box::new(Self {
                node_type: NodeType::InternalNode,
                parent: None,
                parent_idx: None,
                len: 0,
                keys: MaybeUninit::uninit().assume_init(),
                vals: MaybeUninit::uninit().assume_init(),
                edges: Some([NonNull::dangling(); MAX_EDGES]),
            })))
        }
    }

    fn get_keys(&self, index: Range<usize>) -> &[K] {
        unsafe { MaybeUninit::slice_assume_init_ref(self.keys.get_unchecked(index)) }
    }

    fn get_values(&self, index: Range<usize>) -> &[V] {
        unsafe { MaybeUninit::slice_assume_init_ref(self.vals.get_unchecked(index)) }
    }

    fn get_value_ref(&self, idx: usize) -> &V {
        unsafe { &(*self.vals[idx].as_ptr()) }
    }

    fn get_key_ref(&self, idx: usize) -> &K {
        unsafe { &(*self.keys[idx].as_ptr()) }
    }

    /// Assume there is enough space to insert
    fn insert_kv_edge_fit(
        mut this: NonNull<Node<K, V>>,
        k: K,
        v: V,
        idx: usize,
        edge: Option<NonNull<Node<K, V>>>,
    ) {
        unsafe {
            let this_mut = this.as_mut();
            let old_len = this_mut.len;
            if matches!(edge, Some(_)) && matches!(this_mut.node_type, NodeType::Leaf) {
                panic!("insert a edge into a leaf node, which is a impossible behavior.");
            }
            slice_shr(&mut this_mut.keys[idx..old_len], 1);
            slice_shr(&mut this_mut.vals[idx..old_len], 1);
            this_mut.keys[idx].write(k);
            this_mut.vals[idx].write(v);

            // If we insert a value into the internal node, there must be with a edge.
            if let Some(ref mut edges) = this_mut.edges {
                edges_shr(&mut edges[idx + 1..old_len + 1], 1);
                let mut appended_edge = edge.expect("insert into a internal data without edge");
                appended_edge.as_mut().parent = Some(NonNull::from(this));
                edges[idx + 1] = appended_edge;
                // update edge's parent idx
                for (offset, edge) in edges[idx + 1..old_len + 2].iter_mut().enumerate() {
                    edge.as_mut().parent_idx = Some(idx + 1 + offset);
                }
            }

            this_mut.len = old_len + 1;
        }
    }

    /// Insert a kv pair with a possible edge into the Node, if the Node's capacity is full, then split the node.
    /// The idx is the position where the new kv pair will insert.
    fn insert(
        mut this: NonNull<Node<K, V>>,
        k: K,
        v: V,
        idx: usize,
        edge: Option<NonNull<Node<K, V>>>,
    ) -> Option<SplitResult<K, V>> {
        unsafe {
            let this_mut = this.as_mut();
            if this_mut.len < CAPACITY {
                Node::insert_kv_edge_fit(this, k, v, idx, edge);
                None
            } else {
                let (split_point, insertion_point) = splitpoint(idx);
                let (kv, split_node) = this_mut.split_from(split_point);
                match insertion_point {
                    LeftOrRight::Left(pos) => {
                        Node::insert_kv_edge_fit(this, k, v, pos, edge);
                    }
                    LeftOrRight::Right(pos) => {
                        Node::insert_kv_edge_fit(split_node, k, v, pos, edge);
                    }
                }

                Some(SplitResult { kv, split_node })
            }
        }
    }

    /// Split current Node, from the specified idx position, the kv pair in the split point will pop up to the last level.
    /// The right of the split point will be placed to a new leaf Node, as return value with the poped kv pair.
    fn split_from(&mut self, idx: usize) -> ((K, V), NonNull<Node<K, V>>) {
        let mut new_node = Node::new_leaf();
        let left_old_len = self.len;
        let left_new_len = idx;

        // println!("split from idx: {idx}, current len is: {left_old_len}");

        unsafe {
            let mut new_node_mut = new_node.as_mut();
            let right_node_len = left_old_len - idx - 1;
            let k = self.keys[idx].as_mut_ptr().read();
            let v = self.vals[idx].as_mut_ptr().read();
            // When B equals 2, it's possible the right part of new node is empty, then we don't
            // need to move the keys and vals
            if idx + 1 < CAPACITY {
                move_to_slice(
                    &self.keys[idx + 1],
                    &mut new_node_mut.keys[0],
                    left_old_len - idx - 1,
                );
                move_to_slice(
                    &self.vals[idx + 1],
                    &mut new_node_mut.vals[0],
                    left_old_len - idx - 1,
                );
            }
            self.len = left_new_len;

            // If current node is a InternalNode, we need to copy the right part of edges to the new Node.
            if let NodeType::InternalNode = self.node_type {
                if let Some(edges) = self.edges {
                    let mut new_edges = [NonNull::dangling(); MAX_EDGES];
                    move_to_edge(&edges[idx + 1], &mut new_edges[0], left_old_len - idx);
                    new_node_mut.edges = Some(new_edges);
                    new_node_mut.node_type = NodeType::InternalNode;
                    // we need to update the new internal node's edge parent_idx & parent
                    for offset in 0..right_node_len + 1 {
                        new_edges[offset].as_mut().parent_idx = Some(offset);
                        new_edges[offset].as_mut().parent = Some(new_node);
                    }
                }
            }

            new_node_mut.len = right_node_len;

            ((k, v), new_node)
        }
    }

    /// Remove elememnt from current node, if there is underlying value, return it.
    /// If current capacity is not enough,
    /// * When self is a leaf node, then steal from a sibling node OR merge with a sibling node.
    /// * When self is a internal node, then try to move the largest value from left-sub tree or the smallest value from
    /// right-sub tree to fit into the removed position. Then check if the leaf node we just removed has the min capacity or not.
    fn remove<F: FnOnce()>(
        mut this: NonNull<Node<K, V>>,
        idx: usize,
        handle_emptied_internal_root: F,
    ) -> V {
        unsafe {
            let this_mut = this.as_mut();
            let is_leaf_node = matches!(this_mut.node_type, NodeType::Leaf);

            if is_leaf_node {
                let old_len = this_mut.len;
                let removed_val = this_mut.vals[idx].as_mut_ptr().read();
                slice_shl(&mut this_mut.keys[idx + 1..old_len], 1);
                slice_shl(&mut this_mut.vals[idx + 1..old_len], 1);
                this_mut.len -= 1;

                if old_len <= MIN_CAP {
                    if Node::steal_or_merge_with_sibling_recursing(this) {
                        handle_emptied_internal_root();
                    }
                }
                return removed_val;
            } else {
                let mut replaced_node = this_mut.find_largest_node_in_left_tree(idx);

                // This node must be a leaf node, cause for a complete b-tree, every internal node must have underlying edge.
                let replaced_node_ptr = replaced_node.as_mut();
                let removed_idx = replaced_node_ptr.len - 1;
                let replaced_kv = (
                    replaced_node_ptr.keys[removed_idx].as_ptr().read(),
                    replaced_node_ptr.vals[removed_idx].as_ptr().read(),
                );
                replaced_node_ptr.len -= 1;

                let removed_val = this_mut.vals[idx].as_mut_ptr().read();
                // Write the largest value in left-sub tree into removed position.
                this_mut.keys[idx].write(replaced_kv.0);
                this_mut.vals[idx].write(replaced_kv.1);

                if replaced_node_ptr.len < MIN_CAP {
                    if Node::steal_or_merge_with_sibling_recursing(replaced_node) {
                        handle_emptied_internal_root();
                    }
                }

                return removed_val;
            }
        }
    }

    // Remove the kv in the idx position, if there is a underlying edge, remove it too.
    fn remove_kv_fit(&mut self, idx: usize) -> (K, V) {
        unsafe {
            let old_len = self.len;
            let removed_kv = (
                self.keys[idx].as_ptr().read(),
                self.vals[idx].as_ptr().read(),
            );
            // println!("removed kv fit idx: {idx}, current len: {}", old_len);
            slice_shl(&mut self.keys[idx + 1..old_len], 1);
            slice_shl(&mut self.vals[idx + 1..old_len], 1);

            if let Some(edges) = self.edges.as_mut() {
                edges_shl(&mut edges[idx + 1..old_len + 1], 1);
                // update edges parent_idx
                for (offset, edge) in edges[idx..old_len].iter_mut().enumerate() {
                    edge.as_mut().parent_idx = Some(offset + idx);
                }
            }

            self.len = old_len - 1;

            removed_kv
        }
    }

    /// Find the node with lagest value in left tree.
    fn find_largest_node_in_left_tree(&mut self, idx: usize) -> NonNull<Node<K, V>> {
        unsafe {
            let mut node = self.edges.expect("empty left tree edge")[idx];
            loop {
                let node_ptr = node.as_mut();
                if matches!(node_ptr.node_type, NodeType::Leaf) {
                    return node;
                }
                node = node_ptr.edges.expect("empty left tree edge")[node_ptr.len];
            }
        }
    }

    /// Try to steal node from sibling or merge with sibling.
    /// Self maybe leaf node or internal node, when we merge two leaf nodes, there is one parent nodes deletion.
    /// The return value indicates if the root node is empty.
    fn steal_or_merge_with_sibling_recursing(mut this: NonNull<Node<K, V>>) -> bool {
        unsafe {
            loop {
                let underfull_node_ptr = this.as_mut();
                if underfull_node_ptr.len >= MIN_CAP {
                    return false;
                }
                // If this is a root node, it can contains nodes less than MIN_CAP
                if matches!(underfull_node_ptr.parent, None) {
                    // When root node become empty, we need to make its only edge become new root node.
                    return underfull_node_ptr.len == 0;
                }
                let mut parent_node = underfull_node_ptr.parent.unwrap();
                let sibling = underfull_node_ptr.find_sibling_node();
                match sibling {
                    SiblingNode::Left(mut left_node) => {
                        // move all current node and parent_node value into left leaf node.
                        // If parent_node become empty, we need to make current merged node become new root node.
                        if left_node.as_mut().len + underfull_node_ptr.len + 1 <= CAPACITY {
                            // println!("merged by left node..");
                            let removed_parent_idx = underfull_node_ptr.parent_idx.unwrap() - 1;
                            let parent_node_mut = parent_node.as_mut();
                            let old_parent_len = parent_node_mut.len;
                            let parent_kv = (
                                parent_node_mut.keys[removed_parent_idx].as_ptr().read(),
                                parent_node_mut.vals[removed_parent_idx].as_ptr().read(),
                            );
                            // remove parent node and shift all edges to left
                            slice_shl(
                                &mut parent_node_mut.keys[removed_parent_idx + 1..old_parent_len],
                                1,
                            );
                            slice_shl(
                                &mut parent_node_mut.vals[removed_parent_idx + 1..old_parent_len],
                                1,
                            );
                            if let Some(parent_edges) = parent_node_mut.edges.as_mut() {
                                edges_shl(
                                    &mut parent_edges[removed_parent_idx + 2..old_parent_len + 1],
                                    1,
                                );
                                // correct the parent's edges
                                for (offset, edge) in parent_edges
                                    [removed_parent_idx + 1..old_parent_len]
                                    .iter_mut()
                                    .enumerate()
                                {
                                    edge.as_mut().parent_idx =
                                        Some(offset + removed_parent_idx + 1);
                                }
                            }

                            let left_node_mut = left_node.as_mut();
                            let left_old_len = left_node_mut.len;
                            left_node_mut.keys[left_old_len].write(parent_kv.0);
                            left_node_mut.vals[left_old_len].write(parent_kv.1);

                            // move all current node's data into left sibling node
                            if left_old_len + 1 < CAPACITY {
                                move_to_slice(
                                    &underfull_node_ptr.keys[0],
                                    &mut left_node_mut.keys[left_old_len + 1],
                                    underfull_node_ptr.len,
                                );
                                move_to_slice(
                                    &mut underfull_node_ptr.vals[0],
                                    &mut left_node_mut.vals[left_old_len + 1],
                                    underfull_node_ptr.len,
                                );
                            }

                            // if there are edges, it indicates this is a internal node merge.
                            if let Some(underfull_edges) = underfull_node_ptr.edges.as_mut() {
                                move_to_edge(
                                    &underfull_edges[0],
                                    &mut left_node_mut.edges.as_mut().expect(
                                        "a internal node's sibling node must be internal node",
                                    )[left_old_len + 1],
                                    underfull_node_ptr.len + 1,
                                );
                                // correct parent_idx & parent value in the edges.
                                for (offset, moved_edge) in left_node_mut.edges.as_mut().unwrap()
                                    [left_old_len + 1..]
                                    .iter_mut()
                                    .enumerate()
                                {
                                    moved_edge.as_mut().parent_idx =
                                        Some(offset + left_old_len + 1);
                                    moved_edge.as_mut().parent = Some(left_node);
                                }
                            }

                            parent_node_mut.len = old_parent_len - 1;
                            left_node_mut.len = left_old_len + 1 + underfull_node_ptr.len;
                            // drop the memory of right node use
                            alloc::alloc::dealloc(
                                this.as_ptr() as *mut u8,
                                Layout::new::<Node<K, V>>(),
                            );
                            this = parent_node;
                        } else {
                            let left_node_mut = left_node.as_mut();
                            let left_node_old_len = left_node_mut.len;
                            let steal_kv = (
                                left_node_mut.keys[left_node_old_len - 1].as_ptr().read(),
                                left_node_mut.vals[left_node_old_len - 1].as_ptr().read(),
                            );

                            let underfull_old_len = underfull_node_ptr.len;
                            let parent_idx = underfull_node_ptr.parent_idx.unwrap();
                            let parent_node_mut = parent_node.as_mut();
                            let parent_kv = (
                                parent_node_mut.keys[parent_idx - 1].as_ptr().read(),
                                parent_node_mut.vals[parent_idx - 1].as_ptr().read(),
                            );
                            parent_node_mut.keys[parent_idx - 1].write(steal_kv.0);
                            parent_node_mut.vals[parent_idx - 1].write(steal_kv.1);

                            // make room for new insertion
                            slice_shr(&mut underfull_node_ptr.keys[..underfull_old_len], 1);
                            slice_shr(&mut underfull_node_ptr.vals[..underfull_old_len], 1);
                            underfull_node_ptr.keys[0].write(parent_kv.0);
                            underfull_node_ptr.vals[0].write(parent_kv.1);

                            // Append the steal node's edge together into right
                            if let Some(left_edges) = left_node_mut.edges {
                                edges_shr(
                                    &mut underfull_node_ptr
                                        .edges
                                        .as_mut()
                                        .expect("inernal node must have edges")
                                        [..underfull_old_len + 1],
                                    1,
                                );
                                underfull_node_ptr.edges.as_mut().unwrap()[0] =
                                    left_edges[left_node_old_len];

                                // we must correct all parent_idx value in the edges, cause our edges array had shifted.
                                for (pos, edge) in underfull_node_ptr
                                    .edges
                                    .expect("internal node must have edges")
                                    [..underfull_old_len + 1]
                                    .iter_mut()
                                    .enumerate()
                                {
                                    if pos == 0 {
                                        edge.as_mut().parent = Some(this);
                                    }
                                    edge.as_mut().parent_idx = Some(pos);
                                }
                            }

                            left_node_mut.len = left_node_old_len - 1;
                            underfull_node_ptr.len += 1;
                        }
                    }
                    SiblingNode::Right(mut right_node) => {
                        if right_node.as_mut().len + underfull_node_ptr.len + 1 <= CAPACITY {
                            // println!("merged by right node..");
                            let parent_idx = underfull_node_ptr.parent_idx.unwrap();
                            assert!(parent_idx == 0);
                            let parent_kv = parent_node.as_mut().remove_kv_fit(parent_idx);

                            let old_underfull_node_len = underfull_node_ptr.len;
                            let right_node_mut = right_node.as_mut();
                            let right_old_len = right_node_mut.len;
                            let right_new_len = right_old_len + old_underfull_node_len + 1;

                            slice_shr(
                                &mut right_node_mut.keys[..right_old_len],
                                right_new_len - right_old_len,
                            );
                            slice_shr(
                                &mut right_node_mut.vals[..right_old_len],
                                right_new_len - right_old_len,
                            );

                            right_node_mut.keys[old_underfull_node_len].write(parent_kv.0);
                            right_node_mut.vals[old_underfull_node_len].write(parent_kv.1);
                            move_to_slice(
                                &underfull_node_ptr.keys[0],
                                &mut right_node_mut.keys[0],
                                old_underfull_node_len,
                            );
                            move_to_slice(
                                &underfull_node_ptr.vals[0],
                                &mut right_node_mut.vals[0],
                                underfull_node_ptr.len,
                            );

                            if let Some(left_edges) = underfull_node_ptr.edges {
                                // shift to make room for left edges
                                edges_shr(
                                    &mut right_node_mut
                                        .edges
                                        .as_mut()
                                        .expect("inernal node must have edges")
                                        [..right_old_len + 1],
                                    underfull_node_ptr.len + 1,
                                );
                                // move all left edges to the right node
                                move_to_edge(
                                    &left_edges[0],
                                    &mut right_node_mut
                                        .edges
                                        .as_mut()
                                        .expect("inernal node must have edges")[0],
                                    underfull_node_ptr.len + 1,
                                );
                                // update all edges's parent val
                                for (offset, edge) in right_node_mut.edges.unwrap()
                                    [..right_new_len + 1]
                                    .iter_mut()
                                    .enumerate()
                                {
                                    if offset <= old_underfull_node_len {
                                        edge.as_mut().parent = Some(right_node);
                                    }
                                    edge.as_mut().parent_idx = Some(offset);
                                }
                            }

                            right_node_mut.len = right_new_len;
                            // drop the memory of removed left node use
                            alloc::alloc::dealloc(
                                this.as_ptr() as *mut u8,
                                Layout::new::<Node<K, V>>(),
                            );
                            this = parent_node;
                        } else {
                            let parent_node_mut = parent_node.as_mut();
                            let old_underfull_node_len = underfull_node_ptr.len;
                            let parent_idx = underfull_node_ptr.parent_idx.unwrap();
                            assert!(parent_idx == 0);
                            let right_node_mut = right_node.as_mut();
                            let old_right_node_len = right_node_mut.len;

                            let stealed_kv = (
                                right_node_mut.keys[0].as_ptr().read(),
                                right_node_mut.vals[0].as_ptr().read(),
                            );
                            slice_shl(&mut right_node_mut.keys[1..old_right_node_len], 1);
                            slice_shl(&mut right_node_mut.vals[1..old_right_node_len], 1);

                            // The removed parent node idx must be 0, cause we don't have left edge, but we still use parent_idx here.
                            let parent_kv = (
                                parent_node_mut.keys[parent_idx].as_ptr().read(),
                                parent_node_mut.vals[parent_idx].as_ptr().read(),
                            );
                            parent_node_mut.keys[parent_idx].write(stealed_kv.0);
                            parent_node_mut.vals[parent_idx].write(stealed_kv.1);

                            underfull_node_ptr.keys[old_underfull_node_len].write(parent_kv.0);
                            underfull_node_ptr.vals[old_underfull_node_len].write(parent_kv.1);

                            // move edges
                            if let Some(right_edges) = right_node_mut.edges.as_mut() {
                                let old_underfull_node_len = underfull_node_ptr.len;
                                let mut appended_edge = right_edges[0];
                                appended_edge.as_mut().parent_idx =
                                    Some(old_underfull_node_len + 1);
                                appended_edge.as_mut().parent = Some(this);

                                underfull_node_ptr.edges.as_mut().unwrap()
                                    [old_underfull_node_len + 1] = appended_edge;

                                // correct all right node's child node parent_idx value
                                edges_shl(&mut right_edges[1..old_right_node_len + 1], 1);
                                for (offset, edge) in
                                    right_edges[..old_right_node_len].iter_mut().enumerate()
                                {
                                    edge.as_mut().parent_idx = Some(offset);
                                }
                            }

                            right_node_mut.len = old_right_node_len - 1;
                            underfull_node_ptr.len += 1;
                        }
                    }
                }
            }
        }

        // parent_node
    }

    // For a leaf node, it must have a sibling leaf node.
    fn find_sibling_node(&mut self) -> SiblingNode<K, V> {
        let mut parent_node = self.parent.unwrap();
        unsafe {
            let parent_idx = self.parent_idx.unwrap();
            if parent_idx > 0 {
                SiblingNode::Left(
                    parent_node
                        .as_mut()
                        .edges
                        .expect("internal node must have edges")[parent_idx - 1],
                )
            } else {
                SiblingNode::Right(
                    parent_node
                        .as_mut()
                        .edges
                        .expect("internal node must have edges")[parent_idx + 1],
                )
            }
        }
    }

    #[allow(dead_code)]
    pub fn find_leftmost_node(this: NonNull<Node<K, V>>) -> NonNull<Node<K, V>> {
        let mut current_node = this;
        loop {
            unsafe {
                let current_node_mut = current_node.as_mut();
                assert!(current_node_mut.len > 0);
                if matches!(current_node_mut.node_type, NodeType::InternalNode) {
                    current_node = current_node_mut.edges.unwrap()[0];
                } else {
                    return current_node;
                }
            }
        }
    }
}

#[allow(dead_code)]
impl<K, V> BTreeMap<K, V>
where
    K: Ord + Debug,
{
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let mut old_v = None;
        match self.root {
            None => {
                // it's safe to use leaf type represents root node, when we need
                // to merge the leftmost node and make it as new root, which means
                // we only can have two levels in the tree.
                let node = Node::new_leaf();
                Node::insert(node, k, v, 0, None);
                self.root = Some(node);
                self.len += 1;
            }
            Some(root) => match Self::search_in_tree(root, &k) {
                SearchResult::Found((handle, idx)) => {
                    unsafe {
                        old_v = Some(handle.as_ref().vals[idx].as_ptr().read());
                    }
                    self.insert_kv_recurring(handle, k, v, idx);
                }
                SearchResult::NotFound((handle, idx)) => {
                    self.insert_kv_recurring(handle, k, v, idx);
                    self.len += 1;
                }
            },
        }

        old_v
    }

    /// Insert from a start node in recurse way
    fn insert_kv_recurring(
        &mut self,
        node: NonNull<Node<K, V>>,
        mut k: K,
        mut v: V,
        mut idx: usize,
    ) {
        unsafe {
            let mut current_node = node;
            let mut insertion_edge: Option<NonNull<Node<K, V>>> = None;

            loop {
                let current_node_mut = current_node.as_mut();
                match Node::insert(current_node, k, v, idx, insertion_edge) {
                    Some(SplitResult { split_node, kv }) => match current_node_mut.parent {
                        Some(parent_node) => {
                            k = kv.0;
                            v = kv.1;
                            idx = current_node_mut.parent_idx.unwrap();
                            insertion_edge = Some(split_node);
                            current_node = parent_node;
                        }
                        None => {
                            // Need to recreate a root node in this case, and we can assert there is no more insertion.
                            let mut new_root = Node::new_internal_node();
                            Node::insert_kv_edge_fit(new_root, kv.0, kv.1, 0, Some(split_node));

                            let old_root = self.root.take().unwrap().as_mut();
                            old_root.parent = Some(new_root);
                            old_root.parent_idx = Some(0);

                            // we need set new root's first edge to self.
                            new_root.as_mut().edges.as_mut().unwrap()[0] = NonNull::from(old_root);
                            self.root = Some(new_root);
                            break;
                        }
                    },
                    None => {
                        // There is enough space to insert, we just return.
                        break;
                    }
                }
            }
        }
    }

    /// If there is k, return Some(v), else return None.
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let mut root = self.root?;
        if let SearchResult::Found((node, idx)) = Self::search_in_tree(root, k) {
            let mut empty_root_node = false;
            let removed_val = Node::remove(node, idx, || {
                empty_root_node = true;
            });
            self.len -= 1;
            if empty_root_node {
                unsafe {
                    let old_root = self
                        .root
                        .expect("root can not be empty when there is element");
                    self.root = if self.len > 0 {
                        // println!("handle empty root, len is: {}", self.len);
                        let mut new_root = root.as_mut().edges.unwrap()[0];
                        new_root.as_mut().parent = None;
                        new_root.as_mut().parent_idx = None;
                        // IMPORTANT: we can't mark new root's type as NodeType::Leaf, even when current root is empty,
                        // which only means the root and its sibling need to merge to reduce the height of tree,
                        // and we can't know if new root has edges.
                        // new_root.as_mut().node_type = NodeType::Leaf;
                        Some(new_root)
                    } else {
                        None
                    };
                    // drop the memory of right node use
                    alloc::alloc::dealloc(
                        old_root.as_ptr() as *mut u8,
                        Layout::new::<Node<K, V>>(),
                    );
                }
            }
            Some(removed_val)
        } else {
            None
        }
    }

    /// return Some(&V) if the key exists in the tree
    pub fn get(&self, k: &K) -> Option<&V> {
        if let Some(root) = self.root {
            if let SearchResult::Found((node, idx)) = Self::search_in_tree(root, k) {
                unsafe {
                    let r = (&*node.as_ptr()).get_value_ref(idx);
                    Some(r)
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn contains_key(&mut self, k: &K) -> bool {
        if let Some(root) = self.root {
            if let SearchResult::Found(_) = Self::search_in_tree(root, k) {
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    // Search in the tree,
    fn search_in_tree<'a>(mut node: NonNull<Node<K, V>>, k: &'a K) -> SearchResult<K, V> {
        // println!("search key: {:?}", k);
        unsafe {
            let mut curren_node = node.as_mut();
            loop {
                let mut offset: usize = 0;
                for key in curren_node.get_keys(0..curren_node.len) {
                    match k.cmp(key) {
                        Ordering::Equal => {
                            return SearchResult::Found((NonNull::from(curren_node), offset));
                        }
                        Ordering::Less => {
                            break;
                        }
                        Ordering::Greater => {
                            offset += 1;
                        }
                    }
                }

                if matches!(curren_node.node_type, NodeType::Leaf) {
                    return SearchResult::NotFound((NonNull::from(curren_node), offset));
                }
                // println!("edges offset: {offset}");
                // we can assert there must be a non-null ptr for underlying edge, cause when we
                // restructure b-tree, the edges are promised to be 1 more than current node num.
                curren_node = curren_node.edges.unwrap()[offset].as_mut();
            }
        }
    }

    /// used to debug the b-tree's initial structure
    pub fn print_structure(&self) {
        let separator = "-------------------------------------------------------".to_string();
        let mut queue = VecDeque::new();
        let mut sep_idx_queue = VecDeque::new();
        let init_node = if let Some(root) = self.root {
            root
        } else {
            return;
        };
        queue.push_back(init_node);

        loop {
            let mut child_queue = VecDeque::new();
            let mut next_seq_idx = VecDeque::new();
            let mut cur_seq_idx = sep_idx_queue.pop_front();
            let mut counter: usize = 0;
            while let Some(mut node) = queue.pop_front() {
                counter += 1;
                unsafe {
                    let node_mut = node.as_mut();
                    let node_len = node_mut.len;
                    for i in 0..node_len {
                        print!("{:?},", node_mut.keys[i].as_ptr().read());
                    }
                    if matches!(node_mut.node_type, NodeType::InternalNode) {
                        child_queue.extend(
                            node_mut.edges.expect("internal node must have edges")
                                [0..node_mut.len + 1]
                                .into_iter(),
                        );
                        next_seq_idx.push_back(node_len + 1);
                    }
                }

                if let Some(seq_idx) = cur_seq_idx {
                    if matches!(counter.cmp(&seq_idx), Ordering::Equal) {
                        cur_seq_idx = sep_idx_queue.pop_front();
                        counter = 0;
                        print!("      ");
                    }
                }

                if queue.len() != 0 && counter > 0 {
                    // print a separator symbol for sibling nodes
                    print!("|");
                }
            }

            if child_queue.len() == 0 {
                break;
            }
            sep_idx_queue = next_seq_idx;
            queue = child_queue;
            println!();
            println!("{separator}");
        }
        println!();
        println!("{separator}");
    }
}

enum SearchResult<K, V> {
    Found((NonNull<Node<K, V>>, usize)),
    NotFound((NonNull<Node<K, V>>, usize)),
}

/// Shift the slice to the left with some distance
pub fn slice_shl<T>(slice: &mut [MaybeUninit<T>], distance: usize) {
    if slice.len() != 0 {
        let ptr = slice.as_mut_ptr();
        unsafe { std::ptr::copy(ptr, ptr.sub(distance), slice.len()) }
    }
}

/// Shift the slice to the right with some distance
pub fn slice_shr<T>(slice: &mut [MaybeUninit<T>], distance: usize) {
    if slice.len() != 0 {
        let ptr = slice.as_mut_ptr();
        unsafe { std::ptr::copy(ptr, ptr.add(distance), slice.len()) }
    }
}

pub fn edges_shr<T>(edges: &mut [NonNull<T>], distance: usize) {
    if edges.len() != 0 {
        let ptr = edges.as_mut_ptr();
        unsafe { std::ptr::copy(ptr, ptr.add(distance), edges.len()) }
    }
}

/// Important: we should shift the array of NonNull instead of the memory address the NonNull points to.
/// OR we will make our data moved to unknown place.
pub fn edges_shl<T>(edges: &mut [NonNull<T>], distance: usize) {
    if edges.len() != 0 {
        unsafe {
            let ptr = edges.as_mut_ptr();
            std::ptr::copy(ptr, ptr.sub(distance), edges.len())
        }
    }
}

/// Move a slice to another slice
pub fn move_to_slice<T>(src: &MaybeUninit<T>, des: &mut MaybeUninit<T>, len: usize) {
    if len > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const _, des as *mut _, len);
        }
    }
}

pub fn move_to_edge<T>(src: &NonNull<T>, des: &mut NonNull<T>, len: usize) {
    if len > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const _, des as *mut _, len);
        }
    }
}

pub struct BTreeIter<K, V>
where
    K: Ord,
{
    node: NonNull<Node<K, V>>,
    len: usize,
    current: usize,
}

impl<K, V> Iterator for BTreeIter<K, V>
where
    K: Ord,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        unsafe {
            let node_mut = self.node.as_mut();
            let current_pos = self.current;
            if node_mut.len > current_pos {
                let kv = (
                    node_mut.keys[current_pos].as_ptr().read(),
                    node_mut.vals[current_pos].as_ptr().read(),
                );
                self.current += 1;

                // check if current node's has child node
                if matches!(node_mut.node_type, NodeType::InternalNode) {
                    let next_edge_node = node_mut
                        .edges
                        .expect("internal node must have underlying edge")[self.current];
                    // println!("go to leftmost child node from idx {}..", self.current);
                    self.node = Node::find_leftmost_node(next_edge_node);
                    self.current = 0;

                    return Some(kv);
                }

                if self.current == node_mut.len {
                    let mut finished_node = self.node;
                    // * when we read all data in a leaf node, we need to forward to its parent
                    // * if we finish a internal node, we need to pop up.
                    while let Some(parent_node) = finished_node.as_mut().parent {
                        // println!("go to parent node..");
                        let pos_in_parent = finished_node.as_ref().parent_idx.unwrap();
                        self.current = pos_in_parent;
                        // drop the finished node's memory
                        alloc::alloc::dealloc(
                            finished_node.as_ptr() as *mut u8,
                            Layout::new::<Node<K, V>>(),
                        );
                        finished_node = parent_node;
                        // println!(
                        //     "the pos of edge in parent: {pos_in_parent}, len of parent: {}",
                        //     parent_node.as_ref().len
                        // );
                        if pos_in_parent < parent_node.as_ref().len {
                            break;
                        }
                    }
                    self.node = finished_node;
                }

                return Some(kv);
            }

            // when we finish to iterate all nodes, we must remember to dealloc the root node.
            // println!("dealloc addr: {}", self.node.addr());
            alloc::alloc::dealloc(self.node.as_ptr() as *mut u8, Layout::new::<Node<K, V>>());
            // in case to double free root node again
            self.len = 0;
            None
        }
    }
}

impl<K, V> IntoIterator for BTreeMap<K, V>
where
    K: Ord + Debug,
{
    type IntoIter = BTreeIter<K, V>;
    type Item = (K, V);

    fn into_iter(self) -> Self::IntoIter {
        let iter = BTreeIter {
            node: if let Some(root) = self.root {
                Node::find_leftmost_node(root)
            } else {
                NonNull::dangling()
            },
            len: self.len,
            current: 0,
        };
        // Important: we need to drop the memory when iter is dropped, or we will drop it before we
        // use the iterator.
        std::mem::forget(self);
        iter
    }
}

impl<K, V> Drop for BTreeIter<K, V>
where
    K: Ord,
{
    fn drop(&mut self) {
        // make sure all key value pair's drop impl is called, or not will cause memory leak.
        // See: https://users.rust-lang.org/t/why-extra-drop-while-deallocation/82508/8
        // drop all memory of Nodes when we finish to iterate every node
        while let Some(_) = self.next() {}
    }
}

unsafe impl<#[may_dangle] K, #[may_dangle] V> Drop for BTreeMap<K, V>
where
    K: Ord + Debug,
{
    fn drop(&mut self) {
        unsafe {
            drop(std::ptr::read(self).into_iter());
        }
    }
}

#[cfg(test)]
mod btree_api {
    use super::*;

    #[test]
    fn test_cust_btree_basic() {
        let mut btree = BTreeMap::new();
        let key_val_pair = [
            (1, "a"),
            (2, "b"),
            (3, "c"),
            (4, "d"),
            (5, "e"),
            (6, "f"),
            (7, "g"),
            (8, "h"),
            (9, "i"),
            (10, "j"),
            (11, "k"),
            (12, "l"),
            (13, "m"),
            (15, "n"),
            (16, "o"),
            (17, "p"),
            (18, "q"),
            (19, "r"),
            (20, "s"),
            (21, "t"),
            (22, "u"),
            (23, "v"),
            (24, "w"),
            (25, "x"),
            (26, "y"),
            (27, "z"),
        ];

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        // let mut max_iter = 50;
        // for (k, v) in btree.into_iter() {
        //     println!("{k} : {v}");
        //     if max_iter < 0 {
        //         break;
        //     }
        //     max_iter -= 1;
        // }

        // for (idx, val) in key_val_pair {
        //     assert_eq!(
        //         btree.remove(&idx),
        //         Some(val),
        //         "the value removed is not same with insertion"
        //     );
        // }

        for _ in btree.into_iter() {}
    }

    #[test]
    fn test_removed_from_large() {
        let mut btree = BTreeMap::new();
        let key_val_pair = [
            (27, "z"),
            (26, "y"),
            (25, "x"),
            (24, "w"),
            (23, "v"),
            (22, "u"),
            (21, "t"),
            (20, "s"),
            (19, "r"),
            (18, "q"),
            (17, "p"),
            (16, "o"),
            (15, "n"),
            (13, "m"),
            (12, "l"),
            (11, "k"),
            (10, "j"),
            (9, "i"),
            (8, "h"),
            (7, "g"),
            (6, "f"),
            (5, "e"),
            (4, "d"),
            (3, "c"),
            (2, "b"),
            (1, "a"),
        ];

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        for (idx, val) in key_val_pair {
            assert_eq!(
                btree.remove(&idx),
                Some(val),
                "the value removed is not same with insertion"
            );
        }
    }

    #[test]
    fn test_unordered_insert_remove() {
        let mut btree = BTreeMap::new();
        let key_val_pair = [
            (27, "z"),
            (26, "y"),
            (25, "x"),
            (24, "w"),
            (23, "v"),
            (22, "u"),
            (21, "t"),
            (20, "s"),
            (8, "h"),
            (7, "g"),
            (6, "f"),
            (5, "e"),
            (4, "d"),
            (3, "c"),
            (2, "b"),
            (1, "a"),
            (19, "r"),
            (18, "q"),
            (17, "p"),
            (16, "o"),
            (15, "n"),
            (13, "m"),
            (12, "l"),
            (11, "k"),
            (10, "j"),
            (9, "i"),
        ];

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        for (idx, val) in key_val_pair {
            assert_eq!(
                btree.remove(&idx),
                Some(val),
                "the value removed is not same with insertion"
            );
        }
    }

    #[test]
    fn test_large_unordered() {
        let mut btree = BTreeMap::new();
        let key_val_pair = [
            (27, "z"),
            (26, "y"),
            (25, "x"),
            (24, "w"),
            (23, "v"),
            (22, "u"),
            (21, "t"),
            (20, "s"),
            (8, "h"),
            (7, "g"),
            (6, "f"),
            (5, "e"),
            (4, "d"),
            (3, "c"),
            (2, "b"),
            (1, "a"),
            (19, "r"),
            (18, "q"),
            (17, "p"),
            (16, "o"),
            (15, "n"),
            (13, "m"),
            (12, "l"),
            (11, "k"),
            (10, "j"),
            (9, "i"),
            // ---------------- split from first group
            (34, "g2"),
            (35, "h2"),
            (36, "i2"),
            (37, "j2"),
            (28, "a2"),
            (29, "b2"),
            (30, "c2"),
            (31, "d2"),
            (32, "e2"),
            (33, "f2"),
            (38, "k2"),
            (39, "l2"),
            (40, "m2"),
            (41, "n2"),
            (49, "v2"),
            (50, "w2"),
            (51, "x2"),
            (52, "y2"),
            (53, "z2"),
            (42, "o2"),
            (43, "p2"),
            (44, "q2"),
            (45, "r2"),
            (46, "s2"),
            (47, "t2"),
            (48, "u2"),
        ];

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        for (idx, val) in key_val_pair {
            assert_eq!(
                btree.remove(&idx),
                Some(val),
                "the value removed is not same with insertion"
            );
        }
    }

    #[test]
    fn test_insert_after_remove() {
        let mut btree = BTreeMap::new();
        let key_val_pair = [
            (27, "z"),
            (26, "y"),
            (25, "x"),
            (24, "w"),
            (23, "v"),
            (22, "u"),
            (21, "t"),
            (20, "s"),
            (8, "h"),
            (7, "g"),
            (6, "f"),
            (5, "e"),
            (4, "d"),
            (3, "c"),
            (2, "b"),
            (1, "a"),
            (19, "r"),
            (18, "q"),
            (17, "p"),
            (16, "o"),
            (15, "n"),
            (13, "m"),
            (12, "l"),
            (11, "k"),
            (10, "j"),
            (9, "i"),
            // ---------------- split from first group
            (34, "g2"),
            (35, "h2"),
            (36, "i2"),
            (37, "j2"),
            (28, "a2"),
            (29, "b2"),
            (30, "c2"),
            (31, "d2"),
            (32, "e2"),
            (33, "f2"),
            (38, "k2"),
            (39, "l2"),
            (40, "m2"),
            (41, "n2"),
            (49, "v2"),
            (50, "w2"),
            (51, "x2"),
            (52, "y2"),
            (53, "z2"),
            (42, "o2"),
            (43, "p2"),
            (44, "q2"),
            (45, "r2"),
            (46, "s2"),
            (47, "t2"),
            (48, "u2"),
        ];

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        for (idx, val) in key_val_pair.clone() {
            assert_eq!(
                btree.remove(&idx),
                Some(val),
                "the value removed is not same with insertion"
            );
        }

        for (key, val) in key_val_pair.clone().into_iter() {
            btree.insert(key, val);
        }

        for (idx, val) in key_val_pair.clone() {
            assert_eq!(
                btree.remove(&idx),
                Some(val),
                "the value removed is not same with insertion"
            );
        }
    }
}
