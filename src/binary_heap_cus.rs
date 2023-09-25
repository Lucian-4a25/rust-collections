use std::{cmp::Ordering, mem::swap};

use crate::vec_cus::Vec;

/// A priority queue implemention with binary-heap structure
pub struct BinaryHeap<V: Ord> {
    data: Vec<V>,
}

impl<V: Ord> BinaryHeap<V> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        BinaryHeap { data: Vec::new() }
    }

    #[allow(dead_code)]
    pub fn push(&mut self, v: V) {
        let old_len = self.data.len();
        self.data.push(v);

        // remain the tree structure
        self.heap_up(old_len);
    }

    #[allow(dead_code)]
    pub fn pop(&mut self) -> Option<V> {
        self.data.pop().map(|mut v| {
            if self.data.len() > 0 {
                swap(&mut v, &mut self.data[0]);
                self.heap_down(0);
            }
            v
        })
    }

    /// heap-up from a specified position
    fn heap_up(&mut self, pos: usize) {
        // construct a hole to up-heap
        let mut hole = Hole::new(&mut self.data, pos);

        let cur_val = hole.read(hole.pos());
        while let Some(parent_idx) = hole.parent_pos() {
            if hole.get(parent_idx) >= &cur_val {
                break;
            }

            hole.write(hole.pos(), hole.read(parent_idx));
            hole.set_pos(parent_idx);
        }

        hole.write(hole.pos(), cur_val);
    }

    /// heap-down from a specified position
    fn heap_down(&mut self, pos: usize) {
        let end_pos = self.data.len();
        let mut hole = Hole::new(&mut self.data, pos);

        let cur_val = hole.read(hole.pos());
        let mut left_child_pos = hole.left_child_pos();
        while left_child_pos < end_pos {
            let left_child = hole.get(left_child_pos);

            let new_pos = match &cur_val.cmp(left_child) {
                Ordering::Less => {
                    // check if right child is greater
                    if left_child_pos + 1 < end_pos && hole.get(left_child_pos + 1) > left_child {
                        left_child_pos + 1
                    } else {
                        left_child_pos
                    }
                }
                _ => {
                    if left_child_pos + 1 >= end_pos || &cur_val >= hole.get(left_child_pos + 1) {
                        break;
                    } else {
                        left_child_pos + 1
                    }
                }
            };

            hole.write(hole.pos(), hole.read(new_pos));
            hole.set_pos(new_pos);

            left_child_pos = hole.left_child_pos();
        }

        hole.write(hole.pos(), cur_val);
    }
}

pub struct Hole<'a, T: 'a> {
    arr: &'a mut [T],
    // index in the arr
    pos: usize,
}

impl<'a, T: 'a> Hole<'a, T> {
    pub fn new(data: &'a mut [T], pos: usize) -> Self {
        Hole { arr: data, pos }
    }

    fn parent_pos(&self) -> Option<usize> {
        let pos = self.pos();
        if pos > 0 {
            Some((pos - 1) / 2)
        } else {
            None
        }
    }

    fn left_child_pos(&self) -> usize {
        self.pos() * 2 + 1
    }

    #[allow(dead_code)]
    fn right_child_pos(&self) -> usize {
        self.pos() * 2 + 2
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn set_pos(&mut self, new_pos: usize) {
        self.pos = new_pos;
    }

    fn get(&self, idx: usize) -> &T {
        unsafe { self.arr.get_unchecked(idx) }
    }

    fn read(&self, idx: usize) -> T {
        unsafe { self.arr.as_ptr().add(idx).read() }
    }

    fn write(&mut self, idx: usize, v: T) {
        unsafe { self.arr.as_mut_ptr().add(idx).write(v) }
    }
}

#[allow(dead_code)]
pub fn binary_heap_usage() {
    #[derive(Debug)]
    struct D {
        idx: usize,
        d: String,
    }

    impl PartialEq for D {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl PartialOrd for D {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.idx.cmp(&other.idx))
        }
    }

    impl Eq for D {}

    impl Ord for D {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.idx.cmp(&other.idx)
        }
    }

    let mut binary_heap = BinaryHeap::new();

    let insertion_num = 1000000;
    for i in 0..insertion_num / 2 {
        binary_heap.push(D {
            idx: i,
            d: format!("wwwwwwwwwwww{i}"),
        });
    }

    for i in (insertion_num / 2..insertion_num).rev() {
        binary_heap.push(D {
            idx: i,
            d: format!("wwwwwwwwwwww{i}"),
        });
    }

    for i in (0..insertion_num).rev() {
        assert_eq!(
            binary_heap.pop(),
            Some(D {
                idx: i,
                d: format!("wwwwwwwwwwww{i}"),
            })
        );
    }
}

#[cfg(test)]
mod binary_heap_api {
    use super::*;

    #[test]
    fn test_basic() {
        let mut binary_heap = BinaryHeap::new();

        for i in 0..5 {
            binary_heap.push(i);
        }

        for i in (5..10).rev() {
            binary_heap.push(i);
        }

        for i in (0..10).rev() {
            assert_eq!(binary_heap.pop(), Some(i));
        }
    }

    #[test]
    fn test_large_number() {
        binary_heap_usage();
    }
}
