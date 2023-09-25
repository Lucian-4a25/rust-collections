use indexmap::IndexSet;
use std::ops::Index;

/// A Vec which will keep its element position stable when remove happened
pub struct StableVec<T> {
    data: Vec<Option<T>>,
    deleted_ids: IndexSet<usize>,
}

#[allow(dead_code)]
impl<T> StableVec<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn contains(&self, n: usize) -> bool {
        self.data.len() > n && !self.deleted_ids.contains(&n)
    }

    pub fn with_capacity(size: usize) -> Self {
        Self {
            data: Vec::with_capacity(size),
            deleted_ids: Default::default(),
        }
    }

    pub fn insert(&mut self, v: T) -> usize {
        if let Some(id) = self.deleted_ids.pop() {
            self.data[id] = Some(v);
            return id;
        }
        self.data.push(Some(v));
        self.data.len() - 1
    }

    pub fn remove(&mut self, n: usize) -> Option<T> {
        let removed_data = self.data[n].take();
        if removed_data.is_some() {
            self.deleted_ids.insert(n);
        }

        removed_data
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.deleted_ids.clear();
    }

    /// the max position of nodes
    pub fn max_pos(&self) -> usize {
        self.data.len()
    }

    /// the actual number of nodes
    pub fn node_count(&self) -> usize {
        self.data.len() - self.deleted_ids.len()
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
}

impl<T> Index<usize> for StableVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if self.data.len() <= index || self.deleted_ids.contains(&index) {
            panic!("the value doesn't exist in the vec");
        }
        self.data[index].as_ref().unwrap()
    }
}

impl<T> Default for StableVec<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            deleted_ids: Default::default(),
        }
    }
}

#[cfg(test)]
mod test_stablevec {
    use super::*;

    #[test]
    fn test_basic() {
        let mut v: _ = StableVec::new();
        let n1 = v.insert(1);
        let n2 = v.insert(2);
        let n3 = v.insert(3);
        let n4 = v.insert(4);
        let n5 = v.insert(5);

        assert_eq!(v.remove(n3), Some(3));
        assert_eq!(v[n2], 2);
        assert_eq!(v[n4], 4);

        let n6 = v.insert(6);
        assert_eq!(1, v[n1]);
        assert_eq!(5, v[n5]);

        assert_eq!(v[n6], 6);
    }
}
