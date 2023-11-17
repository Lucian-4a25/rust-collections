use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

pub struct LRUCache<K, V> {
    // record the access record
    access_record: VecDeque<K>,
    cache: HashMap<K, (V, usize)>,
    capacity: usize,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Default::default(),
            capacity,
            access_record: Default::default(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(v) = self.cache.get_mut(&key) {
            v.1 += 1;
            self.access_record.push_back(key.clone());
            Some(&v.0)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if let Some(v) = self.cache.get_mut(&key) {
            v.0 = value;
            v.1 += 1;
        } else if self.is_full() {
            self.evict();
            self.cache.insert(key.clone(), (value, 1));
        } else {
            self.cache.insert(key.clone(), (value, 1));
        }
        self.access_record.push_back(key);
    }

    fn evict(&mut self) {
        while let Some(k) = self.access_record.pop_front() {
            match self.cache.entry(k) {
                Entry::Occupied(mut entry) => {
                    let count = entry.get().1;
                    if count == 1 {
                        entry.remove();
                        break;
                    } else {
                        entry.get_mut().1 = count - 1;
                    }
                }
                _ => {
                    unreachable!()
                }
            }
        }
    }

    fn is_full(&self) -> bool {
        self.cache.len() == (self.capacity as usize)
    }

    // fn get_cur_time() -> u128 {
    //     let timing = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
    //     timing.unwrap().as_nanos()
    // }
}
