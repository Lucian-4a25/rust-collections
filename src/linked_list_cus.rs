use std::ptr::NonNull;

struct LinkedList<T> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
}

struct Node<T> {
    prev: Option<NonNull<Node<T>>>,
    next: Option<NonNull<Node<T>>>,
    element: T,
    // _marker: PhantomData<Node<'a, T>>,
}

#[allow(dead_code)]
impl<T> Node<T> {
    fn new(ele: T) -> Self {
        Self {
            next: None,
            prev: None,
            element: ele,
        }
    }
}

#[allow(dead_code)]
impl<T> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
        }
    }

    pub fn push_back(&mut self, v: T) {
        let node = Box::leak(Box::new(Node::new(v)));
        match self.tail {
            None => {
                self.tail = NonNull::new(node as *mut Node<T>);
                self.head = NonNull::new(node as *mut Node<T>);
            }
            Some(tail) => unsafe {
                (*tail.as_ptr()).next = NonNull::new(node as *mut Node<T>);
                node.prev = Some(tail);
                self.tail = NonNull::new(node as *mut Node<T>);
            },
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.pop_back_inner().map(|v| v.element)
    }

    #[inline]
    pub fn pop_back_inner(&mut self) -> Option<Box<Node<T>>> {
        if self.head.is_none() && self.tail.is_none() {
            None
        } else {
            match self.tail {
                Some(tail) => unsafe {
                    let current_node = Box::from_raw(tail.as_ptr());
                    let prev = current_node.prev;
                    match prev {
                        Some(prev) => {
                            (*prev.as_ptr()).next = None;
                            self.tail = Some(prev);
                        }
                        None => {
                            self.head = None;
                            self.tail = None;
                        }
                    }

                    return Some(current_node);
                },
                None => unreachable!(),
            }
        }
    }

    pub fn push_front(&mut self, v: T) {
        let node = Box::leak(Box::new(Node::new(v)));
        match self.head {
            None => {
                self.tail = NonNull::new(node as *mut Node<T>);
                self.head = NonNull::new(node as *mut Node<T>);
            }
            Some(head) => unsafe {
                (*head.as_ptr()).prev = NonNull::new(node as *mut Node<T>);
                node.next = Some(head);
                self.head = NonNull::new(node as *mut Node<T>);
            },
        }
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.pop_front_inner().map(|v| v.element)
    }

    #[inline]
    pub fn pop_front_inner(&mut self) -> Option<Box<Node<T>>> {
        if self.head.is_none() && self.tail.is_none() {
            None
        } else {
            match self.head {
                Some(head) => unsafe {
                    let current_node = Box::from_raw(head.as_ptr());
                    match (*head.as_ptr()).next {
                        Some(next) => {
                            (*next.as_ptr()).prev = None;
                            self.head = Some(next);
                        }
                        None => {
                            self.head = None;
                            self.tail = None;
                        }
                    }

                    return Some(current_node);
                },
                None => unreachable!(),
            }
        }
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        while let Some(node) = self.pop_back_inner() {
            // According the API description of Box::leak, we must compose a new Box by Box::from_raw API,
            // then drop this node exciplity to release the allocated memory.
            drop(node);
        }
    }
}

#[cfg(test)]
mod linked_list {
    use super::*;

    #[test]
    fn test_linkedlist() {
        let mut v = LinkedList::new();
        v.push_back(3);
        v.push_back(4);
        v.push_front(2);
        v.push_front(1);
        v.push_back(5);
        assert_eq!(v.pop_back().unwrap(), 5);
        assert_eq!(v.pop_back().unwrap(), 4);
        assert_eq!(v.pop_back().unwrap(), 3);
        assert_eq!(v.pop_back().unwrap(), 2);
        assert_eq!(v.pop_back().unwrap(), 1);
        assert_eq!(v.pop_back(), None);
    }
}
