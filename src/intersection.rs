use crate::{Bvh, EnlargedBvh, Node, NodeIndex, NodeType};

/// A trait for types that can detect whether it intersects with another one of its type.
///
/// See [`guide`](crate::guide) for examples of how to implement this trait.
pub trait IntersectSelf {
    fn intersects(&self, other: &Self) -> bool;
}

impl<B: IntersectSelf, D> Bvh<B, D> {
    /// Do intersection test using the bvh.
    ///
    /// `q: B` is the bounding volume to query leaf nodes that intersect it.
    ///
    /// `f: impl FnMut(&D)` is called for all leaf nodes that intersect `q`.
    /// The custom data stored in these leaves is passed to `f`.
    #[inline]
    pub fn query_intersection(&self, q: B, mut f: impl FnMut(&D)) {
        if self.root_index == NodeIndex::NULL {
            return;
        }

        // SAFETY: self.root_index is valid as long as it is not NULL.
        let root = unsafe { self.get_node_cfg(self.root_index) };

        if q.intersects(&root.bounding_volume) {
            self.query_intersection_rec(root, &q, &mut f);
        }
    }

    fn query_intersection_rec(&self, node: &Node<B, D>, q: &B, f: &mut impl FnMut(&D)) {
        match &node.ty {
            NodeType::Internal { child1, child2 } => {
                let c1 = unsafe { self.get_node_cfg(*child1) };

                if q.intersects(&c1.bounding_volume) {
                    self.query_intersection_rec(c1, q, f);
                }

                let c2 = unsafe { self.get_node_cfg(*child2) };

                if q.intersects(&c2.bounding_volume) {
                    self.query_intersection_rec(c2, q, f);
                }
            }
            NodeType::Leaf(user_data) => f(user_data),
        }
    }

    /// Do intersecion test using explicit stack instead of recursion.
    ///
    /// The performance of this method very close to [`Self::query_intersection`].
    /// The advantage is that using this does not need to worry about stack overflow.
    /// However, it should be noted that during benchmark,
    /// even 1M leaves does not cause stack overflow when using [`Self::query_intersection`].
    ///
    /// Detailed Performance compared to [`Self::query_intersection`] (tested using `cargo bench --bench query`):
    ///
    /// On Apple M4:
    /// - Basically the same when there are 1K leaves.
    /// - 10% improvement when there are 1M leaves.
    ///
    /// On Intel i5-3470:
    /// - 7% worse when there are 1K leaves
    /// - 2% better when there are 1M leaves.
    ///
    #[inline(always)]
    pub fn query_intersection_stack(
        &self,
        stack: &mut Vec<NodeIndex>,
        q: B,
        mut f: impl FnMut(&D),
    ) {
        if self.root_index == NodeIndex::NULL {
            return;
        }

        // SAFETY: self.root_index is valid as long as it is not NULL.
        let root = unsafe { self.get_node_cfg(self.root_index) };

        if q.intersects(&root.bounding_volume) {
            stack.push(self.root_index);
        }

        while let Some(index) = stack.pop() {
            let node = unsafe { self.get_node_cfg(index) };

            match &node.ty {
                NodeType::Internal { child1, child2 } => {
                    let c1 = unsafe { self.get_node_cfg(*child1) };
                    if q.intersects(&c1.bounding_volume) {
                        stack.push(*child1);
                    }

                    let c2 = unsafe { self.get_node_cfg(*child2) };
                    if q.intersects(&c2.bounding_volume) {
                        stack.push(*child2);
                    }
                }
                NodeType::Leaf(user_data) => f(user_data),
            }
        }
    }
}

impl<B: IntersectSelf, D> EnlargedBvh<B, D> {
    #[inline]
    pub fn query_intersection_stack(
        &self,
        stack: &mut Vec<NodeIndex>,
        bounding_volume: B,
        f: impl FnMut(&D),
    ) {
        self.bvh.query_intersection_stack(stack, bounding_volume, f)
    }

    #[inline]
    pub fn query_intersection(&self, bounding_volume: B, f: impl FnMut(&D)) {
        self.bvh.query_intersection(bounding_volume, f);
    }
}
