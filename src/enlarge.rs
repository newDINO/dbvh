use crate::{BoundingVolume, Bvh, NodeIndex};

/// A trait for types that can be enlarged. Mainly for building an [`EnlargedBvh`].
///
/// See [`guide`](crate::guide) for examples of how to implement this trait.
pub trait Enlarge {
    /// Returns a bounding volume that is larger than itself.
    /// This is an optimization for dynamic bvh so that
    /// when object moves, as long as the bounding volume is within the enlarged leaf bounding volume,
    /// the leaf is not re-inserted.
    fn enlarge(&self, r: f32) -> Self;
}

/// A wrapper of [`Bvh`] that makes leaf bounding volume larger than object bounding volume.
///
/// See the documentation of [`Bvh`] and its methods for more details.
#[derive(Debug)]
pub struct EnlargedBvh<B, D> {
    pub(crate) bvh: Bvh<B, D>,
    enlargement: f32,
}
impl<B: Enlarge + BoundingVolume, D> EnlargedBvh<B, D> {
    pub fn new(enlargement: f32) -> Self {
        Self {
            bvh: Bvh::new(),
            enlargement,
        }
    }

    pub fn remove_leaf(&mut self, index: NodeIndex) {
        self.bvh.remove_leaf(index);
    }
    pub fn insert_leaf(&mut self, bounding_volume: B, user_data: D) -> NodeIndex {
        self.bvh
            .insert_leaf(bounding_volume.enlarge(self.enlargement), user_data)
    }
    pub fn update_leaf(&mut self, index: NodeIndex, bounding_volume: B) {
        let node = &self.bvh.nodes[index.0];
        if !node.bounding_volume.contains(&bounding_volume) {
            self.bvh
                .update_leaf(index, bounding_volume.enlarge(self.enlargement));
        }
    }
    pub fn compact(&mut self, new_index_setter: impl FnMut(&mut D, NodeIndex)) {
        self.bvh.compact(new_index_setter);
    }
}
