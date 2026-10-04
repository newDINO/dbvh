use crate::{Bvh, EnlargedBvh, NodeIndex, NodeType, Vector};

/// A trait for types that can perform ray casting against a bounding volume.
///
/// See [`guide`](crate::guide) for examples of how to implement this trait.
pub trait RayCast {
    type Vector: Vector;

    /// - Returns `Some(t)` if the ray intersects the volume. Returns `None` otherwise.
    /// - Only non-negative `t` values should be considered as valid hits.
    /// - For [`BoundingVolume`](crate::BoundingVolume), this should return `Some(0.0)` when the origin is inside the volume.
    fn ray_cast(&self, origin: &Self::Vector, dir: &Self::Vector) -> Option<f32>;
}

impl<V: Vector, B: RayCast<Vector = V>, D> Bvh<B, D> {
    /// This method traverses the BVH and returns the smallest ray parameter `t`
    /// for which the leaf callback reports a hit.
    ///
    /// # Parameters
    ///
    /// * `origin` — The origin of the ray.
    /// * `dir` — The direction of the ray. **Note:** `dir` is not normalized
    ///   automatically.
    /// * `leaf_ray_cast_f` — A callback invoked for each leaf node that may
    ///   contain a hit. It receives `(origin, dir, leaf_data)` and must return
    ///   `Some((t, custom_data))` for a valid hit with `t >= 0`, or `None` if the ray does not
    ///   hit the leaf's contents. The returned `t` must be in the same parameter
    ///   space as the values returned by `B::ray_cast`.
    ///   `custom_data: R` can be any desired data, e.g. `()`, normal of the intersecting face, index of the entity.
    ///
    #[inline]
    pub fn ray_cast<R>(
        &self,
        origin: &V,
        dir: &V,
        mut leaf_ray_cast_f: impl FnMut(&V, &V, &D) -> Option<(f32, R)>,
    ) -> Option<(f32, R)> {
        if self.root_index == NodeIndex::NULL {
            return None;
        }
        let root = &self.nodes[self.root_index.0];

        let mut result = None;

        if root.bounding_volume.ray_cast(origin, dir).is_some() {
            self.ray_cast_rec(
                self.root_index,
                origin,
                dir,
                &mut leaf_ray_cast_f,
                &mut result,
            )
        }

        result
    }

    fn ray_cast_rec<R>(
        &self,
        index: NodeIndex,
        origin: &V,
        dir: &V,
        leaf_ray_cast_f: &mut impl FnMut(&V, &V, &D) -> Option<(f32, R)>,
        result: &mut Option<(f32, R)>,
    ) {
        let node = &self.nodes[index.0];

        let result_compare = if let Some((t, _)) = result {
            *t
        } else {
            f32::MAX
        };

        match &node.ty {
            NodeType::Leaf(data) => {
                if let Some((t, r)) = leaf_ray_cast_f(origin, dir, data)
                    && t < result_compare
                {
                    *result = Some((t, r));
                }
            }
            NodeType::Internal { child1, child2 } => {
                let node1 = &self.nodes[child1.0];
                let cast1 = node1.bounding_volume.ray_cast(origin, dir);

                let node2 = &self.nodes[child2.0];
                let cast2 = node2.bounding_volume.ray_cast(origin, dir);

                match (cast1, cast2) {
                    (None, None) => {}
                    (Some(t), None) => {
                        if t < result_compare {
                            self.ray_cast_rec(*child1, origin, dir, leaf_ray_cast_f, result);
                        }
                    }
                    (None, Some(t)) => {
                        if t < result_compare {
                            self.ray_cast_rec(*child2, origin, dir, leaf_ray_cast_f, result);
                        }
                    }
                    (Some(t1), Some(t2)) => {
                        let (ia, ta, ib, tb) = if t1 <= t2 {
                            (*child1, t1, *child2, t2)
                        } else {
                            (*child2, t2, *child1, t1)
                        };
                        if ta < result_compare {
                            self.ray_cast_rec(ia, origin, dir, leaf_ray_cast_f, result);

                            if let Some((t, _)) = result {
                                if tb < *t {
                                    self.ray_cast_rec(ib, origin, dir, leaf_ray_cast_f, result);
                                }
                            } else {
                                self.ray_cast_rec(ib, origin, dir, leaf_ray_cast_f, result);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<V: Vector, B: RayCast<Vector = V>, D> EnlargedBvh<B, D> {
    #[inline]
    pub fn ray_cast<R>(
        &self,
        origin: &V,
        dir: &V,
        leaf_ray_cast_f: impl FnMut(&V, &V, &D) -> Option<(f32, R)>,
    ) -> Option<(f32, R)> {
        self.bvh.ray_cast(origin, dir, leaf_ray_cast_f)
    }
}
