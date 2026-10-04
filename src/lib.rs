//! A dynamic bvh using generic [`BoundingVolume`] with zero dependencies.
//!
//! The algorithm is from `dynamic_tree.c` of [box2d](https://github.com/erincatto/box2d) (with some modifications).
//! There is an excellent [GDC talk](https://box2d.org/files/ErinCatto_DynamicBVH_Full.pdf) about this.
//!
//! You may use it with whatever bounding volume you want: 3d AABB, 2d AABB, bounding sphere, etc.
//! See user [`guide`] for how to use this crate with a specific bounding volume implementation.
//!
//! A bvh is a data structure for accelerating various spatial queries such as ray casting, intersection test, etc.
//! Bvh intersection test is about 7 times faster than brute force search when there are 1K objects,
//! and about 1000 times faster when there are 1M objects.
//! Bvh ray cast is about 10 times faster than brute force method with 1000 objects.
//!
//! Dynamic bvh supports removing and reinserting leaves, while still maitaining a balanced tree for fast spatial queries.
//! This allows objects to move around by updating the bounding volume of their corresponding leaves.
//! It is recommended to use [`EnlargedBvh`] for dynamic bvh to keep dynamic objects from being reinserted every tick.
//! It uses a slightly larger bounding volume
//! so that as long as the object doesn't move out of this enlarged bounding volume,
//! its corresponding leaf is not reinserted.
//!
//! A simple example:
//! ```
//! #[path = "../common/aabb.rs"]
//! mod aabb;
//! use aabb::{Aabb, AabbVector};
//! use dbvh::{EnlargedBvh, IntersectSelf};
//! use nalgebra as na;
//!
//! // create object list and bvh
//! let mut list: Vec<Aabb> = Vec::new();
//! let mut bvh: EnlargedBvh<Aabb, usize> = EnlargedBvh::new(0.1);
//!
//! // insert aabb
//! let aabb1 = Aabb {
//!     min: na::Vector3::repeat(0.0),
//!     max: na::Vector3::repeat(1.0),
//! };
//! let index = list.len();
//! list.push(aabb1);
//! bvh.insert_leaf(aabb1, index);
//!
//! // intersection test
//! let aabb2 = Aabb {
//!     min: na::Vector3::repeat(0.5),
//!     max: na::Vector3::repeat(1.5),
//! };
//!
//! let mut all_intersections = Vec::new();
//! bvh.query_intersection(aabb2, |index| {
//!     // Because this is an enlarged bvh,
//!     // index passed to this closure does not always report a real intersection.
//!     if aabb2.intersects(&list[*index]) {
//!         all_intersections.push(*index);
//!     }
//! });
//!
//! assert_eq!(all_intersections, [0]);
//! ```

// Original license of box2d:
//
// MIT License
//
// Copyright (c) 2022 Erin Catto
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

mod enlarge;
pub mod guide;
mod intersection;
mod ray_cast;
mod slot_pool;

pub use enlarge::{Enlarge, EnlargedBvh};
pub use intersection::IntersectSelf;
pub use ray_cast::RayCast;

use slot_pool::{SlotPool, SlotPoolHandle};
use std::fmt::Debug;

/// A basic trait for a bounding volume.
///
/// See [`guide`] for examples of how to implement this trait.
pub trait BoundingVolume {
    type Point: Vector;
    /// Returns the smallest [`BoundingVolume`] that contains both `self` and `other`
    fn union(&self, other: &Self) -> Self;

    /// Normally this is surface area of the bounding volume,
    /// but you can also mulitply it by a coefficient.
    /// For example, for sphere, the surface area is `4.0 * PI * r * r`,
    /// but you can also return `r * r` here as an optimization.
    fn surface_area_heuristic(&self) -> f32;

    /// Position of the centroid of the [`BoundingVolume`].
    fn center(&self) -> Self::Point;

    /// Returns whether `self` can contain another [`BoundingVolume`].
    /// Currently only [`EnlargedBvh`] uses this.
    fn contains(&self, other: &Self) -> bool;
}

/// A basic trait for vectors.
///
/// See [`guide`] for examples of how to implement this trait.
pub trait Vector {
    /// Normally this is the Euclidean distance between two point,
    /// but you can also returns the squared distance as an optimization.
    fn distance_heuristic(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, Debug)]
enum NodeType<D> {
    Internal {
        child1: NodeIndex,
        child2: NodeIndex,
    },
    Leaf(D),
}
impl<D> NodeType<D> {
    fn is_leaf(&self) -> bool {
        match self {
            Self::Internal { .. } => false,
            Self::Leaf(_) => true,
        }
    }
    fn is_internal(&self) -> bool {
        match self {
            Self::Internal { .. } => true,
            Self::Leaf(_) => false,
        }
    }
    fn as_internal(&self) -> (NodeIndex, NodeIndex) {
        match self {
            Self::Internal { child1, child2 } => (*child1, *child2),
            _ => unreachable!(),
        }
    }
    fn as_internal_mut(&mut self) -> (&mut NodeIndex, &mut NodeIndex) {
        match self {
            Self::Internal { child1, child2 } => (child1, child2),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeIndex(SlotPoolHandle);

impl NodeIndex {
    const NULL: Self = Self(SlotPoolHandle::NULL);
}

#[derive(Clone, Debug)]
pub struct Node<B, D> {
    ty: NodeType<D>,
    parent_index: NodeIndex,
    bounding_volume: B,
}

/// A generic bounding volume hierarchy.
///
/// `B` is the type of the [`BoundingVolume`].
///
/// `D` is the type for user data stored in each leaf.
/// `D` should be small because its size affects the size of both leaf [`Node`] and internal [`Node`].
/// For example, it can be used to store the index of an object, entity id, or a pointer to a geometry.
#[derive(Debug)]
pub struct Bvh<B, D> {
    root_index: NodeIndex,
    nodes: SlotPool<Node<B, D>>,
}
impl<B: BoundingVolume, D> Default for Bvh<B, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> Bvh<B, D> {
    pub fn new() -> Self {
        Self {
            root_index: NodeIndex::NULL,
            nodes: SlotPool::new(),
        }
    }

    #[inline]
    unsafe fn get_node_cfg(&self, index: NodeIndex) -> &Node<B, D> {
        #[cfg(debug_assertions)]
        let node = self.nodes.get(index.0).unwrap();

        #[cfg(not(debug_assertions))]
        let node = unsafe { self.nodes.get_unchecked(index.0) };

        node
    }
}

impl<B: BoundingVolume, D> Bvh<B, D> {
    // pub fn nodes(&self) -> &SlotPool<Node> {
    //     &self.nodes
    // }

    pub fn update_leaf(&mut self, index: NodeIndex, bounding_volume: B) {
        let node = self.remove_leaf(index).unwrap();
        let NodeType::Leaf(user_data) = node.ty else {
            unreachable!("Node is not leaf")
        };
        self.insert_leaf_at(bounding_volume, user_data, index);
    }

    pub fn insert_leaf(&mut self, bounding_volume: B, user_data: D) -> NodeIndex {
        if self.root_index == NodeIndex::NULL {
            core::hint::cold_path();

            let node = Node {
                ty: NodeType::Leaf(user_data),
                parent_index: NodeIndex::NULL,
                bounding_volume,
            };
            let index = NodeIndex(self.nodes.insert(node));
            self.root_index = index;

            return index;
        }

        let new_leaf_index = NodeIndex(self.nodes.allocate_slot());

        self.insert_leaf_at(bounding_volume, user_data, new_leaf_index);

        new_leaf_index
    }

    fn insert_leaf_at(&mut self, bounding_volume: B, user_data: D, leaf_index: NodeIndex) {
        debug_assert_ne!(self.root_index, NodeIndex::NULL);

        // Stage 1: find the best sibling for the new leaf
        let best_sibling = self.find_best_sibling(&bounding_volume);

        // Stage 2: create a new parent
        let new_parent_index = NodeIndex(self.nodes.allocate_slot());

        let sibling = &mut self.nodes[best_sibling.0];

        let old_parent_index = sibling.parent_index;
        sibling.parent_index = new_parent_index;

        let new_parent = Node {
            ty: NodeType::Internal {
                child1: best_sibling,
                child2: leaf_index,
            },
            parent_index: old_parent_index,
            bounding_volume: bounding_volume.union(&sibling.bounding_volume),
        };
        let new_leaf = Node {
            ty: NodeType::Leaf(user_data),
            parent_index: new_parent_index,
            bounding_volume,
        };

        if old_parent_index == NodeIndex::NULL {
            core::hint::cold_path();
            self.root_index = new_parent_index;
        } else {
            let old_parent = self.nodes.get_mut(old_parent_index.0).unwrap();

            let (child1, child2) = old_parent.ty.as_internal_mut();

            #[cfg(debug_assertions)]
            if *child1 == best_sibling {
                *child1 = new_parent_index;
            } else if *child2 == best_sibling {
                *child2 = new_parent_index;
            } else {
                unreachable!();
            }
            #[cfg(not(debug_assertions))]
            if *child1 == best_sibling {
                *child1 = new_parent_index;
            } else {
                *child2 = new_parent_index;
            }
        }

        if self
            .nodes
            .insert_at(new_parent, new_parent_index.0)
            .is_err()
        {
            panic!("Inserting new_parent failed.");
        }
        if self.nodes.insert_at(new_leaf, leaf_index.0).is_err() {
            panic!("Inserting new_leaf failed")
        }

        // Stage 3: walk back up the tree refitting AABBs
        let mut index = self.nodes.get(leaf_index.0).unwrap().parent_index;
        while index != NodeIndex::NULL {
            let node = &self.nodes.get(index.0).unwrap();
            let parent_index = node.parent_index;

            let (child1, child2) = node.ty.as_internal();
            let aabb1 = &self.nodes.get(child1.0).unwrap().bounding_volume;
            let aabb2 = &self.nodes.get(child2.0).unwrap().bounding_volume;

            self.nodes.get_mut(index.0).unwrap().bounding_volume = aabb1.union(aabb2);

            self.rotate_to_balance(index);

            index = parent_index;
        }
    }

    #[inline]
    fn find_best_sibling(&self, bounding_volume: &B) -> NodeIndex {
        let center = bounding_volume.center();

        let area = bounding_volume.surface_area_heuristic();

        let mut inherited_cost = 0.0;

        let mut best_sibling = self.root_index();
        let mut best_cost = f32::MAX;

        let mut index = self.root_index();

        let root_node = self.nodes.get(index.0).unwrap();
        let mut area_base = root_node.bounding_volume.surface_area_heuristic();
        let mut direct_cost = root_node
            .bounding_volume
            .union(bounding_volume)
            .surface_area_heuristic();

        loop {
            let node = unsafe { self.get_node_cfg(index) };
            let NodeType::Internal {
                child1: child1_index,
                child2: child2_index,
            } = node.ty
            else {
                break;
            };

            let cost = direct_cost + inherited_cost;

            if cost < best_cost {
                best_cost = cost;
                best_sibling = index;
            }

            inherited_cost += direct_cost - area_base;

            let child1 = unsafe { self.get_node_cfg(child1_index) };
            let child2 = unsafe { self.get_node_cfg(child2_index) };

            let leaf1 = child1.ty.is_leaf();
            let leaf2 = child2.ty.is_leaf();

            let direct_cost1 = child1
                .bounding_volume
                .union(bounding_volume)
                .surface_area_heuristic();
            let (area1, lower_cost1) = if leaf1 {
                let cost1 = direct_cost1 + inherited_cost;
                if cost1 < best_cost {
                    best_cost = cost1;
                    best_sibling = child1_index;
                }
                (0.0, f32::MAX)
            } else {
                let area1 = child1.bounding_volume.surface_area_heuristic();
                let lower_cost1 = inherited_cost + direct_cost1 + (area - area1).min(0.0);
                (area1, lower_cost1)
            };

            let direct_cost2 = child2
                .bounding_volume
                .union(bounding_volume)
                .surface_area_heuristic();
            let (area2, lower_cost2) = if leaf2 {
                let cost2 = direct_cost2 + inherited_cost;
                if cost2 < best_cost {
                    best_cost = cost2;
                    best_sibling = child2_index;
                }
                (0.0, f32::MAX)
            } else {
                let area2 = child2.bounding_volume.surface_area_heuristic();
                let lower_cost2 = inherited_cost + direct_cost2 + (area - area2).min(0.0);
                (area2, lower_cost2)
            };

            if leaf1 && leaf2 || best_cost <= lower_cost1 && best_cost <= lower_cost2 {
                break;
            }

            let (lower_cost1, lower_cost2) = if lower_cost1 == lower_cost2 && !leaf1 {
                core::hint::cold_path();

                debug_assert!(lower_cost1 < f32::MAX);
                debug_assert!(lower_cost2 < f32::MAX);

                let d1 = child1.bounding_volume.center().distance_heuristic(&center);
                let d2 = child2.bounding_volume.center().distance_heuristic(&center);

                (d1, d2)
            } else {
                (lower_cost1, lower_cost2)
            };

            if lower_cost1 < lower_cost2 && !leaf1 {
                index = child1_index;
                area_base = area1;
                direct_cost = direct_cost1;
            } else {
                index = child2_index;
                area_base = area2;
                direct_cost = direct_cost2;
            }

            // // If index is leaf, it can only be child2_index,
            // // and best_cost > lower_cost2 || best_cost > lower_cost1
            // debug_assert!(self.nodes[index.0].ty.is_internal());
        }

        best_sibling
    }

    fn rotate_to_balance(&mut self, index: NodeIndex) {
        let i = index.0;
        let node = &self.nodes[i];

        let NodeType::Internal {
            child1: child1_index,
            child2: child2_index,
        } = node.ty
        else {
            return;
        };
        let (c1i, c2i) = (child1_index.0, child2_index.0);

        let child1 = &self.nodes[c1i];
        let child2 = &self.nodes[c2i];

        match (&child1.ty, &child2.ty) {
            (NodeType::Leaf(_), NodeType::Leaf(_)) => {}
            (
                NodeType::Leaf(_),
                NodeType::Internal {
                    child1: grandchild3_index,
                    child2: grandchild4_index,
                },
            ) => {
                let grandchild3_index = *grandchild3_index;
                let grandchild4_index = *grandchild4_index;
                let (g3i, g4i) = (grandchild3_index.0, grandchild4_index.0);

                let base_cost = child2.bounding_volume.surface_area_heuristic();

                let aabb_c1g4 = child1
                    .bounding_volume
                    .union(&self.nodes[g4i].bounding_volume);
                let cost_c1g3 = aabb_c1g4.surface_area_heuristic();

                let aabb_c1g3 = child1
                    .bounding_volume
                    .union(&self.nodes[g3i].bounding_volume);
                let cost_c1g4 = aabb_c1g3.surface_area_heuristic();

                if base_cost < cost_c1g3 && base_cost < cost_c1g4 {
                    return;
                }

                self.nodes[c1i].parent_index = child2_index;

                if cost_c1g3 < cost_c1g4 {
                    *self.nodes[i].ty.as_internal_mut().0 = grandchild3_index;
                    *self.nodes[c2i].ty.as_internal_mut().0 = child1_index;

                    self.nodes[g3i].parent_index = index;

                    self.nodes[c2i].bounding_volume = aabb_c1g4;
                } else {
                    *self.nodes[i].ty.as_internal_mut().0 = grandchild4_index;
                    *self.nodes[c2i].ty.as_internal_mut().1 = child1_index;

                    self.nodes[g4i].parent_index = index;

                    self.nodes[c2i].bounding_volume = aabb_c1g3;
                }
            }
            (
                NodeType::Internal {
                    child1: grandchild1_index,
                    child2: grandchild2_index,
                },
                NodeType::Leaf(_),
            ) => {
                let grandchild1_index = *grandchild1_index;
                let grandchild2_index = *grandchild2_index;
                let (g1i, g2i) = (grandchild1_index.0, grandchild2_index.0);

                let base_cost = child1.bounding_volume.surface_area_heuristic();

                let aabb_c2g2 = child2
                    .bounding_volume
                    .union(&self.nodes[g2i].bounding_volume);
                let cost_c2g1 = aabb_c2g2.surface_area_heuristic();

                let aabb_c2g1 = child2
                    .bounding_volume
                    .union(&self.nodes[g1i].bounding_volume);
                let cost_c2g2 = aabb_c2g1.surface_area_heuristic();

                if base_cost < cost_c2g1 && base_cost < cost_c2g2 {
                    return;
                }

                self.nodes[c2i].parent_index = child1_index;

                if cost_c2g1 < cost_c2g2 {
                    *self.nodes[i].ty.as_internal_mut().1 = grandchild1_index;
                    *self.nodes[c1i].ty.as_internal_mut().0 = child2_index;

                    self.nodes[g1i].parent_index = index;

                    self.nodes[c1i].bounding_volume = aabb_c2g2;
                } else {
                    *self.nodes[i].ty.as_internal_mut().1 = grandchild2_index;
                    *self.nodes[c1i].ty.as_internal_mut().1 = child2_index;

                    self.nodes[g2i].parent_index = index;

                    self.nodes[c1i].bounding_volume = aabb_c2g1;
                }
            }
            (
                NodeType::Internal {
                    child1: grandchild1_index,
                    child2: grandchild2_index,
                },
                NodeType::Internal {
                    child1: grandchild3_index,
                    child2: grandchild4_index,
                },
            ) => {
                let g1i = grandchild1_index.0;
                let g2i = grandchild2_index.0;
                let g3i = grandchild3_index.0;
                let g4i = grandchild4_index.0;

                let area_c1 = child1.bounding_volume.surface_area_heuristic();
                let area_c2 = child2.bounding_volume.surface_area_heuristic();
                let base_cost = area_c1 + area_c2;

                enum RotationType {
                    None,
                    C1G3,
                    C1G4,
                    C2G1,
                    G2C2,
                }

                let mut best_cost = base_cost;
                let mut best_rotation = RotationType::None;

                let aabb_c1g4 = child1
                    .bounding_volume
                    .union(&self.nodes[g4i].bounding_volume);
                let cost_c1g3 = area_c1 + aabb_c1g4.surface_area_heuristic();
                if cost_c1g3 < best_cost {
                    best_cost = cost_c1g3;
                    best_rotation = RotationType::C1G3;
                }

                let aabb_c1g3 = child1
                    .bounding_volume
                    .union(&self.nodes[g3i].bounding_volume);
                let cost_c1g4 = area_c1 + aabb_c1g3.surface_area_heuristic();
                if cost_c1g4 < best_cost {
                    best_cost = cost_c1g4;
                    best_rotation = RotationType::C1G4;
                }

                let aabb_c2g2 = child2
                    .bounding_volume
                    .union(&self.nodes[g2i].bounding_volume);
                let cost_c2g1 = area_c2 + aabb_c2g2.surface_area_heuristic();
                if cost_c2g1 < best_cost {
                    best_cost = cost_c2g1;
                    best_rotation = RotationType::C2G1;
                }

                let aabb_c2g1 = child2
                    .bounding_volume
                    .union(&self.nodes[g1i].bounding_volume);
                let cost_c2g2 = area_c2 + aabb_c2g1.surface_area_heuristic();
                if cost_c2g2 < best_cost {
                    // best_cost = cost_c2g2;
                    best_rotation = RotationType::G2C2;
                }

                match best_rotation {
                    RotationType::None => {}
                    RotationType::C1G3 => {
                        *self.nodes[i].ty.as_internal_mut().0 = *grandchild3_index;
                        *self.nodes[c2i].ty.as_internal_mut().0 = child1_index;

                        self.nodes[c1i].parent_index = child2_index;
                        self.nodes[g3i].parent_index = index;

                        self.nodes[c2i].bounding_volume = aabb_c1g4;
                    }
                    RotationType::C1G4 => {
                        *self.nodes[i].ty.as_internal_mut().0 = *grandchild4_index;
                        *self.nodes[c2i].ty.as_internal_mut().1 = child1_index;

                        self.nodes[c1i].parent_index = child2_index;
                        self.nodes[g4i].parent_index = index;

                        self.nodes[c2i].bounding_volume = aabb_c1g3;
                    }
                    RotationType::C2G1 => {
                        *self.nodes[i].ty.as_internal_mut().1 = *grandchild1_index;
                        *self.nodes[c1i].ty.as_internal_mut().0 = child2_index;

                        self.nodes[c2i].parent_index = child1_index;
                        self.nodes[g1i].parent_index = index;

                        self.nodes[c1i].bounding_volume = aabb_c2g2;
                    }
                    RotationType::G2C2 => {
                        *self.nodes[i].ty.as_internal_mut().1 = *grandchild2_index;
                        *self.nodes[c1i].ty.as_internal_mut().1 = child2_index;

                        self.nodes[c2i].parent_index = child1_index;
                        self.nodes[g2i].parent_index = index;

                        self.nodes[c1i].bounding_volume = aabb_c2g1;
                    }
                }
            }
        }
    }

    fn root_index(&self) -> NodeIndex {
        self.root_index
    }

    pub fn remove_leaf(&mut self, index: NodeIndex) -> Option<Node<B, D>> {
        let node = self.nodes.get(index.0)?;

        if node.ty.is_internal() {
            panic!("Node {:?} to be removed is not leaf!", index.0);
        }

        if index == self.root_index() {
            self.root_index = NodeIndex::NULL;
            self.nodes.remove(index.0);
            return None;
        }

        let parent_index = node.parent_index;

        let parent = &self.nodes[parent_index.0];

        let (child1, child2) = parent.ty.as_internal();

        let sibling_index = if child1 == index {
            child2
        } else if child2 == index {
            child1
        } else {
            unreachable!()
        };

        let new_parent_index = if node.parent_index == self.root_index() {
            self.root_index = sibling_index;
            NodeIndex::NULL
        } else {
            let grand_parent_index = parent.parent_index;
            let grand_parent = &mut self.nodes[grand_parent_index.0];

            let (child1, child2) = grand_parent.ty.as_internal_mut();

            if *child1 == parent_index {
                *child1 = sibling_index;
            } else if *child2 == parent_index {
                *child2 = sibling_index;
            } else {
                unreachable!()
            }

            grand_parent_index
        };

        self.nodes[sibling_index.0].parent_index = new_parent_index;

        // Must remove node first, then remove parent.
        let node = self.nodes.remove(index.0);
        self.nodes.remove(parent_index.0);
        node
    }
}
