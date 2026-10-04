#[path = "../common/aabb.rs"]
mod aabb;
#[path = "../common/rand_vec3.rs"]
mod rand_vec3;
use rand_vec3::rand_vec3;

use std::collections::{HashMap, HashSet};

use aabb::Aabb;
use dbvh::{EnlargedBvh, IntersectSelf, NodeIndex};

use nalgebra as na;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn compact() {
    let mut id: usize = 0;

    let mut aabbs: HashMap<usize, (Aabb, NodeIndex)> = HashMap::new();
    let mut bvh: EnlargedBvh<Aabb, usize> = EnlargedBvh::new(0.1);

    let min_start = na::Vector3::new(-11.0, -11.3, -9.8);
    let max_start = na::Vector3::new(5.6, 9.7, 4.2);

    let max_size = na::Vector3::new(3.0, 3.0, 3.0);

    let mut rng = ChaCha8Rng::from_seed([123; 32]);

    // spawn new object
    for _ in 0..500 {
        let aabb = {
            let min = rand_vec3(&mut rng, min_start, max_start);
            let max = min + rand_vec3(&mut rng, na::Vector3::zeros(), max_size);
            Aabb { min, max }
        };
        let node_index = bvh.insert_leaf(aabb, id);
        aabbs.insert(id, (aabb, node_index));
        id += 1;
    }

    // compact the tree and test
    bvh.compact(|id, new_node_index| {
        aabbs.get_mut(id).unwrap().1 = new_node_index;
    });
    test_intersection(&mut rng, &bvh, &aabbs, min_start, max_start);

    // remove some of them
    let mut n_removed = 0;
    for _ in 0..500 {
        let to_remove = rng.random_range(0..aabbs.len());
        if let Some((_, node_index)) = aabbs.remove(&to_remove) {
            bvh.remove_leaf(node_index);
            n_removed += 1;
        }
    }
    assert!(n_removed > 0);

    // compact the tree and test
    bvh.compact(|id, new_node_index| {
        aabbs.get_mut(id).unwrap().1 = new_node_index;
    });
    test_intersection(&mut rng, &bvh, &aabbs, min_start, max_start);
}

fn test_intersection(
    rng: &mut ChaCha8Rng,
    bvh: &EnlargedBvh<Aabb, usize>,
    aabbs: &HashMap<usize, (Aabb, NodeIndex)>,
    min_pos: na::Vector3<f32>,
    max_pos: na::Vector3<f32>,
) {
    let mut intersection_bf: HashSet<usize> = HashSet::new();
    let mut intersection_bvh: HashSet<usize> = HashSet::new();
    for _ in 0..100 {
        let aabb = {
            let min = rand_vec3(rng, min_pos, max_pos);
            let max = min + rand_vec3(rng, na::Vector3::zeros(), na::Vector3::repeat(1.0));
            Aabb { min, max }
        };
        aabbs.iter().for_each(|(index, (other, _))| {
            if aabb.intersects(other) {
                intersection_bf.insert(*index);
            }
        });

        bvh.query_intersection(aabb, |index| {
            if aabb.intersects(&aabbs[&index].0) {
                intersection_bvh.insert(*index);
            }
        });

        assert_eq!(intersection_bf, intersection_bvh);
        intersection_bf.clear();
        intersection_bvh.clear();
    }
}
