use geo::prelude::*;
use geo::{CoordinateType, LineString, Polygon};
use geo_clipper::{Clipper, EndType, JoinType};
use imageproc::definitions::Point;
use num_traits::{Num, NumCast};

#[derive(PartialEq, Eq)]
pub enum ClippingType {
    Shrink,
    Expand,
}

pub fn clip_polygon<T: Num + NumCast + Eq + CoordinateType>(
    polygon: &[Point<T>],
    factor: f64,
    clipping_type: ClippingType,
) -> Option<Vec<Point<i32>>> {
    let prep_poly = Polygon::new(
        LineString::from(
            polygon
                .iter()
                .map(|p| (p.x.to_f64().unwrap(), p.y.to_f64().unwrap()))
                .collect::<Vec<(f64, f64)>>(),
        ),
        vec![],
    );
    let mut distance = prep_poly.unsigned_area() * factor / prep_poly.exterior().euclidean_length();
    if clipping_type == ClippingType::Shrink {
        distance *= -1.;
    }
    let clipped = prep_poly.offset(distance, JoinType::Miter(2.), EndType::ClosedPolygon, 1.);
    if clipped.0.is_empty() || clipped.0[0].exterior().num_coords() == 0 {
        return None;
    }
    let exterior_poly = clipped.0[0].exterior();
    let shrinked_points = exterior_poly
        .points_iter()
        .take(exterior_poly.num_coords() - 1)
        .map(|p| Point::new(p.x() as i32, p.y() as i32))
        .collect::<Vec<Point<i32>>>();
    Some(shrinked_points)
}

pub fn shrink_polygon<T: Num + NumCast + Eq + CoordinateType>(
    polygon: &[Point<T>],
    factor: f64,
) -> Option<Vec<Point<i32>>> {
    clip_polygon(polygon, factor, ClippingType::Shrink)
}

pub fn expand_polygon<T: Num + NumCast + Eq + CoordinateType>(
    polygon: &[Point<T>],
    factor: f64,
) -> Option<Vec<Point<i32>>> {
    clip_polygon(polygon, factor, ClippingType::Expand)
}
