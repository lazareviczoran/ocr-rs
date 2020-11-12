use crate::image_ops;
use crate::polygon::expand_polygon;
use anyhow::Result;
use geo::prelude::*;
use geo::{LineString, MultiPolygon, Polygon};
use geo_clipper::Clipper;
use image::{GrayImage, Luma};
use imageproc::contours::{approx_poly_dp, arc_length, find_contours, get_distance, min_area_rect};
use imageproc::definitions::{HasWhite, Point};
use imageproc::drawing::{self};
use itertools::izip;
use num_traits::{clamp, Num, NumCast};
use std::sync::Mutex;
use tch::{Kind, Tensor};

lazy_static! {
    static ref COUNTER: Mutex<[usize; 1]> = Mutex::new([0]);
}

#[derive(Copy, Clone, Debug)]
pub struct MetricsItem {
    precision: f64,
    recall: f64,
    hmean: f64,
    gt_care: usize,
    det_care: usize,
    det_matched: usize,
}

pub struct PolygonScores {
    pub polygons: Vec<MultiPolygon<u32>>,
    pub scores: Vec<Vec<f64>>,
}

pub fn get_boxes_and_box_scores(pred: &Tensor, adjust_values: &Tensor) -> Result<PolygonScores> {
    let thresh = 0.6;
    let mut boxes_batch = Vec::new();
    let mut scores_batch = Vec::new();
    let segmentation = binarize(&pred, thresh)?;

    for batch_index in 0..pred.size()[0] {
        let (boxes, scores) = get_polygons_from_bitmap(
            &pred.get(batch_index),
            &segmentation.get(batch_index),
            &adjust_values.get(batch_index),
        )?;
        boxes_batch.push(boxes);
        scores_batch.push(scores);
    }
    Ok(PolygonScores {
        polygons: boxes_batch,
        scores: scores_batch,
    })
}

pub fn get_polygons_from_bitmap(
    pred: &Tensor,
    bitmap: &Tensor,
    adjust_values: &Tensor,
) -> Result<(MultiPolygon<u32>, Vec<f64>)> {
    let max_candidates = 1000;
    let box_thresh = 0.7;
    let save_pred_to_file = false;
    let min_size = 5.;

    // convert bitmap to GrayImage
    let image = image_ops::convert_tensor_to_image(&(bitmap.get(0) * 255))?;
    if save_pred_to_file {
        image.save(&format!(
            "text_detection_results/prediction_binarized{}.png",
            COUNTER.lock().unwrap()[0]
        ))?;
        COUNTER.lock().unwrap()[0] += 1;
    }

    let contours = find_contours(&image)
        .into_iter()
        .map(|c| c.points)
        .collect::<Vec<Vec<Point<u32>>>>();
    let num_contours = contours.len().min(max_candidates);
    let mut boxes = Vec::with_capacity(num_contours);
    let mut scores = Vec::with_capacity(num_contours);

    for contour in &contours {
        let mut epsilon = 0.01 * arc_length(&contour, true);
        if epsilon == 0. {
            epsilon = 0.01;
        }
        let mut points = approx_poly_dp(&contour, epsilon, true);
        if points.len() > 1 && points[0] == points[points.len() - 1] {
            points.pop();
        }

        if points.len() < 4 {
            continue;
        }
        let score = box_score_fast(&pred.get(0), &points)?;
        if box_thresh > score {
            continue;
        }
        let expanded = expand_polygon(&points, 2.).unwrap();
        let (_, sside) = get_min_area_bounding_box(&expanded);
        if sside < min_size {
            continue;
        }

        boxes.push(Polygon::new(
            LineString::from(
                expanded
                    .iter()
                    .map(|p| {
                        (
                            (p.x as f64 / adjust_values.double_value(&[0])).round() as u32,
                            (p.y as f64 / adjust_values.double_value(&[1])).round() as u32,
                        )
                    })
                    .collect::<Vec<(u32, u32)>>(),
            ),
            vec![],
        ));
        scores.push(score);
    }

    Ok((MultiPolygon::from(boxes), scores))
}

pub fn binarize(pred: &Tensor, thresh: f64) -> Result<Tensor> {
    Ok(pred.gt(thresh).to_kind(Kind::Uint8))
}

pub fn get_min_area_bounding_box<T>(contour: &[Point<T>]) -> (Vec<Point<T>>, f64)
where
    T: Num + NumCast + PartialEq + Eq + Copy + std::cmp::Ord,
{
    let mut b_box = min_area_rect(contour);
    b_box.sort_by(|a, b| a.x.cmp(&b.x));
    let i1 = if b_box[1].y > b_box[0].y { 0 } else { 1 };
    let i2 = if b_box[3].y > b_box[2].y { 2 } else { 3 };
    let i3 = if b_box[3].y > b_box[2].y { 3 } else { 2 };
    let i4 = if b_box[1].y > b_box[0].y { 1 } else { 0 };

    let res = vec![b_box[i1], b_box[i2], b_box[i3], b_box[i4]];
    let w = get_distance(&res[0], &res[1]);
    let h = get_distance(&res[0], &res[3]);
    (res, w.min(h))
}

pub fn box_score_fast(bitmap: &Tensor, points: &[Point<u32>]) -> Result<f64> {
    let size = bitmap.size();
    let w = size[size.len() - 2] as u32;
    let h = size[size.len() - 1] as u32;
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        points.iter().fold((u32::MAX, 0, u32::MAX, 0), |acc, p| {
            (
                acc.0.min(p.x),
                acc.1.max(p.x),
                acc.2.min(p.y),
                acc.3.max(p.y),
            )
        });
    min_x = clamp(min_x, 0, w - 1);
    max_x = clamp(max_x, 0, w - 1);
    min_y = clamp(min_y, 0, h - 1);
    max_y = clamp(max_y, 0, h - 1);
    let mut mask_image = GrayImage::new((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
    let moved_points: Vec<Point<i32>> = points
        .iter()
        .map(|p| Point::new((p.x - min_x) as i32, (p.y - min_y) as i32))
        .collect();
    drawing::draw_polygon_mut(&mut mask_image, &moved_points, Luma::white());

    let mask = (image_ops::convert_image_to_tensor(&mask_image)? / 255).to_kind(Kind::Uint8);

    let partial_bitmap = bitmap
        .narrow(0, min_y as i64, (max_y - min_y + 1) as i64)
        .narrow(1, min_x as i64, (max_x - min_x + 1) as i64)
        * &mask;

    let mean = partial_bitmap.sum(Kind::Double) / mask.sum(Kind::Double);

    Ok(mean.double_value(&[]))
}

//  Args:
//      polygons: tensor of shape (N, K, M, 2), the polygons of objective regions.
//      ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
//      pred: vector of length N of predicted polygons (each tensor of shape (M, 2) coresponds to a single image)
//      scores: vector of length N of predicted polygons scores
pub fn validate_measure(
    polygons: &[MultiPolygon<u32>],
    ignore_tags: &[Vec<bool>],
    pred: &[MultiPolygon<u32>],
    scores: &[Vec<f64>],
) -> Result<Vec<MetricsItem>> {
    let box_thresh = 0.6;
    let mut result = Vec::with_capacity(polygons.len());
    for (curr_polygons, curr_ignore_flags, curr_pred, curr_scores) in izip!(
        polygons.iter(),
        ignore_tags.iter(),
        pred.iter(),
        scores.iter()
    ) {
        let mut pred_polygons = Vec::new();
        for (&score, pred) in curr_scores.iter().zip(curr_pred.0.iter()) {
            if score >= box_thresh {
                pred_polygons.push(pred.clone());
            }
        }

        result.push(evaluate_image(
            &curr_polygons,
            &curr_ignore_flags,
            &MultiPolygon::from(pred_polygons),
        )?);
    }
    Ok(result)
}

pub fn gather_measure(metrics: &[Vec<MetricsItem>]) -> Result<(f64, f64, f64)> {
    let raw_metrics = metrics.iter().fold(Vec::new(), |mut acc, batch_metrics| {
        acc.append(&mut batch_metrics.to_vec());
        acc
    });

    Ok(combine_results(&raw_metrics)?)
}

pub fn combine_results(results: &[MetricsItem]) -> Result<(f64, f64, f64)> {
    let mut num_global_care_gt = 0;
    let mut num_global_care_det = 0;
    let mut matched_sum = 0;

    for res in results {
        num_global_care_gt += res.gt_care;
        num_global_care_det += res.det_care;
        matched_sum += res.det_matched;
    }
    let mut method_recall = 0.;
    if num_global_care_gt != 0 {
        method_recall = matched_sum as f64 / num_global_care_gt as f64;
    };
    let mut method_precision = 0.;
    if num_global_care_det != 0 {
        method_precision = matched_sum as f64 / num_global_care_det as f64;
    };
    let mut method_hmean = 0.;
    if method_recall + method_precision != 0. {
        method_hmean = 2. * (method_recall * method_precision) / (method_recall + method_precision);
    };
    Ok((method_precision, method_recall, method_hmean))
}

pub fn evaluate_image(
    gt_points: &MultiPolygon<u32>,
    ignore_flags: &[bool],
    pred: &MultiPolygon<u32>,
) -> Result<MetricsItem> {
    let area_precision_constraint = 0.5;
    let iou_constraint = 0.5;

    let mut det_matched = 0;

    let mut gt_polys = Vec::new();
    let mut det_polys = Vec::new();

    let mut gt_dont_care_pols_num = Vec::new();
    let mut det_dont_care_pols_num = Vec::new();

    let mut pairs = Vec::new();
    let mut det_matched_nums = Vec::new();

    for (n, curr_gt_points) in gt_points.0.iter().enumerate() {
        let ext_poly = curr_gt_points.exterior();
        let polygon = Polygon::new(
            LineString::from(
                ext_poly
                    .points_iter()
                    .take(ext_poly.num_coords() - 1)
                    .map(|p| (p.x() as f64, p.y() as f64))
                    .collect::<Vec<(f64, f64)>>(),
            ),
            vec![],
        );

        gt_polys.push(polygon);
        if ignore_flags[n] {
            gt_dont_care_pols_num.push(gt_polys.len() - 1);
        }
    }

    for pred_tensor in &pred.0 {
        let ext_pred = pred_tensor.exterior();
        let polygon = Polygon::new(
            LineString::from(
                ext_pred
                    .points_iter()
                    .take(ext_pred.num_coords() - 1)
                    .map(|p| (p.x() as f64, p.y() as f64))
                    .collect::<Vec<(f64, f64)>>(),
            ),
            vec![],
        );
        det_polys.push(polygon.clone());
        if !gt_dont_care_pols_num.is_empty() {
            for &dont_care_poly in &gt_dont_care_pols_num {
                let intersected_area = get_intersection(&gt_polys[dont_care_poly], &polygon)?;
                let pd_area = polygon.unsigned_area();
                let precision = if pd_area == 0. {
                    0.
                } else {
                    intersected_area / pd_area
                };
                if precision > area_precision_constraint {
                    det_dont_care_pols_num.push(det_polys.len() - 1);
                    break;
                }
            }
        }
    }

    if !gt_polys.is_empty() && !det_polys.is_empty() {
        let size = [gt_polys.len(), det_polys.len()];
        let mut iou_mat = vec![vec![0.; size[1]]; size[0]];
        let mut gt_rect_mat = vec![0; gt_polys.len()];
        let mut det_rect_mat = vec![0; det_polys.len()];
        for gt_num in 0..gt_polys.len() {
            for det_num in 0..det_polys.len() {
                iou_mat[gt_num][det_num] =
                    get_intersection_over_union(&det_polys[det_num], &gt_polys[gt_num])?;

                if gt_rect_mat[gt_num] == 0
                    && det_rect_mat[det_num] == 0
                    && !gt_dont_care_pols_num.contains(&gt_num)
                    && !gt_dont_care_pols_num.contains(&det_num)
                    && iou_mat[gt_num][det_num] > iou_constraint
                {
                    gt_rect_mat[gt_num] = 1;
                    det_rect_mat[det_num] = 1;
                    det_matched += 1;
                    pairs.push((gt_num, det_num));
                    det_matched_nums.push(det_num);
                }
            }
        }
    }

    let num_gt_care = gt_polys.len() - gt_dont_care_pols_num.len();
    let num_det_care = det_polys.len() - det_dont_care_pols_num.len();
    let recall;
    let precision;
    if num_gt_care == 0 {
        recall = 1.;
        precision = if num_det_care > 0 { 0. } else { 1. };
    } else {
        recall = det_matched as f64 / num_gt_care as f64;
        precision = if num_det_care == 0 {
            0.
        } else {
            det_matched as f64 / num_det_care as f64
        };
    }

    let hmean;
    if precision + recall == 0. {
        hmean = 0.;
    } else {
        hmean = 2. * precision * recall / (precision + recall);
    };

    Ok(MetricsItem {
        precision,
        recall,
        hmean,
        gt_care: num_gt_care,
        det_care: num_det_care,
        det_matched,
    })
}

fn get_intersection(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    let intersection = poly1.intersection(poly2, 1.);
    Ok(intersection.unsigned_area())
}

fn get_union(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    let union = poly1.union(poly2, 1.);
    Ok(union.unsigned_area())
}

fn get_intersection_over_union(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    Ok(get_intersection(poly1, poly2)? / get_union(poly1, poly2)?)
}

#[cfg(test)]
mod tests {
    use super::image_ops::convert_image_to_tensor;
    use super::*;
    use image::open;

    #[test]
    fn get_min_area_bounding_box_test() {
        let res = get_min_area_bounding_box(&[
            Point::new(141, 24),
            Point::new(61, 16),
            Point::new(57, 53),
            Point::new(137, 61),
        ]);
        assert_eq!(
            res.0,
            vec![
                Point::new(60, 15),
                Point::new(142, 23),
                Point::new(138, 62),
                Point::new(57, 54)
            ]
        );
        assert!((res.1 - 39.11521443121589).abs() < f64::EPSILON);
    }

    #[test]
    fn box_score_with_whole_matrix_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(0, 0),
            Point::new(4, 0),
            Point::new(4, 4),
            Point::new(0, 4),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 10. / 25.);
        Ok(())
    }

    #[test]
    fn box_score_partial_matrix_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(1, 0),
            Point::new(4, 0),
            Point::new(4, 3),
            Point::new(1, 3),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 8. / 16.);
        Ok(())
    }

    #[test]
    fn box_score_partial_matrix_test_2() -> Result<()> {
        let values = vec![
            0, 0, 0, 1, 0, //
            0, 0, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 0, 0, //
            0, 1, 1, 0, 0, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        let points = vec![
            Point::new(2, 0),
            Point::new(4, 1),
            Point::new(2, 4),
            Point::new(1, 3),
        ];
        assert_eq!(box_score_fast(&pred, &points)?, 9. / 12.);
        Ok(())
    }

    #[test]
    fn binarize_test() -> Result<()> {
        let values = vec![
            0.01, 0.2, 0.57, 0.58, 0.18, //
            0.39, 0.01, 0.61, 1.0, 0.42, //
            0.4, 0.94, 0.835, 0.793, 0.32, //
            0.57, 0.77, 0.62, 0.51, 0.29, //
            0.11, 0.69, 0.59, 0.21, 0.35, //
        ];
        let pred = Tensor::of_slice(&values).view([5, 5]);
        assert_eq!(
            binarize(&pred, 0.57)?,
            Tensor::of_slice(&[
                0, 0, 0, 1, 0, //
                0, 0, 1, 1, 0, //
                0, 1, 1, 1, 0, //
                0, 1, 1, 0, 0, //
                0, 1, 1, 0, 0, //
            ])
            .view((5, 5))
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_without_adjustments() -> Result<()> {
        let polygons_image = open("test_data/gt_shrinked_img55.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]);
        let expected_boxes = MultiPolygon(vec![
            // 1
            Polygon::new(
                LineString::from(vec![
                    (562, 75),
                    (559, 108),
                    (532, 108),
                    (427, 105),
                    (435, 68),
                ]),
                vec![],
            ),
            // 2
            Polygon::new(
                LineString::from(vec![
                    (547, 178),
                    (515, 255),
                    (404, 212),
                    (287, 226),
                    (263, 160),
                    (407, 125),
                ]),
                vec![],
            ),
            // 3
            Polygon::new(
                LineString::from(vec![
                    (448, 245),
                    (450, 301),
                    (427, 301),
                    (345, 292),
                    (332, 233),
                ]),
                vec![],
            ),
            // 4
            Polygon::new(
                LineString::from(vec![
                    (400, 322),
                    (534, 303),
                    (550, 361),
                    (401, 385),
                    (263, 319),
                    (278, 271),
                ]),
                vec![],
            ),
        ]);
        let expected_scores = vec![
            0.9819034852546917,
            0.9938022931515339,
            0.9911894273127754,
            0.9923459624952162,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_with_2x_adjustments() -> Result<()> {
        let polygons_image = open("test_data/gt_shrinked_img55.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // resized by w x 2, h x 2
        let adjust_values = Tensor::of_slice(&[2., 2.]);
        let expected_boxes = MultiPolygon::from(vec![
            // 1
            Polygon::new(
                LineString::from(vec![(281, 38), (280, 54), (266, 54), (214, 53), (218, 34)]),
                vec![],
            ),
            // 2
            Polygon::new(
                LineString::from(vec![
                    (274, 89),
                    (258, 128),
                    (202, 106),
                    (144, 113),
                    (132, 80),
                    (204, 63),
                ]),
                vec![],
            ),
            // 3
            Polygon::new(
                LineString::from(vec![
                    (224, 123),
                    (225, 151),
                    (214, 151),
                    (173, 146),
                    (166, 117),
                ]),
                vec![],
            ),
            // 4
            Polygon::new(
                LineString::from(vec![
                    (200, 161),
                    (267, 152),
                    (275, 181),
                    (201, 193),
                    (132, 160),
                    (139, 136),
                ]),
                vec![],
            ),
        ]);
        let expected_scores = vec![
            0.9819034852546917,
            0.9938022931515339,
            0.9911894273127754,
            0.9923459624952162,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn evaluate_image_test_one_matching_polygon() -> Result<()> {
        let gt_polygons_tensor = MultiPolygon::from(vec![
            // poly 1
            Polygon::new(
                LineString::from(vec![(0, 0), (10, 0), (10, 10), (0, 10)]),
                vec![],
            ),
            // poly 2
            Polygon::new(
                LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                vec![],
            ),
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiPolygon::from(vec![Polygon::new(
            LineString::from(vec![(1, 1), (10, 0), (10, 10), (0, 10)]),
            vec![],
        )]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

        assert_eq!(metrics.gt_care, 2);
        assert_eq!(metrics.det_care, 1);
        assert_eq!(metrics.det_matched, 1);
        assert!((metrics.precision - 1.).abs() < f64::EPSILON);
        assert!((metrics.recall - 0.5).abs() < f64::EPSILON);
        assert!((metrics.hmean - 0.6666666666666666).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn evaluate_image_test_with_ignored_polygons() -> Result<()> {
        let gt_polygons_tensor = MultiPolygon::from(vec![
            // poly 1
            Polygon::new(
                LineString::from(vec![(0, 0), (10, 0), (10, 10), (0, 10)]),
                vec![],
            ),
            // poly 2
            Polygon::new(
                LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                vec![],
            ),
        ]);
        let ignore_polygons = [true, true];
        let pred = MultiPolygon::from(vec![Polygon::new(
            LineString::from(vec![(1, 1), (10, 0), (10, 10), (0, 10)]),
            vec![],
        )]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

        assert_eq!(metrics.gt_care, 0);
        assert_eq!(metrics.det_care, 0);
        assert_eq!(metrics.det_matched, 0);
        assert!((metrics.precision - 1.).abs() < f64::EPSILON);
        assert!((metrics.recall - 1.).abs() < f64::EPSILON);
        assert!((metrics.hmean - 1.).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn evaluate_image_test_with_both_matched_polygons() -> Result<()> {
        let gt_polygons_tensor = MultiPolygon::from(vec![
            // poly 1
            Polygon::new(
                LineString::from(vec![(0, 0), (10, 0), (10, 10), (0, 10)]),
                vec![],
            ),
            // poly 2
            Polygon::new(
                LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                vec![],
            ),
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiPolygon::from(vec![
            Polygon::new(
                LineString::from(vec![(1, 1), (10, 0), (10, 10), (0, 10)]),
                vec![],
            ),
            Polygon::new(
                LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                vec![],
            ),
        ]);
        let metrics = evaluate_image(&gt_polygons_tensor, &ignore_polygons, &pred)?;

        assert_eq!(metrics.gt_care, 2);
        assert_eq!(metrics.det_care, 2);
        assert_eq!(metrics.det_matched, 2);
        assert!((metrics.precision - 1.).abs() < f64::EPSILON);
        assert!((metrics.recall - 1.).abs() < f64::EPSILON);
        assert!((metrics.hmean - 1.).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn validate_measure_test() -> Result<()> {
        let gt_polygons_tensor = vec![
            // image 1
            MultiPolygon::from(vec![
                // poly 1
                Polygon::new(
                    LineString::from(vec![(0, 0), (10, 0), (10, 10), (0, 10)]),
                    vec![],
                ),
                // poly 2
                Polygon::new(
                    LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                    vec![],
                ),
            ]),
            // image 2
            MultiPolygon::from(vec![
                // poly 1
                Polygon::new(
                    LineString::from(vec![(0, 0), (10, 0), (10, 10), (0, 10)]),
                    vec![],
                ),
                // poly 2
                Polygon::new(
                    LineString::from(vec![(20, 20), (30, 20), (30, 30), (20, 30)]),
                    vec![],
                ),
            ]),
        ];
        let ignore_polygons = [
            vec![false, false], // image 1
            vec![false, false], // image 2
        ];
        let pred = vec![
            MultiPolygon::from(vec![Polygon::new(
                LineString::from(vec![(1, 1), (10, 0), (10, 10), (0, 10)]),
                vec![],
            )]),
            MultiPolygon::from(vec![Polygon::new(
                LineString::from(vec![(45, 61), (47, 41), (60, 60), (39, 48)]),
                vec![],
            )]),
        ];
        let scores = [vec![0.9], vec![0.9]];
        let metrics = validate_measure(&gt_polygons_tensor, &ignore_polygons, &pred, &scores)?;
        assert_eq!(metrics.len(), 2);

        // image 1 metrics (having 1 matched polygon)
        assert_eq!(metrics[0].gt_care, 2);
        assert_eq!(metrics[0].det_care, 1);
        assert_eq!(metrics[0].det_matched, 1);
        assert!((metrics[0].precision - 1.).abs() < f64::EPSILON);
        assert!((metrics[0].recall - 0.5).abs() < f64::EPSILON);
        assert!((metrics[0].hmean - 0.6666666666666666).abs() < f64::EPSILON);

        // image 2 metrics (having 0 matched polygons)
        assert_eq!(metrics[1].gt_care, 2);
        assert_eq!(metrics[1].det_care, 1);
        assert_eq!(metrics[1].det_matched, 0);
        assert!(metrics[1].precision < f64::EPSILON);
        assert!(metrics[1].recall < f64::EPSILON);
        assert!(metrics[1].hmean < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn combine_results_test() -> Result<()> {
        let metrics = [
            MetricsItem {
                precision: 1.,
                recall: 0.5,
                hmean: 0.6666666666666666,
                gt_care: 2,
                det_care: 1,
                det_matched: 1,
            },
            MetricsItem {
                precision: 1.,
                recall: 1.,
                hmean: 1.,
                gt_care: 0,
                det_care: 0,
                det_matched: 0,
            },
            MetricsItem {
                precision: 1.,
                recall: 1.,
                hmean: 1.,
                gt_care: 2,
                det_care: 2,
                det_matched: 2,
            },
            MetricsItem {
                precision: 0.3333333333333333,
                recall: 0.2,
                hmean: 0.25,
                gt_care: 5,
                det_care: 3,
                det_matched: 1,
            },
        ];
        assert_eq!(
            combine_results(&metrics)?,
            (0.6666666666666666, 0.4444444444444444, 0.5333333333333333)
        );
        Ok(())
    }

    #[test]
    fn gather_measure_test() -> Result<()> {
        let metrics = [
            vec![
                MetricsItem {
                    precision: 1.,
                    recall: 0.5,
                    hmean: 0.6666666666666666,
                    gt_care: 2,
                    det_care: 1,
                    det_matched: 1,
                },
                MetricsItem {
                    precision: 1.,
                    recall: 1.,
                    hmean: 1.,
                    gt_care: 0,
                    det_care: 0,
                    det_matched: 0,
                },
            ],
            vec![
                MetricsItem {
                    precision: 1.,
                    recall: 1.,
                    hmean: 1.,
                    gt_care: 2,
                    det_care: 2,
                    det_matched: 2,
                },
                MetricsItem {
                    precision: 0.3333333333333333,
                    recall: 0.2,
                    hmean: 0.25,
                    gt_care: 5,
                    det_care: 3,
                    det_matched: 1,
                },
            ],
        ];
        assert_eq!(
            gather_measure(&metrics)?,
            (0.6666666666666666, 0.4444444444444444, 0.5333333333333333)
        );
        Ok(())
    }
}
