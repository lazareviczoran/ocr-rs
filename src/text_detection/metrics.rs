use crate::image_ops::{self, BatchPolygons, MultiplePolygons};
use anyhow::Result;
use geo::prelude::*;
use geo::{LineString, Polygon};
use geo_booleanop::boolean::BooleanOp;
use image::{GrayImage, Luma};
use imageproc::contours::{approx_poly_dp, arc_length, find_contours, get_distance, min_area_rect};
use imageproc::definitions::{HasWhite, Point};
use imageproc::drawing::{self};
use num_traits::clamp;
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

pub fn get_boxes_and_box_scores(
    pred: &Tensor,
    adjust_values: &Tensor,
    is_output_polygon: bool,
) -> Result<(BatchPolygons, Vec<Vec<f64>>)> {
    let thresh = 0.6;
    let mut boxes_batch = Vec::new();
    let mut scores_batch = Vec::new();
    let segmentation = binarize(&pred, thresh)?;

    for batch_index in 0..pred.size()[0] {
        let (boxes, scores) = if is_output_polygon {
            get_polygons_from_bitmap(
                &pred.get(batch_index),
                &segmentation.get(batch_index),
                &adjust_values.get(batch_index),
            )?
        } else {
            get_boxes_from_bitmap(
                &pred.get(batch_index),
                &segmentation.get(batch_index),
                &adjust_values.get(batch_index),
            )?
        };
        boxes_batch.push(boxes);
        scores_batch.push(scores);
    }
    Ok((BatchPolygons(boxes_batch), scores_batch))
}

pub fn get_polygons_from_bitmap(
    pred: &Tensor,
    bitmap: &Tensor,
    adjust_values: &Tensor,
) -> Result<(MultiplePolygons, Vec<f64>)> {
    let max_candidates = 1000;
    let box_thresh = 0.7;
    let save_pred_to_file = false;

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
    let mut boxes = vec![image_ops::Polygon { points: vec![] }; num_contours];
    let mut scores = vec![0.; num_contours];

    for i in 0..num_contours {
        let mut epsilon = 0.01 * arc_length(&contours[i], true);
        if epsilon == 0. {
            epsilon = 0.01;
        }
        let mut points = approx_poly_dp(&contours[i], epsilon, true);
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

        boxes[i].points = points.iter().fold(Vec::new(), |mut acc, p| {
            acc.push(Point::new(
                (p.x as f64 / adjust_values.double_value(&[0])).round() as u32,
                (p.y as f64 / adjust_values.double_value(&[1])).round() as u32,
            ));
            acc
        });
        scores[i] = score;
    }

    Ok((MultiplePolygons(boxes), scores))
}

pub fn binarize(pred: &Tensor, thresh: f64) -> Result<Tensor> {
    Ok(pred.gt(thresh).to_kind(Kind::Uint8))
}

pub fn get_boxes_from_bitmap(
    pred: &Tensor,
    bitmap: &Tensor,
    adjust_values: &Tensor,
) -> Result<(MultiplePolygons, Vec<f64>)> {
    let max_candidates = 1000;
    let min_size = 3.;
    let box_thresh = 0.7;

    // convert bitmap to GrayImage
    let image = image_ops::convert_tensor_to_image(&(bitmap.get(0) * 255))?;

    let contours = find_contours(&image)
        .into_iter()
        .map(|c| c.points)
        .collect::<Vec<Vec<Point<u32>>>>();
    let num_contours = contours.len().min(max_candidates);
    let mut boxes = vec![image_ops::Polygon { points: vec![] }; num_contours];
    let mut scores = vec![0.; num_contours];

    for i in 0..num_contours {
        let (points, sside) = get_mini_area_bounding_box(&contours[i]);
        if sside < min_size {
            continue;
        }
        let score = box_score_fast(&pred.get(0), &points)?;
        if box_thresh > score {
            continue;
        }

        boxes[i].points = points.iter().fold(Vec::new(), |mut acc, p| {
            acc.push(Point::new(
                (p.x as f64 / adjust_values.double_value(&[0, 0])).round() as u32,
                (p.y as f64 / adjust_values.double_value(&[0, 1])).round() as u32,
            ));
            acc
        });
        scores[i] = score;
    }

    Ok((MultiplePolygons(boxes), scores))
}

pub fn get_mini_area_bounding_box(contour: &[Point<u32>]) -> (Vec<Point<u32>>, f64) {
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
        points
            .iter()
            .fold((std::u32::MAX, 0, std::u32::MAX, 0), |acc, p| {
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
    polygons: &BatchPolygons,
    ignore_tags: &[Vec<bool>],
    pred: &BatchPolygons,
    scores: &[Vec<f64>],
    idx: usize,
    // is_output_polygon: bool,
) -> Result<MetricsItem> {
    let is_output_polygon = true;
    let box_thresh = 0.6;
    let mut pred_polygons = Vec::new();
    for j in 0..pred.0[0].0.len() {
        if is_output_polygon || scores[0][j] >= box_thresh {
            pred_polygons.push(pred.0[0].0[j].clone());
        }
    }

    evaluate_image(
        &polygons.0[idx],
        &ignore_tags[idx],
        &MultiplePolygons(pred_polygons),
    )
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
    let method_recall = if num_global_care_gt == 0 {
        0.
    } else {
        matched_sum as f64 / num_global_care_gt as f64
    };
    let method_precision = if num_global_care_det == 0 {
        0.
    } else {
        matched_sum as f64 / num_global_care_det as f64
    };
    let method_hmean = if method_recall + method_precision == 0. {
        0.
    } else {
        2. * (method_recall * method_precision) / (method_recall + method_precision)
    };
    Ok((method_precision, method_recall, method_hmean))
}

pub fn evaluate_image(
    gt_points: &MultiplePolygons,
    ignore_flags: &[bool],
    pred: &MultiplePolygons,
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
        let polygon = Polygon::new(
            LineString::from(
                curr_gt_points
                    .points
                    .iter()
                    .map(|p| (p.x as f64, p.y as f64))
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
        let polygon = Polygon::new(
            LineString::from(
                pred_tensor
                    .points
                    .iter()
                    .map(|p| (p.x as f64, p.y as f64))
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
    let intersection = poly1.intersection(poly2);
    Ok(intersection.unsigned_area())
}

fn get_union(poly1: &Polygon<f64>, poly2: &Polygon<f64>) -> Result<f64> {
    let union = poly1.union(poly2);
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
    use imageproc::drawing::draw_polygon_mut;

    #[test]
    fn get_mini_area_bounding_box_test() {
        let res = get_mini_area_bounding_box(&[
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
    fn get_boxes_from_bitmap_test_perfect_match_with_2x_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        // perfect match
        let mut prediction_image = GrayImage::new(w as u32, h as u32);
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(60, 16),
                Point::new(141, 24),
                Point::new(137, 61),
                Point::new(57, 53),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(235, 17),
                Point::new(302, 21),
                Point::new(299, 59),
                Point::new(233, 54),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(7, 75),
                Point::new(148, 95),
                Point::new(145, 111),
                Point::new(4, 91),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(250, 79),
                Point::new(303, 85),
                Point::new(299, 126),
                Point::new(245, 120),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(25, 181),
                Point::new(125, 181),
                Point::new(125, 268),
                Point::new(25, 268),
            ],
            Luma::white(),
        );
        draw_polygon_mut(
            &mut prediction_image,
            &[
                Point::new(190, 250),
                Point::new(255, 185),
                Point::new(285, 215),
                Point::new(220, 280),
            ],
            Luma::white(),
        );
        let pred_tensor = (convert_image_to_tensor(&prediction_image)? / 255.).view((1, h, w));

        // 2x adjustment
        let adjust_values = Tensor::of_slice(&[2., 2.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            // expected boxes are reajdusted to the original image size (divided by adjust values)
            image_ops::Polygon {
                points: vec![
                    Point::new(30, 8),
                    Point::new(71, 12),
                    Point::new(69, 31),
                    Point::new(29, 27),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(118, 9),
                    Point::new(151, 11),
                    Point::new(150, 30),
                    Point::new(117, 27),
                ],
            }, // 2
            image_ops::Polygon {
                points: vec![
                    Point::new(4, 38),
                    Point::new(74, 48),
                    Point::new(73, 56),
                    Point::new(2, 46),
                ],
            }, // 3
            image_ops::Polygon {
                points: vec![
                    Point::new(125, 40),
                    Point::new(152, 43),
                    Point::new(150, 63),
                    Point::new(123, 60),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(13, 91),
                    Point::new(63, 91),
                    Point::new(63, 134),
                    Point::new(13, 134),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(95, 125),
                    Point::new(128, 93),
                    Point::new(143, 108),
                    Point::new(110, 140),
                ],
            }, // 6
        ]);
        let expected_scores = vec![1.; 6];
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_boxes_from_bitmap_test_partial_match_without_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            // expected boxes with score over thresh (0.7) should appear
            image_ops::Polygon { points: vec![] }, // 1
            image_ops::Polygon { points: vec![] }, // 2
            image_ops::Polygon { points: vec![] }, // 3
            image_ops::Polygon {
                points: vec![
                    Point::new(250, 79),
                    Point::new(303, 85),
                    Point::new(299, 126),
                    Point::new(245, 120),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(25, 181),
                    Point::new(125, 181),
                    Point::new(125, 268),
                    Point::new(25, 268),
                ],
            }, // 5
            image_ops::Polygon { points: vec![] }, // 6
        ]);
        let expected_scores = vec![0., 0., 0., 0.703405017921147, 0.7521377137713772, 0.];
        assert_eq!(
            get_boxes_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_without_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // no adjustment
        let adjust_values = Tensor::of_slice(&[1., 1.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(100, 20),
                    Point::new(90, 35),
                    Point::new(60, 25),
                    Point::new(90, 40),
                    Point::new(80, 55),
                    Point::new(101, 50),
                    Point::new(130, 60),
                    Point::new(115, 45),
                    Point::new(140, 30),
                    Point::new(120, 35),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(275, 20),
                    Point::new(265, 35),
                    Point::new(235, 25),
                    Point::new(265, 40),
                    Point::new(255, 55),
                    Point::new(276, 50),
                    Point::new(299, 58),
                    Point::new(299, 54),
                    Point::new(290, 45),
                    Point::new(299, 40),
                    Point::new(299, 34),
                    Point::new(295, 35),
                ],
            }, // 2
            image_ops::Polygon { points: vec![] }, // 3, it is ignored since its a triangle
            image_ops::Polygon {
                points: vec![
                    Point::new(250, 80),
                    Point::new(250, 120),
                    Point::new(296, 125),
                    Point::new(299, 125),
                    Point::new(299, 104),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(50, 181),
                    Point::new(25, 225),
                    Point::new(49, 268),
                    Point::new(100, 268),
                    Point::new(125, 225),
                    Point::new(100, 181),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(269, 200),
                    Point::new(240, 210),
                    Point::new(190, 250),
                    Point::new(220, 280),
                ],
            }, // 6
        ]);
        let expected_scores = vec![
            1.0,
            0.9980139026812314,
            0.0,
            0.9924146649810367,
            1.0,
            0.9978822532825075,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn get_polygons_from_bitmap_test_with_2x_adjustments() -> Result<()> {
        let polygons_image = open("test_data/polygon.png")?.to_luma();
        let h = polygons_image.height() as i64;
        let w = polygons_image.width() as i64;
        let bit_tensor = convert_image_to_tensor(&polygons_image)?;
        let bitmap_tensor = (&bit_tensor / 255.).to_kind(Kind::Uint8).view((1, h, w));

        let pred_tensor = (convert_image_to_tensor(&polygons_image)? / 255.).view((1, h, w));
        // resized by w x 2, h x 2
        let adjust_values = Tensor::of_slice(&[2., 2.]).view((1, 2));
        let expected_boxes = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(50, 10),
                    Point::new(45, 18),
                    Point::new(30, 13),
                    Point::new(45, 20),
                    Point::new(40, 28),
                    Point::new(51, 25),
                    Point::new(65, 30),
                    Point::new(58, 23),
                    Point::new(70, 15),
                    Point::new(60, 18),
                ],
            }, // 1
            image_ops::Polygon {
                points: vec![
                    Point::new(138, 10),
                    Point::new(133, 18),
                    Point::new(118, 13),
                    Point::new(133, 20),
                    Point::new(128, 28),
                    Point::new(138, 25),
                    Point::new(150, 29),
                    Point::new(150, 27),
                    Point::new(145, 23),
                    Point::new(150, 20),
                    Point::new(150, 17),
                    Point::new(148, 18),
                ],
            }, // 2
            image_ops::Polygon { points: vec![] }, // 3, it is ignored since its a triangle
            image_ops::Polygon {
                points: vec![
                    Point::new(125, 40),
                    Point::new(125, 60),
                    Point::new(148, 63),
                    Point::new(150, 63),
                    Point::new(150, 52),
                ],
            }, // 4
            image_ops::Polygon {
                points: vec![
                    Point::new(25, 91),
                    Point::new(13, 113),
                    Point::new(25, 134),
                    Point::new(50, 134),
                    Point::new(63, 113),
                    Point::new(50, 91),
                ],
            }, // 5
            image_ops::Polygon {
                points: vec![
                    Point::new(135, 100),
                    Point::new(120, 105),
                    Point::new(95, 125),
                    Point::new(110, 140),
                ],
            }, // 6
        ]);
        let expected_scores = vec![
            1.0,
            0.9980139026812314,
            0.0,
            0.9924146649810367,
            1.0,
            0.9978822532825075,
        ];
        assert_eq!(
            get_polygons_from_bitmap(&pred_tensor, &bitmap_tensor, &adjust_values)?,
            (expected_boxes, expected_scores)
        );
        Ok(())
    }

    #[test]
    fn evaluate_image_test_one_matching_polygon() -> Result<()> {
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            }, // poly 2
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiplePolygons(vec![image_ops::Polygon {
            points: vec![
                Point::new(1, 1),
                Point::new(10, 0),
                Point::new(10, 10),
                Point::new(0, 10),
            ],
        }]);
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
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ], // poly 2
            },
        ]);
        let ignore_polygons = [true, true];
        let pred = MultiplePolygons(vec![image_ops::Polygon {
            points: vec![
                Point::new(1, 1),
                Point::new(10, 0),
                Point::new(10, 10),
                Point::new(0, 10),
            ],
        }]);
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
        let gt_polygons_tensor = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(0, 0),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            }, // poly 1
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            }, // poly 2
        ]);
        let ignore_polygons = [false, false];
        let pred = MultiplePolygons(vec![
            image_ops::Polygon {
                points: vec![
                    Point::new(1, 1),
                    Point::new(10, 0),
                    Point::new(10, 10),
                    Point::new(0, 10),
                ],
            },
            image_ops::Polygon {
                points: vec![
                    Point::new(20, 20),
                    Point::new(30, 20),
                    Point::new(30, 30),
                    Point::new(20, 30),
                ],
            },
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

    // #[test]
    // fn validate_measure_test() -> Result<()> {
    //     let gt_polygons_tensor = BatchPolygons(vec![
    //         // image 1
    //         MultiplePolygons(vec![
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(0, 0),
    //                     Point::new(10, 0),
    //                     Point::new(10, 10),
    //                     Point::new(0, 10),
    //                 ],
    //             }, // poly 1
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(20, 20),
    //                     Point::new(30, 20),
    //                     Point::new(30, 30),
    //                     Point::new(20, 30),
    //                 ],
    //             }, // poly 2
    //         ]),
    //         // image 2
    //         MultiplePolygons(vec![
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(0, 0),
    //                     Point::new(10, 0),
    //                     Point::new(10, 10),
    //                     Point::new(0, 10),
    //                 ],
    //             }, // poly 1
    //             image_ops::Polygon {
    //                 points: vec![
    //                     Point::new(20, 20),
    //                     Point::new(30, 20),
    //                     Point::new(30, 30),
    //                     Point::new(20, 30),
    //                 ],
    //             }, // poly 2
    //         ]),
    //     ]);
    //     let ignore_polygons = [
    //         vec![false, false], // image 1
    //         vec![false, false], // image 2
    //     ];
    //     let pred = BatchPolygons(vec![
    //         MultiplePolygons(vec![image_ops::Polygon {
    //             points: vec![
    //                 Point::new(1, 1),
    //                 Point::new(10, 0),
    //                 Point::new(10, 10),
    //                 Point::new(0, 10),
    //             ],
    //         }]),
    //         MultiplePolygons(vec![image_ops::Polygon {
    //             points: vec![
    //                 Point::new(45, 61),
    //                 Point::new(47, 41),
    //                 Point::new(60, 60),
    //                 Point::new(39, 48),
    //             ],
    //         }]),
    //     ]);
    //     let scores = [vec![0.9], vec![0.9]];
    //     let metrics = validate_measure(&gt_polygons_tensor, &ignore_polygons, &pred, &scores)?;
    //     assert_eq!(metrics.len(), 2);

    //     // image 1 metrics (having 1 matched polygon)
    //     assert_eq!(metrics[0].gt_care, 2);
    //     assert_eq!(metrics[0].det_care, 1);
    //     assert_eq!(metrics[0].det_matched, 1);
    //     assert!(metrics[0].precision - 1. < ERROR_THRESHOLD);
    //     assert!(metrics[0].recall - 0.5 < ERROR_THRESHOLD);
    //     assert!(metrics[0].hmean - 0.6666666666666666 < ERROR_THRESHOLD);

    //     // image 2 metrics (having 0 matched polygons)
    //     assert_eq!(metrics[1].gt_care, 2);
    //     assert_eq!(metrics[1].det_care, 1);
    //     assert_eq!(metrics[1].det_matched, 0);
    //     assert!(metrics[1].precision < ERROR_THRESHOLD);
    //     assert!(metrics[1].recall < ERROR_THRESHOLD);
    //     assert!(metrics[1].hmean < ERROR_THRESHOLD);
    //     Ok(())
    // }

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
