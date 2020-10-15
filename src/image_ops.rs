use super::dataset::TextDetectionDataset;
use super::utils::{VALUES_COUNT, VALUES_MAP};
use anyhow::{anyhow, Result};
use image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::definitions::Point;
use imageproc::drawing::draw_polygon_mut;
use log::{error, trace};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tch::vision::dataset::Dataset;
use tch::{Kind, Tensor};

const CHAR_REC_IMAGES_PATH: &str = "images";
const CHAR_REC_TRAIN_IMAGES_FILE: &str = "training_images_data_char";
const CHAR_REC_TRAIN_LABELS_FILE: &str = "training_labels_data_char";
const CHAR_REC_TEST_IMAGES_FILE: &str = "test_images_data_char";
const CHAR_REC_TEST_LABELS_FILE: &str = "test_labels_data_char";
pub const TEXT_DET_IMAGES_PATH: &str = "text-detection-images/totaltext";
pub const TEXT_DET_TRAIN_IMAGES_FILE: &str = "training_images_data_text_det";
pub const TEXT_DET_TRAIN_GT_FILE: &str = "training_gt_data_text_det";
pub const TEXT_DET_TRAIN_MASK_FILE: &str = "training_mask_data_text_det";
pub const TEXT_DET_TRAIN_ADJ_FILE: &str = "training_adj_data_text_det";
pub const TEXT_DET_TRAIN_POLYS_FILE: &str = "training_polys_data_text_det";
pub const TEXT_DET_TRAIN_IGNORE_FLAGS_FILE: &str = "training_ignore_flags_data_text_det";
pub const TEXT_DET_TEST_IMAGES_FILE: &str = "test_images_data_text_det";
pub const TEXT_DET_TEST_GT_FILE: &str = "test_gt_data_text_det";
pub const TEXT_DET_TEST_MASK_FILE: &str = "test_mask_data_text_det";
pub const TEXT_DET_TEST_ADJ_FILE: &str = "test_adj_data_text_det";
pub const TEXT_DET_TEST_POLYS_FILE: &str = "test_polys_data_text_det";
pub const TEXT_DET_TEST_IGNORE_FLAGS_FILE: &str = "test_ignore_flags_data_text_det";
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 800;
const WHITE_COLOR: Luma<u8> = Luma([255]);
const BLACK_COLOR: Luma<u8> = Luma([0]);
const MIN_TEXT_SIZE: u32 = 8;

lazy_static! {
    static ref CHAR_REC_FILE_NAME_FORMAT_REGEX: Regex =
        Regex::new(r"(.*)-(upper|lower|num)-([a-zA-z0-9])-img\.png").unwrap();
    static ref TEXT_DET_FILE_NAME_FORMAT_REGEX: Regex = Regex::new(r"img[0-9]+\.jpg").unwrap();
}

pub fn load_values() -> Result<Dataset> {
    trace!("loading character recognition values");
    let instant = Instant::now();
    let train_images = load_values_from_file(CHAR_REC_TRAIN_IMAGES_FILE)?;
    let train_labels = load_values_from_file(CHAR_REC_TRAIN_LABELS_FILE)?;
    let test_images = load_values_from_file(CHAR_REC_TEST_IMAGES_FILE)?;
    let test_labels = load_values_from_file(CHAR_REC_TEST_LABELS_FILE)?;
    trace!(
        "Finished loading values in {} ms",
        instant.elapsed().as_millis()
    );

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: VALUES_COUNT as i64,
    })
}

pub fn load_image_as_tensor(file_path: &str) -> Result<Tensor> {
    if !Path::new(file_path).exists() {
        return Err(anyhow!("File {} doesn't exist", file_path));
    }
    let image = open(file_path)?.into_luma();
    let dim = image.width() * image.height();
    let images_tensor = convert_image_to_tensor(&image)?.view((1, dim as i64)) / 255.;
    Ok(images_tensor)
}

fn load_values_from_file(file_path: &str) -> Result<Tensor> {
    let data_tensor;
    if Path::new(file_path).exists() {
        data_tensor = Tensor::load(file_path)?;
    } else {
        let images_path;
        let labels_path;
        let images_dir;
        let is_images =
            file_path == CHAR_REC_TEST_IMAGES_FILE || file_path == CHAR_REC_TRAIN_IMAGES_FILE;
        if file_path == CHAR_REC_TEST_IMAGES_FILE || file_path == CHAR_REC_TEST_LABELS_FILE {
            images_path = CHAR_REC_TEST_IMAGES_FILE;
            labels_path = CHAR_REC_TEST_LABELS_FILE;
            images_dir = format!("{}/test", CHAR_REC_IMAGES_PATH);
        } else {
            images_path = CHAR_REC_TRAIN_IMAGES_FILE;
            labels_path = CHAR_REC_TRAIN_LABELS_FILE;
            images_dir = format!("{}/training", CHAR_REC_IMAGES_PATH);
        }
        if is_images {
            let images_data = load_images(&images_dir)?;
            images_data.save(images_path)?;
            data_tensor = images_data;
        } else {
            let labels_data = load_labels(&images_dir)?;
            labels_data.save(labels_path)?;
            data_tensor = labels_data;
        }
    }

    Ok(data_tensor)
}

fn load_images(dir_path: &str) -> Result<Tensor> {
    if let Ok(files) = fs::read_dir(dir_path) {
        let files_info: Vec<std::fs::DirEntry> = files.map(|f| f.unwrap()).collect();
        let (images_tensor, rows, cols) = files_info
            .par_iter()
            .fold(
                || (Tensor::new(), 0, 0),
                |(mut p_vec, mut rows, mut cols), file| {
                    let filename = file.file_name();
                    if CHAR_REC_FILE_NAME_FORMAT_REGEX
                        .captures(filename.to_str().unwrap())
                        .is_some()
                    {
                        rows += 1;
                        let image = open(&file.path().display().to_string())
                            .unwrap()
                            .into_luma();
                        let image_tensor = convert_image_to_tensor(&image)
                            .unwrap()
                            .view((1, (image.width() * image.height()) as i64));
                        cols = image_tensor.size()[1];
                        p_vec = Tensor::cat(&[p_vec, image_tensor], 0);
                    }
                    (p_vec, rows, cols)
                },
            )
            .reduce(
                || (Tensor::new(), 0, 0),
                |(acc_vec, total_rows, _), (partial, p_rows, p_cols)| {
                    (
                        Tensor::cat(&[acc_vec, partial], 0),
                        total_rows + p_rows,
                        p_cols,
                    )
                },
            );
        trace!("Creating tensor for images r-{} c-{}", rows, cols);
        return Ok(images_tensor.view((rows as i64, cols as i64)) / 255.);
    }
    Err(anyhow!("Could not open dir {}", dir_path))
}

fn load_labels(dir_path: &str) -> Result<Tensor> {
    if let Ok(files) = fs::read_dir(dir_path) {
        let filenames: Vec<String> = files
            .map(|f| f.unwrap().file_name().into_string().unwrap())
            .collect();
        let (labels, rows, _cols) = filenames
            .par_iter()
            .fold(
                || (Vec::new(), 0, 0),
                |(mut p_vec, mut rows, _), filename| {
                    if let Some(caps) = CHAR_REC_FILE_NAME_FORMAT_REGEX.captures(filename) {
                        rows += 1;
                        let _font = String::from(&caps[1]);
                        let _letter_type = String::from(&caps[2]);
                        if let Some(letter) = caps[3].chars().next() {
                            p_vec.push(*VALUES_MAP.get(&letter).unwrap());
                        }
                    }
                    (p_vec, rows, 1)
                },
            )
            .reduce(
                || (Vec::new(), 0, 0),
                |(mut acc_vec, total_rows, _), (mut partial, p_rows, p_cols)| {
                    acc_vec.append(&mut partial);
                    (acc_vec, total_rows + p_rows, p_cols)
                },
            );
        trace!("Creating tensor for labels r-{} c-1", rows);
        let labels_tensor = Tensor::of_slice(&labels).to_kind(Kind::Int64);
        return Ok(labels_tensor);
    }
    Err(anyhow!("Could not open dir {}", dir_path))
}

pub fn preprocess_image(file_path: &str, target_dim: (u32, u32)) -> Result<(GrayImage, f64, f64)> {
    let (width, height) = target_dim;
    let instant = Instant::now();
    let rgba_image = open(file_path)?.into_rgba();
    let original_width = rgba_image.width();
    let original_height = rgba_image.height();
    let dyn_image = DynamicImage::ImageRgba8(rgba_image)
        .resize(width, height, FilterType::Triangle)
        .to_luma();

    // correction for pixel coords
    let adjust_x = dyn_image.width() as f64 / original_width as f64;
    let adjust_y = dyn_image.height() as f64 / original_height as f64;

    let mut pixel_values = vec![0; (width * height) as usize];
    pixel_values
        .par_iter_mut()
        .enumerate()
        .for_each(|(n, val)| {
            let y = n as u32 / width;
            let x = n as u32 - y * width;
            if x < dyn_image.width() && y < dyn_image.height() {
                *val = dyn_image.get_pixel(x, y).0[0];
            }
        });
    trace!(
        "finished preprocessing in {} ns",
        instant.elapsed().as_nanos()
    );
    Ok((
        ImageBuffer::from_vec(width, height, pixel_values).unwrap(),
        adjust_x,
        adjust_y,
    ))
}

fn generate_gt_and_mask_images(
    polygons: &MultiplePolygons,
    adjust_x: f64,
    adjust_y: f64,
    target_dim: (u32, u32),
) -> Result<(GrayImage, GrayImage, Vec<bool>)> {
    let (width, height) = target_dim;
    let mut gt_image = GrayImage::new(width, height);
    let mut mask_temp = DynamicImage::new_luma8(width, height);
    let mut ignore_flags = vec![false; polygons.0.len()];
    mask_temp.invert();
    let mut mask_image = mask_temp.to_luma();
    for (pos, poly) in polygons.0.iter().enumerate() {
        let num_of_points = poly.points.len();
        let (min_x, max_x, min_y, max_y) =
            poly.points
                .iter()
                .fold((std::u32::MAX, 0, std::u32::MAX, 0), |acc, p| {
                    (
                        acc.0.min(p.x),
                        acc.1.max(p.x),
                        acc.2.min(p.y),
                        acc.3.max(p.y),
                    )
                });
        let poly_width = max_x - min_x;
        let poly_height = max_y - min_y;
        let poly_values = (0..num_of_points)
            .map(|i| {
                Point::new(
                    (poly.points[i].x as f64 * adjust_x) as i32,
                    (poly.points[i].y as f64 * adjust_y) as i32,
                )
            })
            .collect::<Vec<Point<i32>>>();
        if poly_values.len() < 4 {
            ignore_flags[pos] = true;
            continue;
        }
        if poly_height.min(poly_width) < MIN_TEXT_SIZE {
            draw_polygon_mut(&mut mask_image, &poly_values, BLACK_COLOR);
            ignore_flags[pos] = true;
        } else {
            draw_polygon_mut(&mut gt_image, &poly_values, WHITE_COLOR);
        }
    }
    Ok((gt_image, mask_image, ignore_flags))
}

fn load_polygons(file_path: &str) -> Result<MultiplePolygons> {
    let polygons;
    if let Ok(mut file) = File::open(file_path) {
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        polygons = content
            .split_terminator('\n')
            .collect::<Vec<&str>>()
            .par_iter()
            .map(|row| {
                // row is in format
                // x1,y1,x2,y2,...,x{n},y{n},{FOUND TEXT}
                // where n is number of points in the polygon (not a fixed value)

                let raw_values = row.split_terminator(',').collect::<Vec<&str>>();

                let values = raw_values[0..raw_values.len() - 1]
                    .iter()
                    .flat_map(|v| v.parse())
                    .collect::<Vec<u32>>();
                Polygon {
                    points: values
                        .chunks_exact(2)
                        .map(|point| Point::new(point[0], point[1]))
                        .collect(),
                }
            })
            .collect();
    } else {
        return Err(anyhow!("didn't find file {}", file_path));
    }

    Ok(MultiplePolygons(polygons))
}

pub fn load_text_detection_image(
    file_path: &str,
    target_dim: (u32, u32),
) -> Result<(Tensor, Tensor, Tensor, Tensor, MultiplePolygons, Vec<bool>)> {
    let instant = Instant::now();
    let (preprocessed_image, adjust_x, adjust_y) = preprocess_image(file_path, target_dim)?;

    let path_parts = file_path.split_terminator('/').collect::<Vec<&str>>();
    let last_idx = path_parts.len() - 1;
    let polygons = load_polygons(&format!(
        "{}/gts/{}/{}.txt",
        TEXT_DET_IMAGES_PATH,
        path_parts[last_idx - 1],
        path_parts[last_idx]
    ))?;

    let (gt_image, mask_image, ignore_flags) =
        generate_gt_and_mask_images(&polygons, adjust_x, adjust_y, target_dim)?;

    let image_tensor = convert_image_to_tensor(&preprocessed_image)?.to_kind(Kind::Uint8);
    let gt_tensor = (convert_image_to_tensor(&gt_image)? / 255.).to_kind(Kind::Uint8);
    let mask_tensor = (convert_image_to_tensor(&mask_image)? / 255.).to_kind(Kind::Uint8);
    let adjust_tensor = Tensor::of_slice(&[adjust_x, adjust_y]).view((1, 2));

    trace!(
        "finished loading and preparing text detection images in {:?} ns",
        instant.elapsed().as_nanos()
    );

    Ok((
        image_tensor,
        gt_tensor,
        mask_tensor,
        adjust_tensor,
        polygons,
        ignore_flags,
    ))
}

/// image size (w, h)
pub fn convert_image_to_tensor(image: &GrayImage) -> Result<Tensor> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let mut pixel_values = vec![0f64; w * h];
    pixel_values
        .par_iter_mut()
        .enumerate()
        .for_each(|(n, val)| {
            let y = n as u32 / w as u32;
            let x = n as u32 - y * w as u32;

            *val = image.get_pixel(x, y).0[0] as f64;
        });
    Ok(Tensor::of_slice(&pixel_values).view((h as i64, w as i64)))
}

/// tensor size (H x W)
pub fn convert_tensor_to_image(tensor: &Tensor) -> Result<GrayImage> {
    let size = tensor.size();
    if size.len() > 2 {
        return Err(anyhow!("tensor must be in 2 dimensions"));
    }
    let h = size[size.len() - 2] as usize;
    let w = size[size.len() - 1] as usize;
    let mut pixel_values = vec![0; w * h];
    pixel_values.iter_mut().enumerate().for_each(|(n, val)| {
        let y = n as i64 / w as i64;
        let x = n as i64 - y * w as i64;
        *val = tensor.int64_value(&[y, x]) as u8;
    });
    Ok(ImageBuffer::from_vec(w as u32, h as u32, pixel_values).unwrap())
}

pub fn load_text_detection_tensor_files(target_dir: &str) -> Result<TextDetectionDataset> {
    if let Ok(files) = fs::read_dir(&target_dir) {
        let mut train_images = BTreeSet::new();
        let mut train_gt = BTreeSet::new();
        let mut train_mask = BTreeSet::new();
        let mut train_adj = BTreeSet::new();
        let mut train_polys = BTreeSet::new();
        let mut train_ignore_flags = BTreeSet::new();
        let mut test_images = BTreeSet::new();
        let mut test_gt = BTreeSet::new();
        let mut test_mask = BTreeSet::new();
        let mut test_adj = BTreeSet::new();
        let mut test_polys = BTreeSet::new();
        let mut test_ignore_flags = BTreeSet::new();
        files.for_each(|f| {
            let file = f.unwrap();
            let filename = file.file_name().into_string().unwrap();
            let path = file.path().display().to_string();
            if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_IMAGES_FILE)) {
                train_images.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_GT_FILE)) {
                train_gt.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_MASK_FILE)) {
                train_mask.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_ADJ_FILE)) {
                train_adj.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_POLYS_FILE)) {
                train_polys.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_IGNORE_FLAGS_FILE)) {
                train_ignore_flags.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_IMAGES_FILE)) {
                test_images.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_GT_FILE)) {
                test_gt.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_MASK_FILE)) {
                test_mask.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_ADJ_FILE)) {
                test_adj.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_POLYS_FILE)) {
                test_polys.insert(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_IGNORE_FLAGS_FILE)) {
                test_ignore_flags.insert(path);
            }
        });
        if train_images.len() != train_gt.len()
            || train_images.len() != train_mask.len()
            || train_images.len() != train_adj.len()
        {
            return Err(
                anyhow!(
                    "training tensors number doesn't match: images tensors {} - gt tensors {} - mask tensors {}",
                    train_images.len(),
                    train_gt.len(),
                    train_mask.len()
                )
            );
        } else if test_images.len() != test_gt.len()
            || test_images.len() != test_mask.len()
            || test_images.len() != test_adj.len()
        {
            return Err(
                anyhow!(
                    "test tensors number doesn't match: images tensors {} - gt tensors {} - mask tensors {}",
                    test_images.len(),
                    test_gt.len(),
                    test_mask.len()
                )
            );
        }
        Ok(TextDetectionDataset {
            train_images: train_images.into_iter().collect(),
            train_gt: train_gt.into_iter().collect(),
            train_mask: train_mask.into_iter().collect(),
            train_adj: train_adj.into_iter().collect(),
            train_polys: train_polys.into_iter().collect(),
            train_ignore_flags: train_ignore_flags.into_iter().collect(),
            test_images: test_images.into_iter().collect(),
            test_gt: test_gt.into_iter().collect(),
            test_mask: test_mask.into_iter().collect(),
            test_adj: test_adj.into_iter().collect(),
            test_polys: test_polys.into_iter().collect(),
            test_ignore_flags: test_ignore_flags.into_iter().collect(),
        })
    } else {
        Err(anyhow!("The directory doesn't exist"))
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq, Eq)]
pub struct PointDef<T: Copy + PartialEq + Eq> {
    pub x: T,
    pub y: T,
}
impl<T: Copy + PartialEq + Eq> From<Point<T>> for PointDef<T> {
    fn from(def: Point<T>) -> Self {
        Self { x: def.x, y: def.y }
    }
}

fn points_vec_ser<S: Serializer>(vec: &[Point<u32>], serializer: S) -> Result<S::Ok, S::Error> {
    let vec2: Vec<PointDef<u32>> = vec.iter().map(|x| PointDef::from(*x)).collect();

    vec2.serialize(serializer)
}
fn points_vec_deser<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<Point<u32>>, D::Error> {
    let vec: Vec<PointDef<u32>> = Deserialize::deserialize(deserializer)?;

    Ok(vec.iter().map(|p| Point::<u32>::new(p.x, p.y)).collect())
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Polygon {
    #[serde(serialize_with = "points_vec_ser")]
    #[serde(deserialize_with = "points_vec_deser")]
    pub points: Vec<Point<u32>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct MultiplePolygons(pub Vec<Polygon>);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct BatchPolygons(pub Vec<MultiplePolygons>);

pub fn generate_text_det_tensor_chunks(
    images_base_dir: &str,
    train: bool,
    target_dim: Option<(u32, u32)>,
) -> Result<()> {
    let instant = Instant::now();
    let dim = target_dim.unwrap_or((DEFAULT_WIDTH, DEFAULT_HEIGHT));
    let images_file;
    let gt_file;
    let mask_file;
    let adj_file;
    let polys_file;
    let ignore_flags_file;
    let window_size;
    let target_dir;
    if train {
        images_file = get_target_filename(TEXT_DET_TRAIN_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TRAIN_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TRAIN_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TRAIN_ADJ_FILE);
        polys_file = get_target_filename(TEXT_DET_TRAIN_POLYS_FILE);
        ignore_flags_file = get_target_filename(TEXT_DET_TRAIN_IGNORE_FLAGS_FILE);
        window_size = 40;
        target_dir = "train";
    } else {
        images_file = get_target_filename(TEXT_DET_TEST_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TEST_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TEST_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TEST_ADJ_FILE);
        polys_file = get_target_filename(TEXT_DET_TEST_POLYS_FILE);
        ignore_flags_file = get_target_filename(TEXT_DET_TEST_IGNORE_FLAGS_FILE);
        window_size = 10;
        target_dir = "test";
    };

    let images_dir = format!("{}/images/{}", images_base_dir, target_dir);
    if let Ok(files) = fs::read_dir(&images_dir) {
        let filenames: Vec<String> = files
            .map(|f| f.unwrap().path().display().to_string())
            .collect();

        filenames
            .chunks(window_size)
            .enumerate()
            .for_each(|(pos, chunk)| {
                let (images, gts, masks, adjust_values, batch_polygons, batch_ignore_flags) = chunk
                    .par_iter()
                    .fold(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Vec::new(),
                                Vec::new(),
                            )
                        },
                        |(
                            mut im_acc,
                            mut gt_acc,
                            mut mask_acc,
                            mut adjust_acc,
                            mut polygons_acc,
                            mut ignore_flags_acc,
                        ),
                         filename| {
                            if TEXT_DET_FILE_NAME_FORMAT_REGEX.captures(filename).is_some() {
                                match load_text_detection_image(filename, dim) {
                                    Ok((im, gt, mask, adj_values, polygons, ignore_flags)) => {
                                        polygons_acc.push(polygons);
                                        ignore_flags_acc.push(ignore_flags);
                                        if im_acc.numel() == 1 {
                                            return (
                                                im,
                                                gt,
                                                mask,
                                                adj_values,
                                                polygons_acc,
                                                ignore_flags_acc,
                                            );
                                        }
                                        im_acc = Tensor::cat(&[im_acc, im], 0);
                                        gt_acc = Tensor::cat(&[gt_acc, gt], 0);
                                        mask_acc = Tensor::cat(&[mask_acc, mask], 0);
                                        adjust_acc = Tensor::cat(&[adjust_acc, adj_values], 0);
                                    }
                                    Err(msg) => {
                                        error!("Error while loading single image data: {}", msg);
                                    }
                                }
                            }
                            (
                                im_acc,
                                gt_acc,
                                mask_acc,
                                adjust_acc,
                                polygons_acc,
                                ignore_flags_acc,
                            )
                        },
                    )
                    .reduce(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Vec::new(),
                                Vec::new(),
                            )
                        },
                        |(
                            im_acc,
                            gt_acc,
                            mask_acc,
                            adj_acc,
                            mut polys_acc,
                            mut ignore_flags_acc,
                        ),
                         (
                            part_im,
                            part_gt,
                            part_mask,
                            part_adj,
                            mut part_polys,
                            mut part_ignore_flags,
                        )| {
                            polys_acc.append(&mut part_polys);
                            ignore_flags_acc.append(&mut part_ignore_flags);
                            if im_acc.numel() == 1 {
                                return (
                                    part_im,
                                    part_gt,
                                    part_mask,
                                    part_adj,
                                    polys_acc,
                                    ignore_flags_acc,
                                );
                            } else if part_im.numel() == 1 {
                                return (
                                    im_acc,
                                    gt_acc,
                                    mask_acc,
                                    adj_acc,
                                    polys_acc,
                                    ignore_flags_acc,
                                );
                            }
                            (
                                Tensor::cat(&[im_acc, part_im], 0),
                                Tensor::cat(&[gt_acc, part_gt], 0),
                                Tensor::cat(&[mask_acc, part_mask], 0),
                                Tensor::cat(&[adj_acc, part_adj], 0),
                                polys_acc,
                                ignore_flags_acc,
                            )
                        },
                    );
                let mut save_res = images
                    .view((-1, dim.0 as i64, dim.1 as i64))
                    .save(format!("{}.{}", images_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving image tensor {}", msg);
                }
                save_res = gts
                    .view((-1, dim.0 as i64, dim.1 as i64))
                    .save(format!("{}.{}", gt_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving gt tensor {}", msg);
                }
                save_res = masks
                    .view((-1, dim.0 as i64, dim.1 as i64))
                    .save(format!("{}.{}", mask_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving mask tensor {}", msg);
                }
                save_res = adjust_values
                    .view((-1, 2))
                    .save(format!("{}.{}", adj_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving adj tensor {}", msg);
                }
                let save_res = save_batch_polygons(
                    &BatchPolygons(batch_polygons),
                    &format!("{}.{}", polys_file, pos),
                    true,
                );
                if let Err(msg) = save_res {
                    error!("Error while saving polygons vec {}", msg);
                }
                let save_res = save_vec_to_file(
                    &batch_ignore_flags,
                    &format!("{}.{}", ignore_flags_file, pos),
                    true,
                );
                if let Err(msg) = save_res {
                    error!("Error while saving ignore flags vec {}", msg);
                }
            });
        trace!(
            "finished generating tensors in {} ms",
            instant.elapsed().as_millis()
        );
        Ok(())
    } else {
        Err(anyhow!(
            "didn't find text detection images dir: {}",
            images_dir
        ))
    }
}

#[cfg(test)]
fn get_target_filename(name: &str) -> String {
    let mut s = String::from("test-");
    s.push_str(name);
    s
}

#[cfg(not(test))]
fn get_target_filename(name: &str) -> String {
    String::from(name)
}

fn save_vec_to_file<T: Serialize>(value: &[T], file_path: &str, overwrite: bool) -> Result<()> {
    if Path::new(file_path).exists() {
        if overwrite {
            fs::remove_file(file_path)?;
        } else {
            return Err(anyhow!("file {} already exists", file_path));
        }
    }
    serde_json::to_writer(File::create(file_path)?, value)?;

    Ok(())
}

fn save_batch_polygons(polygons: &BatchPolygons, file_path: &str, overwrite: bool) -> Result<()> {
    if Path::new(file_path).exists() {
        if overwrite {
            fs::remove_file(file_path)?;
        } else {
            return Err(anyhow!("file {} already exists", file_path));
        }
    }
    serde_json::to_writer(File::create(file_path)?, polygons)?;

    Ok(())
}

pub fn load_polygons_vec_from_file(file_path: &str) -> Result<BatchPolygons> {
    if !Path::new(file_path).exists() {
        return Err(anyhow!("file {} doesn't exists", file_path));
    }
    let reader = BufReader::new(File::open(file_path)?);
    let value = serde_json::from_reader(reader)?;
    Ok(value)
}

pub fn load_ignore_flags_vec_from_file(file_path: &str) -> Result<Vec<Vec<bool>>> {
    if !Path::new(file_path).exists() {
        return Err(anyhow!("file {} doesn't exists", file_path));
    }
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let value = serde_json::from_reader(reader)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_ops;
    use image::open;
    use tch::{Device, Tensor};

    #[test]
    fn tensor_cat() {
        let device = Device::cuda_if_available();
        let zeros = Tensor::zeros(&[2, 3], (Kind::Double, device));
        let ones = Tensor::ones(&[2, 3], (Kind::Double, device));
        assert_eq!(
            Tensor::cat(&[zeros, ones], 0).view((-1, 2, 3)),
            Tensor::of_slice(&[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).view((2, 2, 3))
        );
    }

    #[test]
    fn conversion_5x5_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 0, 1, //
            0, 0, 0, 1, 1, //
            0, 0, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            1, 1, 1, 1, 1, //
        ];
        let original_tensor = Tensor::of_slice(&values).view([5, 5]);
        let original_image = GrayImage::from_vec(5, 5, values).unwrap();
        let converted_image = convert_tensor_to_image(&original_tensor)?;
        assert_eq!(original_image, converted_image);

        let converted_tensor = convert_image_to_tensor(&original_image)?;
        assert_eq!(original_tensor, converted_tensor);
        Ok(())
    }

    #[test]
    fn conversion_of_different_dim_test() -> Result<()> {
        let values = vec![
            0, 0, 0, 0, 1, //
            0, 0, 0, 1, 1, //
            0, 0, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            1, 1, 1, 1, 1, //
            0, 1, 1, 1, 1, //
            0, 0, 1, 1, 1, //
            0, 0, 0, 1, 1, //
            0, 0, 0, 0, 1, //
        ];
        let original_tensor = Tensor::of_slice(&values).view([9, 5]);
        let original_image = GrayImage::from_vec(5, 9, values).unwrap();
        let converted_image = convert_tensor_to_image(&original_tensor)?;
        assert_eq!(original_image, converted_image);

        let converted_tensor = convert_image_to_tensor(&original_image)?;
        assert_eq!(original_tensor, converted_tensor);
        Ok(())
    }

    #[test]
    fn tensor_generating_tests() -> Result<()> {
        // generate tensors
        generate_text_det_tensor_chunks("test_data/text_det", true, None)?;
        generate_text_det_tensor_chunks("test_data/text_det", false, None)?;

        // load expected images
        let train_img1 = open("test_data/preprocessed_img55.png")?.to_luma();
        let train_img2 = open("test_data/preprocessed_img224.png")?.to_luma();
        let train_gt_img1 = open("test_data/gt_img55.png")?.to_luma();
        let train_gt_img2 = open("test_data/gt_img224.png")?.to_luma();
        let train_mask_img1 = open("test_data/mask_img55.png")?.to_luma();
        let train_mask_img2 = open("test_data/mask_img224.png")?.to_luma();
        let test_img1 = open("test_data/preprocessed_img494.png")?.to_luma();
        let test_img2 = open("test_data/preprocessed_img545.png")?.to_luma();
        let test_gt_img1 = open("test_data/gt_img494.png")?.to_luma();
        let test_gt_img2 = open("test_data/gt_img545.png")?.to_luma();
        let test_mask_img1 = open("test_data/mask_img494.png")?.to_luma();
        let test_mask_img2 = open("test_data/mask_img545.png")?.to_luma();

        // assert generated values
        let filenames: Vec<String> = [
            TEXT_DET_TRAIN_IMAGES_FILE,
            TEXT_DET_TRAIN_GT_FILE,
            TEXT_DET_TRAIN_MASK_FILE,
            TEXT_DET_TRAIN_ADJ_FILE,
            TEXT_DET_TRAIN_POLYS_FILE,
            TEXT_DET_TRAIN_IGNORE_FLAGS_FILE,
            TEXT_DET_TEST_IMAGES_FILE,
            TEXT_DET_TEST_GT_FILE,
            TEXT_DET_TEST_MASK_FILE,
            TEXT_DET_TEST_ADJ_FILE,
            TEXT_DET_TEST_POLYS_FILE,
            TEXT_DET_TEST_IGNORE_FLAGS_FILE,
        ]
        .iter()
        .map(|&x| {
            let mut s = get_target_filename(x);
            s.push_str(".0");
            s
        })
        .collect();

        let train_images_tensor = Tensor::load(&filenames[0])?;
        assert_eq!(train_images_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_images_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&train_img1)?
        );
        assert_eq!(
            train_images_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&train_img2)?
        );

        let train_gts_tensor = Tensor::load(&filenames[1])?;
        assert_eq!(train_gts_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_gts_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&train_gt_img1)? / 255.
        );
        assert_eq!(
            train_gts_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&train_gt_img2)? / 255.
        );

        let train_masks_tensor = Tensor::load(&filenames[2])?;
        assert_eq!(train_masks_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_masks_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&train_mask_img1)? / 255.
        );
        assert_eq!(
            train_masks_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&train_mask_img2)? / 255.
        );

        let train_adj_tensor = Tensor::load(&filenames[3])?;
        assert_eq!(train_adj_tensor.size(), [2, 2]);

        let mut original_dim = (300., 200.);
        let mut resized = (800., 533.);
        assert_eq!(
            train_adj_tensor.get(0),
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
        );

        original_dim = (180., 240.);
        resized = (600., 800.);
        assert_eq!(
            train_adj_tensor.get(1),
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
        );

        let train_polygons_vec = image_ops::load_polygons_vec_from_file(&filenames[4])?;
        assert_eq!(train_polygons_vec.0.len(), 2);
        assert_eq!(train_polygons_vec.0[0].0.len(), 4);
        assert_eq!(train_polygons_vec.0[1].0.len(), 3);

        let train_ignore_flags_vec = image_ops::load_ignore_flags_vec_from_file(&filenames[5])?;
        assert_eq!(train_ignore_flags_vec.len(), 2);
        assert_eq!(train_ignore_flags_vec[0].len(), 4);
        assert_eq!(train_ignore_flags_vec[1].len(), 3);

        let test_images_tensor = Tensor::load(&filenames[6])?;
        assert_eq!(test_images_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_images_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_img1)?
        );
        assert_eq!(
            test_images_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_img2)?
        );

        let test_gts_tensor = Tensor::load(&filenames[7])?;
        assert_eq!(test_gts_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_gts_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_gt_img1)? / 255.
        );
        assert_eq!(
            test_gts_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_gt_img2)? / 255.
        );

        let test_masks_tensor = Tensor::load(&filenames[8])?;
        assert_eq!(test_masks_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_masks_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_mask_img1)? / 255.
        );
        assert_eq!(
            test_masks_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_mask_img2)? / 255.
        );

        let test_adj_tensor = Tensor::load(&filenames[9])?;
        assert_eq!(test_adj_tensor.size(), [2, 2]);

        let test_polygons_vec = image_ops::load_polygons_vec_from_file(&filenames[10])?;
        assert_eq!(test_polygons_vec.0.len(), 2);
        assert_eq!(test_polygons_vec.0[0].0.len(), 3);
        assert_eq!(test_polygons_vec.0[1].0.len(), 3);

        let test_ignore_flags_vec = image_ops::load_ignore_flags_vec_from_file(&filenames[11])?;
        assert_eq!(test_ignore_flags_vec.len(), 2);
        assert_eq!(test_ignore_flags_vec[0].len(), 3);
        assert_eq!(test_ignore_flags_vec[1].len(), 3);

        original_dim = (200., 200.);
        resized = (800., 800.);
        assert_eq!(
            test_adj_tensor.get(0),
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
        );

        original_dim = (184., 274.);
        resized = (537., 800.);
        assert_eq!(
            test_adj_tensor.get(1),
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
        );

        // cleanup generated files
        for f in filenames {
            fs::remove_file(&f)?;
        }

        Ok(())
    }

    #[test]
    fn save_vec_to_file_and_load_from_file_test() -> Result<()> {
        let polys_file_path = "test_polys_vec_file";
        let ign_flags_file_path = "test_ign_flags_vec_file";

        let polys_test_vec = BatchPolygons(vec![
            MultiplePolygons(vec![
                // image 1
                image_ops::Polygon {
                    points: vec![
                        Point::new(12, 21),
                        Point::new(22, 21),
                        Point::new(22, 11),
                        Point::new(12, 11),
                    ],
                }, // poly 1
                image_ops::Polygon {
                    points: vec![
                        Point::new(120, 210),
                        Point::new(220, 210),
                        Point::new(220, 110),
                        Point::new(120, 110),
                    ],
                }, // poly 2
            ]),
            MultiplePolygons(vec![
                // image 2
                image_ops::Polygon {
                    points: vec![
                        Point::new(34, 43),
                        Point::new(44, 43),
                        Point::new(44, 33),
                        Point::new(34, 33),
                    ],
                }, // poly 3
                image_ops::Polygon {
                    points: vec![
                        Point::new(34, 43),
                        Point::new(44, 43),
                        Point::new(44, 33),
                        Point::new(34, 33),
                    ],
                }, // poly 4
            ]),
        ]);
        assert_eq!(Path::new(polys_file_path).exists(), false);
        save_batch_polygons(&polys_test_vec, polys_file_path, true)?;
        assert_eq!(Path::new(polys_file_path).exists(), true);

        let loaded_vec = load_polygons_vec_from_file(polys_file_path)?;
        assert_eq!(loaded_vec, polys_test_vec);

        fs::remove_file(polys_file_path)?;

        let ignore_flags_test_vec = vec![vec![true, false], vec![false, true]];
        assert_eq!(Path::new(ign_flags_file_path).exists(), false);
        save_vec_to_file(&ignore_flags_test_vec, ign_flags_file_path, true)?;
        assert_eq!(Path::new(ign_flags_file_path).exists(), true);

        let loaded_vec = load_ignore_flags_vec_from_file(ign_flags_file_path)?;
        assert_eq!(loaded_vec, ignore_flags_test_vec);

        fs::remove_file(ign_flags_file_path)?;

        Ok(())
    }
}
