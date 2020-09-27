use super::dataset::TextDetectionDataset;
use super::utils::{VALUES_COUNT, VALUES_MAP};
use anyhow::{anyhow, Result};
use image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::drawing::{self, Point};
use log::{debug, error, trace};
use rayon::prelude::*;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;
use tch::vision::dataset::Dataset;
use tch::{IndexOp, Kind, Tensor};

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
pub const TEXT_DET_TEST_IMAGES_FILE: &str = "test_images_data_text_det";
pub const TEXT_DET_TEST_GT_FILE: &str = "test_gt_data_text_det";
pub const TEXT_DET_TEST_MASK_FILE: &str = "test_mask_data_text_det";
pub const TEXT_DET_TEST_ADJ_FILE: &str = "test_adj_data_text_det";
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 800;
const WHITE_COLOR: Luma<u8> = Luma([255]);
const BLACK_COLOR: Luma<u8> = Luma([0]);
const MIN_TEXT_SIZE: i32 = 8;

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
    polygons: &[Tensor],
    adjust_x: f64,
    adjust_y: f64,
    target_dim: (u32, u32),
) -> Result<(GrayImage, GrayImage, Vec<bool>)> {
    let (width, height) = target_dim;
    let mut gt_image = GrayImage::new(width, height);
    let mut mask_temp = DynamicImage::new_luma8(width, height);
    let mut ignore_flags = vec![false; polygons.len()];
    mask_temp.invert();
    let mut mask_image = mask_temp.to_luma();
    for (pos, poly) in polygons.iter().enumerate() {
        let (num_of_points, _) = poly.size2()?;
        let poly_width = (poly.i((.., 0)).max() - poly.i((.., 0)).min()).int64_value(&[]) as i32;
        let poly_height = (poly.i((.., 1)).max() - poly.i((.., 1)).min()).int64_value(&[]) as i32;
        let poly_values = (0..num_of_points)
            .map(|i| {
                Point::new(
                    (poly.double_value(&[i, 0]) * adjust_x) as i32,
                    (poly.double_value(&[i, 1]) * adjust_y) as i32,
                )
            })
            .collect::<Vec<Point<i32>>>();
        if poly_values.len() < 4 {
            ignore_flags[pos] = true;
            continue;
        }
        if poly_height.min(poly_width) < MIN_TEXT_SIZE {
            drawing::draw_polygon_mut(&mut mask_image, &poly_values, BLACK_COLOR);
            ignore_flags[pos] = true;
        } else {
            drawing::draw_polygon_mut(&mut gt_image, &poly_values, WHITE_COLOR);
        }
    }
    Ok((gt_image, mask_image, ignore_flags))
}

fn load_polygons(file_path: &str) -> Result<Vec<Tensor>> {
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
                    .collect::<Vec<i32>>();
                Tensor::of_slice(&values).view((-1, 2))
            })
            .collect();
    } else {
        return Err(anyhow!("didn't find file {}", file_path));
    }

    Ok(polygons)
}

pub fn load_text_detection_image(
    file_path: &str,
    target_dim: (u32, u32),
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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

    let (gt_image, mask_image, _ignore_flags) =
        generate_gt_and_mask_images(&polygons, adjust_x, adjust_y, target_dim)?;

    let image_tensor = convert_image_to_tensor(&preprocessed_image)?.to_kind(Kind::Uint8);
    let gt_tensor = (convert_image_to_tensor(&gt_image)? / 255.).to_kind(Kind::Uint8);
    let mask_tensor = (convert_image_to_tensor(&mask_image)? / 255.).to_kind(Kind::Uint8);
    let adjust_tensor = Tensor::of_slice(&[adjust_x, adjust_y]).view((1, 2));

    trace!(
        "finished loading and preparing text detection images in {:?} ns",
        instant.elapsed().as_nanos()
    );

    Ok((image_tensor, gt_tensor, mask_tensor, adjust_tensor))
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
        let mut train_images = Vec::new();
        let mut train_gt = Vec::new();
        let mut train_mask = Vec::new();
        let mut train_adj = Vec::new();
        let mut test_images = Vec::new();
        let mut test_gt = Vec::new();
        let mut test_mask = Vec::new();
        let mut test_adj = Vec::new();
        files.for_each(|f| {
            let file = f.unwrap();
            let filename = file.file_name().into_string().unwrap();
            let path = file.path().display().to_string();
            if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_IMAGES_FILE)) {
                train_images.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_GT_FILE)) {
                train_gt.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_MASK_FILE)) {
                train_mask.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TRAIN_ADJ_FILE)) {
                train_adj.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_IMAGES_FILE)) {
                test_images.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_GT_FILE)) {
                test_gt.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_MASK_FILE)) {
                test_mask.push(path);
            } else if filename.starts_with(&get_target_filename(TEXT_DET_TEST_ADJ_FILE)) {
                test_adj.push(path);
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
            train_images,
            train_gt,
            train_mask,
            train_adj,
            test_images,
            test_gt,
            test_mask,
            test_adj,
        })
    } else {
        Err(anyhow!("The directory doesn't exist"))
    }
}

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
    let window_size;
    let target_dir;
    if train {
        images_file = get_target_filename(TEXT_DET_TRAIN_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TRAIN_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TRAIN_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TRAIN_ADJ_FILE);
        window_size = 40;
        target_dir = "train";
    } else {
        images_file = get_target_filename(TEXT_DET_TEST_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TEST_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TEST_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TEST_ADJ_FILE);
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
                let (images, gts, masks, adjust_values) = chunk
                    .par_iter()
                    .fold(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::new(),
                            )
                        },
                        |(mut im_acc, mut gt_acc, mut mask_acc, mut adjust_acc), filename| {
                            if TEXT_DET_FILE_NAME_FORMAT_REGEX.captures(filename).is_some() {
                                match load_text_detection_image(filename, dim) {
                                    Ok((im, gt, mask, adj_values)) => {
                                        if im_acc.numel() == 1 {
                                            return (im, gt, mask, adj_values);
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
                            (im_acc, gt_acc, mask_acc, adjust_acc)
                        },
                    )
                    .reduce(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::new(),
                            )
                        },
                        |(im_acc, gt_acc, mask_acc, adj_acc),
                         (part_im, part_gt, part_mask, part_adj)| {
                            if im_acc.numel() == 1 {
                                return (part_im, part_gt, part_mask, part_adj);
                            }
                            (
                                Tensor::cat(&[im_acc, part_im], 0),
                                Tensor::cat(&[gt_acc, part_gt], 0),
                                Tensor::cat(&[mask_acc, part_mask], 0),
                                Tensor::cat(&[adj_acc, part_adj], 0),
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

/// Finds contours on the provided image. Works on binarized images only.
pub fn find_contours(original_image: &GrayImage) -> Result<Vec<Vec<(usize, usize)>>> {
    let mut nbd = 1;
    let mut _lnbd = 1;
    let mut pos2 = (0, 0);
    let mut skip_tracing;
    let mut image =
        vec![vec![0i32; original_image.height() as usize]; original_image.width() as usize];

    for y in 0..original_image.height() {
        for x in 0..original_image.width() {
            if original_image.get_pixel(x, y).0[0] > 0 {
                image[x as usize][y as usize] = 1;
            }
        }
    }
    let mut neighbour_indices_diffs = VecDeque::from(vec![
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ]);
    let mut x = 0;
    let mut y = 0;
    let last_pixel = (image.len() - 1, image[0].len() - 1);

    let mut contours = Vec::new();

    while (x, y) != last_pixel {
        if image[x][y] != 0 {
            skip_tracing = false;
            if image[x][y] == 1 && x > 0 && image[x - 1][y] == 0 {
                nbd += 1;
                pos2 = (x - 1, y);
            } else if image[x][y] > 0 && x < image.len() - 1 && image[x + 1][y] == 0 {
                nbd += 1;
                pos2 = (x + 1, y);
                if image[x][y] > 1 {
                    _lnbd = image[x][y];
                }
            } else {
                skip_tracing = true;
            }

            if !skip_tracing {
                let x_i32 = x as i32;
                let y_i32 = y as i32;
                let initial_pos_diff = (pos2.0 as i32 - x_i32, pos2.1 as i32 - y_i32);
                let rotate_pos = neighbour_indices_diffs
                    .iter()
                    .position(|&x| x == initial_pos_diff)
                    .unwrap();
                neighbour_indices_diffs.rotate_left(rotate_pos);
                if let Some(pos1) = neighbour_indices_diffs
                    .iter()
                    .find(|(x_diff, y_diff)| {
                        let curr_x = x_i32 + *x_diff;
                        let curr_y = y_i32 + *y_diff;
                        curr_x > -1
                            && curr_x < image.len() as i32
                            && curr_y > -1
                            && curr_y < image[0].len() as i32
                            && image[curr_x as usize][curr_y as usize] > 0
                    })
                    .map(|diff| ((x as i32 + diff.0) as usize, (y as i32 + diff.1) as usize))
                {
                    contours.push(Vec::new());
                    pos2 = pos1;
                    let mut pos3 = (x, y);
                    loop {
                        contours[nbd as usize - 2].push(pos3);
                        let initial_pos_diff =
                            (pos2.0 as i32 - pos3.0 as i32, pos2.1 as i32 - pos3.1 as i32);
                        let rotate_pos = neighbour_indices_diffs
                            .iter()
                            .position(|&x| x == initial_pos_diff)
                            .unwrap();
                        neighbour_indices_diffs.rotate_left(rotate_pos);
                        let pos4 = neighbour_indices_diffs
                            .iter()
                            .rev()
                            .find(|(x_diff, y_diff)| {
                                let curr_x = pos3.0 as i32 + *x_diff;
                                let curr_y = pos3.1 as i32 + *y_diff;
                                curr_x > -1
                                    && curr_x < image.len() as i32
                                    && curr_y > -1
                                    && curr_y < image[0].len() as i32
                                    && image[curr_x as usize][curr_y as usize] != 0
                            })
                            .map(|diff| {
                                (
                                    (pos3.0 as i32 + diff.0) as usize,
                                    (pos3.1 as i32 + diff.1) as usize,
                                )
                            })
                            .unwrap();

                        if pos3.0 + 1 >= image.len() || image[pos3.0 + 1][pos3.1] == 0 {
                            image[pos3.0][pos3.1] = -nbd;
                        }
                        if (pos3.0 + 1 >= image.len() || image[pos3.0 + 1][pos3.1] != 0)
                            && image[pos3.0][pos3.1] == 1
                        {
                            image[pos3.0][pos3.1] = nbd;
                        }
                        if pos4 == (x, y) && pos3 == pos1 {
                            break;
                        }
                        pos2 = pos3;
                        pos3 = pos4;
                    }
                } else {
                    image[x][y] = -nbd;
                }
            }

            if image[x][y] != 1 {
                _lnbd = image[x][y].abs();
            }
        }
        if x == last_pixel.0 {
            x = 0;
            y += 1;
            _lnbd = 1;
        } else {
            x += 1;
        }
    }

    Ok(contours)
}

fn arc_lenght(arc: &[(usize, usize)], closed: bool) -> f64 {
    let mut length = arc.windows(2).fold(0., |acc, pts| {
        acc + ((pts[0].0 as f64 - pts[1].0 as f64).powf(2.)
            + (pts[0].1 as f64 - pts[1].1 as f64).powf(2.))
        .sqrt()
    });
    if closed {
        length += ((arc[0].0 as f64 - arc[arc.len() - 1].0 as f64).powf(2.)
            + (arc[0].1 as f64 - arc[arc.len() - 1].1 as f64).powf(2.))
        .sqrt();
    }
    length
}

fn approx_poly_dp(curve: &[(usize, usize)], epsilon: f64, closed: bool) -> Vec<(usize, usize)> {
    // Find the point with the maximum distance
    let mut dmax = 0.;
    let mut index = 0;
    let end = curve.len() - 1;
    let line_args = line_params(&[curve[0], curve[end]]);
    for (i, point) in curve.iter().enumerate().skip(1) {
        let d = perpendicular_distance(line_args, *point);
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    let mut res = Vec::new();

    // If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon {
        // Recursive call
        let mut partial1 = approx_poly_dp(&curve[0..=index], epsilon, false);
        let mut partial2 = approx_poly_dp(&curve[index..=end], epsilon, false);

        // Build the result list
        partial1.pop();
        res.append(&mut partial1);
        res.append(&mut partial2);
    } else {
        res.push(curve[0]);
        res.push(curve[end]);
    }

    if closed {
        res.pop();
    }

    res
}

fn line_params(points: &[(usize, usize)]) -> (f64, f64, f64) {
    let p1 = points[0];
    let p2 = points[1];
    let a = p1.1 as f64 - p2.1 as f64;
    let b = p2.0 as f64 - p1.0 as f64;
    let c = (p1.0 * p2.1) as f64 - (p2.0 * p1.1) as f64;

    (a, b, c)
}

#[allow(clippy::many_single_char_names)]
fn perpendicular_distance(line_args: (f64, f64, f64), point: (usize, usize)) -> f64 {
    let (a, b, c) = line_args;
    let (x, y) = point;

    (a * x as f64 + b * y as f64 + c).abs() / (a.powf(2.) + b.powf(2.)).sqrt()
}

pub fn min_area_rect(contour: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let hull = convex_hull(&contour);
    match hull.len() {
        0 => panic!("no points are defined"),
        1 => vec![hull[0]; 4],
        2 => vec![hull[0], hull[1], hull[1], hull[0]],
        _ => rotating_calipers(&hull),
    }
}

fn rotating_calipers(points: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let n = points.len();
    let edges: Vec<(f64, f64)> = (0..n - 1)
        .map(|i| {
            let next = i + 1;
            (
                points[next].0 as f64 - points[i].0 as f64,
                points[next].1 as f64 - points[i].1 as f64,
            )
        })
        .collect();

    let mut edge_angles: Vec<f64> = edges
        .iter()
        .map(|e| ((e.1.atan2(e.0) + PI) % (PI / 2.)).abs())
        .collect();
    edge_angles.dedup();

    let mut min_area = std::f64::MAX;
    let mut res = vec![(0., 0.); 4];
    for angle in edge_angles {
        let r = [[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];

        let rotated_points: Vec<(f64, f64)> = points
            .iter()
            .map(|p| {
                (
                    p.0 as f64 * r[0][0] + p.1 as f64 * r[1][0],
                    p.0 as f64 * r[0][1] + p.1 as f64 * r[1][1],
                )
            })
            .collect();
        let (min_x, max_x, min_y, max_y) = rotated_points.iter().fold(
            (std::f64::MAX, std::f64::MIN, std::f64::MAX, std::f64::MIN),
            |acc, p| {
                (
                    acc.0.min(p.0),
                    acc.1.max(p.0),
                    acc.2.min(p.1),
                    acc.3.max(p.1),
                )
            },
        );
        let width = max_x - min_x;
        let height = max_y - min_y;
        let area = width * height;
        if area < min_area {
            min_area = area;

            res[0] = (
                max_x * r[0][0] + min_y * r[0][1],
                max_x * r[1][0] + min_y * r[1][1],
            );
            res[1] = (
                min_x * r[0][0] + min_y * r[0][1],
                min_x * r[1][0] + min_y * r[1][1],
            );
            res[2] = (
                min_x * r[0][0] + max_y * r[0][1],
                min_x * r[1][0] + max_y * r[1][1],
            );
            res[3] = (
                max_x * r[0][0] + max_y * r[0][1],
                max_x * r[1][0] + max_y * r[1][1],
            );
        }
    }

    res.sort_by(|a, b| {
        if a.0 < b.0 {
            std::cmp::Ordering::Less
        } else if a.0 > b.0 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let i1 = if res[1].1 > res[0].1 { 0 } else { 1 };
    let i2 = if res[3].1 > res[2].1 { 2 } else { 3 };
    let i3 = if res[3].1 > res[2].1 { 3 } else { 2 };
    let i4 = if res[1].1 > res[0].1 { 1 } else { 0 };
    vec![
        (res[i1].0.floor() as usize, res[i1].1.floor() as usize),
        (res[i2].0.ceil() as usize, res[i2].1.floor() as usize),
        (res[i3].0.ceil() as usize, res[i3].1.ceil() as usize),
        (res[i4].0.floor() as usize, res[i4].1.ceil() as usize),
    ]
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

///
/// Finds points of the smallest convex polygon that contains all the contour points.
/// https://en.wikipedia.org/wiki/Graham_scan
///
fn convex_hull(points_slice: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut points = Vec::from(points_slice);
    let (start_point_pos, start_point) = points.iter().enumerate().fold(
        (std::usize::MAX, (std::usize::MAX, std::usize::MAX)),
        |(pos, p0), (i, &point)| {
            if point.1 < p0.1 || point.1 == p0.1 && point.0 < p0.0 {
                return (i, point);
            }
            (pos, p0)
        },
    );
    points.swap(0, start_point_pos);
    points.remove(0);
    points.sort_by(|a, b| {
        let orientation = get_orientation(&start_point, a, b);
        if orientation == 0 {
            if get_distance(&start_point, a) < get_distance(&start_point, b) {
                return std::cmp::Ordering::Less;
            }
            return std::cmp::Ordering::Greater;
        }
        if orientation == 2 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    let mut iter = points.iter().peekable();
    let mut remaining_points = Vec::with_capacity(points.len());
    while let Some(mut p) = iter.next() {
        while iter.peek().is_some() && get_orientation(&start_point, p, iter.peek().unwrap()) == 0 {
            p = iter.next().unwrap();
        }
        remaining_points.push(p);
    }

    let mut stack = vec![start_point];

    for point in points.iter() {
        while stack.len() > 1
            && get_orientation(&stack[stack.len() - 2], &stack[stack.len() - 1], point) != 2
        {
            stack.pop();
        }
        stack.push(*point);
    }
    stack
}

fn get_orientation(p: &(usize, usize), q: &(usize, usize), r: &(usize, usize)) -> u8 {
    let val = (q.1 as i32 - p.1 as i32) * (r.0 as i32 - q.0 as i32)
        - (q.0 as i32 - p.0 as i32) * (r.1 as i32 - q.1 as i32);
    match val.cmp(&0) {
        Ordering::Equal => 0,   // colinear
        Ordering::Greater => 1, // clockwise (right)
        Ordering::Less => 2,    // counter-clockwise (left)
    }
}

pub fn get_distance(p1: &(usize, usize), p2: &(usize, usize)) -> f64 {
    ((p1.0 as f64 - p2.0 as f64) * (p1.0 as f64 - p2.0 as f64)
        + (p1.1 as f64 - p2.1 as f64) * (p1.1 as f64 - p2.1 as f64))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::open;
    use imageproc::drawing::{draw_polygon_mut, Point};
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
    fn find_contours_test() -> Result<()> {
        let image = open("test_data/polygon.png")?.to_luma();
        let contours = find_contours(&image)?;
        assert_eq!(contours.len(), 6);
        Ok(())
    }

    #[test]
    fn line_params_test() {
        let p1 = (5, 7);
        let p2 = (10, 3);
        assert_eq!(line_params(&[p1, p2]), (4., 5., -55.));
    }

    #[test]
    fn perpendicular_distance_test() {
        let line_args = (8., 7., 5.);
        let point = (2, 3);
        assert!(perpendicular_distance(line_args, point) - 3.9510276472 < 1e-10);
    }

    #[test]
    fn get_contours_approx_points() -> Result<()> {
        let mut image = GrayImage::from_pixel(300, 300, Luma([0]));
        let white = Luma([255]);

        let star = vec![
            Point::new(100, 20),
            Point::new(120, 35),
            Point::new(140, 30),
            Point::new(115, 45),
            Point::new(130, 60),
            Point::new(100, 50),
            Point::new(80, 55),
            Point::new(90, 40),
            Point::new(60, 25),
            Point::new(90, 35),
        ];
        draw_polygon_mut(&mut image, &star, white);
        let contours = find_contours(&image)?;
        let c1_approx = approx_poly_dp(&contours[0], arc_lenght(&contours[0], true) * 0.01, true);
        assert_eq!(
            c1_approx,
            vec![
                (100, 20),
                (90, 35),
                (60, 25),
                (90, 40),
                (80, 55),
                (101, 50),
                (130, 60),
                (115, 45),
                (140, 30),
                (120, 35)
            ]
        );
        Ok(())
    }

    #[test]
    fn get_convex_hull_points() {
        let star = vec![
            (100, 20),
            (90, 35),
            (60, 25),
            (90, 40),
            (80, 55),
            (101, 50),
            (130, 60),
            (115, 45),
            (140, 30),
            (120, 35),
        ];
        let points = convex_hull(&star);
        assert_eq!(
            points,
            [(100, 20), (140, 30), (130, 60), (80, 55), (60, 25)]
        );
    }

    #[test]
    fn min_area_test() {
        assert_eq!(
            min_area_rect(&[(100, 20), (140, 30), (130, 60), (80, 55), (60, 25)]),
            [(60, 16), (141, 24), (137, 61), (57, 53)]
        )
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
            TEXT_DET_TEST_IMAGES_FILE,
            TEXT_DET_TEST_GT_FILE,
            TEXT_DET_TEST_MASK_FILE,
            TEXT_DET_TEST_ADJ_FILE,
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

        let test_images_tensor = Tensor::load(&filenames[4])?;
        assert_eq!(test_images_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_images_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_img1)?
        );
        assert_eq!(
            test_images_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_img2)?
        );

        let test_gts_tensor = Tensor::load(&filenames[5])?;
        assert_eq!(test_gts_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_gts_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_gt_img1)? / 255.
        );
        assert_eq!(
            test_gts_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_gt_img2)? / 255.
        );

        let test_masks_tensor = Tensor::load(&filenames[6])?;
        assert_eq!(test_masks_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_masks_tensor.get(0).to_kind(Kind::Double),
            convert_image_to_tensor(&test_mask_img1)? / 255.
        );
        assert_eq!(
            test_masks_tensor.get(1).to_kind(Kind::Double),
            convert_image_to_tensor(&test_mask_img2)? / 255.
        );

        let test_adj_tensor = Tensor::load(&filenames[7])?;
        assert_eq!(test_adj_tensor.size(), [2, 2]);

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
}
