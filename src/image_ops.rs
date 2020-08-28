use log::info;

use super::utils::{VALUES_COUNT, VALUES_MAP};
use anyhow::{anyhow, Result};
use image::{imageops::FilterType, open, DynamicImage, GrayImage, Luma};
use imageproc::definitions::Image;
use imageproc::drawing::{self, Point};
use log::debug;
use rayon::prelude::*;
use regex::Regex;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;
use tch::vision::dataset::Dataset;
use tch::{Kind, Tensor};

const TRAIN_IMAGES_FILE: &str = "training_images_data";
const TRAIN_LABELS_FILE: &str = "training_labels_data";
const TEST_IMAGES_FILE: &str = "test_images_data";
const TEST_LABELS_FILE: &str = "test_labels_data";
const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 800;
const WHITE_COLOR: Luma<u8> = Luma([255]);
const BLACK_COLOR: Luma<u8> = Luma([0]);

lazy_static! {
    static ref FILE_NAME_FORMAT_REGEX: Regex =
        Regex::new(r"(.*)-(upper|lower|num)-([a-zA-z0-9])-img\.png").unwrap();
}

pub fn load_values() -> Result<Dataset> {
    info!("loading values");
    let instant = Instant::now();
    let train_images = load_values_from_file(TRAIN_IMAGES_FILE)?;
    let train_labels = load_values_from_file(TRAIN_LABELS_FILE)?;
    let test_images = load_values_from_file(TEST_IMAGES_FILE)?;
    let test_labels = load_values_from_file(TEST_LABELS_FILE)?;
    info!(
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
    let image_pixels = get_image_pixel_colors_grayscale(file_path)?;

    let images_tensor = Tensor::of_slice(&image_pixels)
        .view((1, image_pixels.len() as i64))
        .to_kind(Kind::Float)
        / 255.;
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
        let is_images = file_path == TEST_IMAGES_FILE || file_path == TRAIN_IMAGES_FILE;
        if file_path == TEST_IMAGES_FILE || file_path == TEST_LABELS_FILE {
            images_path = TEST_IMAGES_FILE;
            labels_path = TEST_LABELS_FILE;
            images_dir = "images/test";
        } else {
            images_path = TRAIN_IMAGES_FILE;
            labels_path = TRAIN_LABELS_FILE;
            images_dir = "images/training";
        }
        if is_images {
            let images_data = load_images(images_dir)?;
            images_data.save(images_path)?;
            data_tensor = images_data;
        } else {
            let labels_data = load_labels(images_dir)?;
            labels_data.save(labels_path)?;
            data_tensor = labels_data;
        }
    }

    Ok(data_tensor)
}

fn load_images(dir_path: &str) -> Result<Tensor> {
    if let Ok(files) = fs::read_dir(dir_path) {
        let files_info: Vec<std::fs::DirEntry> = files.map(|f| f.unwrap()).collect();
        let (pixels, rows, cols) = files_info
            .par_iter()
            .fold(
                || (Vec::new(), 0, 0),
                |(mut p_vec, mut rows, mut cols), file| {
                    let filename = file.file_name();
                    if FILE_NAME_FORMAT_REGEX
                        .captures(filename.to_str().unwrap())
                        .is_some()
                    {
                        rows += 1;
                        let mut image_pixels =
                            get_image_pixel_colors_grayscale(&file.path().display().to_string())
                                .unwrap();
                        cols = image_pixels.len();
                        p_vec.append(&mut image_pixels);
                    }
                    (p_vec, rows, cols)
                },
            )
            .reduce(
                || (Vec::new(), 0, 0),
                |(mut acc_vec, total_rows, _), (mut partial, p_rows, p_cols)| {
                    acc_vec.append(&mut partial);
                    (acc_vec, total_rows + p_rows, p_cols)
                },
            );
        info!("Creating tensor for images r-{} c-{}", rows, cols);
        let images_tensor = Tensor::of_slice(&pixels)
            .view((rows as i64, cols as i64))
            .to_kind(Kind::Float)
            / 255.;
        return Ok(images_tensor);
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
                    if let Some(caps) = FILE_NAME_FORMAT_REGEX.captures(filename) {
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
        info!("Creating tensor for labels r-{} c-1", rows);
        let labels_tensor = Tensor::of_slice(&labels).to_kind(Kind::Int64);
        return Ok(labels_tensor);
    }
    Err(anyhow!("Could not open dir {}", dir_path))
}

fn get_image_pixel_colors_grayscale(file_path: &str) -> Result<Vec<u8>> {
    let mut pixels = Vec::new();
    let rgba_image = open(file_path)?.into_rgba();
    let gray_image = DynamicImage::ImageRgba8(rgba_image).into_luma();
    for i in 0..gray_image.height() {
        for j in 0..gray_image.width() {
            pixels.push(gray_image.get_pixel(i, j).0[0]);
        }
    }
    Ok(pixels)
}

pub fn preprocess_image(file_path: &str) -> Result<(DynamicImage, f64, f64)> {
    let instant = Instant::now();
    let rgba_image = open(file_path)?.into_rgba();
    let original_width = rgba_image.width();
    let original_height = rgba_image.height();
    let dyn_image = DynamicImage::ImageRgba8(rgba_image)
        .resize(800, 800, FilterType::Triangle)
        .to_luma();
    let mut dyn_image_800x800 = DynamicImage::new_luma8(800, 800);

    // correction for pixel coords
    let adjust_x = dyn_image.width() as f64 / original_width as f64;
    let adjust_y = dyn_image.height() as f64 / original_height as f64;

    for (x, y, p) in dyn_image.enumerate_pixels() {
        *dyn_image_800x800
            .as_mut_luma8()
            .unwrap()
            .get_pixel_mut(x, y) = *p;
    }
    debug!(
        "finished preprocessing in {} ns",
        instant.elapsed().as_nanos()
    );
    Ok((dyn_image_800x800, adjust_x, adjust_y))
}

fn generate_gt_and_mask_images(
    polygons: &[Tensor],
    adjust_x: f64,
    adjust_y: f64,
) -> Result<(Image<Luma<u8>>, Image<Luma<u8>>)> {
    let mut gt_image = GrayImage::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    let mut mask_temp = DynamicImage::new_luma8(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    mask_temp.invert();
    let mut mask_image = mask_temp.to_luma();
    for poly in polygons {
        let (num_of_points, _) = poly.size2()?;
        let poly_values = (0..num_of_points)
            .map(|i| {
                Point::new(
                    (poly.double_value(&[i, 0]) * adjust_x) as i32,
                    (poly.double_value(&[i, 1]) * adjust_y) as i32,
                )
            })
            .collect::<Vec<Point<i32>>>();
        drawing::draw_convex_polygon_mut(&mut gt_image, &poly_values, WHITE_COLOR);
        drawing::draw_convex_polygon_mut(&mut mask_image, &poly_values, BLACK_COLOR);
    }
    Ok((gt_image, mask_image))
}

fn load_polygons(file_path: &str) -> Result<Vec<Tensor>> {
    let polygons;
    if let Ok(mut file) = File::open(file_path) {
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        polygons = content
            .split_terminator('\n')
            .map(|row| {
                // row is in format
                // x1,y1,x2,y2,...,x{n},y{n},{FOUND TEXT}
                // where n is number of points in the polygon (not a fixed value)

                let values = row
                    .split_terminator(',')
                    .filter_map(|v| {
                        if let Ok(value_i32) = v.parse::<i32>() {
                            Some(value_i32)
                        } else {
                            // ignore the text
                            None
                        }
                    })
                    .collect::<Vec<i32>>();
                Tensor::of_slice(&values).view((-1, 2))
            })
            .collect();
    } else {
        return Err(anyhow!("didn't find file {}", file_path));
    }

    Ok(polygons)
}

pub fn load_text_detection_image(file_path: &str) -> Result<()> {
    let instant = Instant::now();
    let (preprocessed_image, adjust_x, adjust_y) = preprocess_image(file_path)?;

    let path_parts = file_path.split_terminator('/').collect::<Vec<&str>>();
    let last_idx = path_parts.len() - 1;
    let polygons = load_polygons(&format!(
        "text-detection-images/totaltext/gts/{}/{}.txt",
        path_parts[last_idx - 1],
        path_parts[last_idx]
    ))?;

    let (gt_image, mask_image) = generate_gt_and_mask_images(&polygons, adjust_x, adjust_y)?;

    debug!(
        "finished loading and preparing text detection images in {:?} ns",
        instant.elapsed().as_nanos()
    );

    Ok(())
}
