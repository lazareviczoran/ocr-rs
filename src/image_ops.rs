use super::dataset::TextDetectionDataset;
use super::utils::{VALUES_COUNT, VALUES_MAP};
use anyhow::{anyhow, Result};
use image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::drawing::{self, Point};
use log::{debug, error, trace};
use rayon::prelude::*;
use regex::Regex;
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
pub const TEXT_DET_TEST_IMAGES_FILE: &str = "test_images_data_text_det";
pub const TEXT_DET_TEST_GT_FILE: &str = "test_gt_data_text_det";
pub const TEXT_DET_TEST_MASK_FILE: &str = "test_mask_data_text_det";
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
        let (pixels, rows, cols) = files_info
            .par_iter()
            .fold(
                || (Vec::new(), 0, 0),
                |(mut p_vec, mut rows, mut cols), file| {
                    let filename = file.file_name();
                    if CHAR_REC_FILE_NAME_FORMAT_REGEX
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
        trace!("Creating tensor for images r-{} c-{}", rows, cols);
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

fn get_image_pixel_colors_grayscale(file_path: &str) -> Result<Vec<u8>> {
    let rgba_image = open(file_path)?.into_rgba();
    let gray_image = DynamicImage::ImageRgba8(rgba_image).into_luma();
    let mut pixels = vec![0; (gray_image.width() * gray_image.height()) as usize];
    pixels.par_iter_mut().enumerate().for_each(|(n, val)| {
        let y = n as u32 / gray_image.width();
        let x = n as u32 - y * gray_image.width();
        *val = gray_image.get_pixel(x, y).0[0];
    });
    Ok(pixels)
}

pub fn preprocess_image(file_path: &str) -> Result<(GrayImage, f64, f64)> {
    let instant = Instant::now();
    let rgba_image = open(file_path)?.into_rgba();
    let original_width = rgba_image.width();
    let original_height = rgba_image.height();
    let dyn_image = DynamicImage::ImageRgba8(rgba_image)
        .resize(DEFAULT_WIDTH, DEFAULT_HEIGHT, FilterType::Triangle)
        .to_luma();

    // correction for pixel coords
    let adjust_x = dyn_image.width() as f64 / original_width as f64;
    let adjust_y = dyn_image.height() as f64 / original_height as f64;

    let mut pixel_values = vec![0; (DEFAULT_WIDTH * DEFAULT_HEIGHT) as usize];
    pixel_values
        .par_iter_mut()
        .enumerate()
        .for_each(|(n, val)| {
            let y = n as u32 / DEFAULT_WIDTH;
            let x = n as u32 - y * DEFAULT_WIDTH;
            if x < dyn_image.width() && y < dyn_image.height() {
                *val = dyn_image.get_pixel(x, y).0[0];
            }
        });
    trace!(
        "finished preprocessing in {} ns",
        instant.elapsed().as_nanos()
    );
    Ok((
        ImageBuffer::from_vec(DEFAULT_WIDTH, DEFAULT_HEIGHT, pixel_values).unwrap(),
        adjust_x,
        adjust_y,
    ))
}

fn generate_gt_and_mask_images(
    polygons: &[Tensor],
    adjust_x: f64,
    adjust_y: f64,
) -> Result<(GrayImage, GrayImage, Vec<bool>)> {
    let mut gt_image = GrayImage::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    let mut mask_temp = DynamicImage::new_luma8(DEFAULT_WIDTH, DEFAULT_HEIGHT);
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

pub fn load_text_detection_image(file_path: &str) -> Result<(Tensor, Tensor, Tensor)> {
    let instant = Instant::now();
    let (preprocessed_image, adjust_x, adjust_y) = preprocess_image(file_path)?;

    let path_parts = file_path.split_terminator('/').collect::<Vec<&str>>();
    let last_idx = path_parts.len() - 1;
    let polygons = load_polygons(&format!(
        "{}/gts/{}/{}.txt",
        TEXT_DET_IMAGES_PATH,
        path_parts[last_idx - 1],
        path_parts[last_idx]
    ))?;

    let (gt_image, mask_image, _ignore_flags) =
        generate_gt_and_mask_images(&polygons, adjust_x, adjust_y)?;
    let image_tensor = create_tensor_from_image(&preprocessed_image)?;
    let gt_tensor = (create_tensor_from_image(&gt_image)? / 255.).to_kind(Kind::Uint8);
    let mask_tensor = (create_tensor_from_image(&mask_image)? / 255.).to_kind(Kind::Uint8);

    trace!(
        "finished loading and preparing text detection images in {:?} ns",
        instant.elapsed().as_nanos()
    );

    Ok((image_tensor, gt_tensor, mask_tensor))
}

fn create_tensor_from_image(image: &GrayImage) -> Result<Tensor> {
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
    Ok(Tensor::of_slice(&pixel_values).view((w as i64, h as i64)))
}

pub fn load_text_detection_images() -> Result<TextDetectionDataset> {
    trace!("loading text detection values");
    let instant = Instant::now();
    let (train_images, train_gt, train_mask) =
        load_text_det_values_from_file(TEXT_DET_IMAGES_PATH, true)?;
    let (test_images, test_gt, test_mask) =
        load_text_det_values_from_file(TEXT_DET_IMAGES_PATH, false)?;
    trace!(
        "Finished loading values in {} ms",
        instant.elapsed().as_millis()
    );

    Ok(TextDetectionDataset {
        train_images,
        train_gt,
        train_mask,
        test_images,
        test_gt,
        test_mask,
    })
}

pub fn generate_text_det_tensor_chunks(images_base_dir: &str, train: bool) -> Result<()> {
    let instant = Instant::now();
    let images_file;
    let gt_file;
    let mask_file;
    let window_size;
    let target_dir;
    if train {
        images_file = TEXT_DET_TRAIN_IMAGES_FILE;
        gt_file = TEXT_DET_TRAIN_GT_FILE;
        mask_file = TEXT_DET_TRAIN_MASK_FILE;
        window_size = 40;
        target_dir = "train";
    } else {
        images_file = TEXT_DET_TEST_IMAGES_FILE;
        gt_file = TEXT_DET_TEST_GT_FILE;
        mask_file = TEXT_DET_TEST_MASK_FILE;
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
                let (images, gts, masks) = chunk
                    .par_iter()
                    .fold(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                            )
                        },
                        |(mut im_acc, mut gt_acc, mut mask_acc), filename| {
                            if TEXT_DET_FILE_NAME_FORMAT_REGEX.captures(filename).is_some() {
                                match load_text_detection_image(filename) {
                                    Ok((im, gt, mask)) => {
                                        if im_acc.numel() == 1 {
                                            return (im, gt, mask);
                                        }
                                        im_acc = Tensor::cat(&[im_acc, im], 0);
                                        gt_acc = Tensor::cat(&[gt_acc, gt], 0);
                                        mask_acc = Tensor::cat(&[mask_acc, mask], 0);
                                    }
                                    Err(msg) => {
                                        error!("Error while loading single image data: {}", msg);
                                    }
                                }
                            }
                            (im_acc, gt_acc, mask_acc)
                        },
                    )
                    .reduce(
                        || {
                            (
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                                Tensor::of_slice(&[0]),
                            )
                        },
                        |(im_acc, gt_acc, mask_acc), (part_im, part_gt, part_mask)| {
                            if im_acc.numel() == 1 {
                                return (part_im, part_gt, part_mask);
                            }
                            (
                                Tensor::cat(&[im_acc, part_im], 0),
                                Tensor::cat(&[gt_acc, part_gt], 0),
                                Tensor::cat(&[mask_acc, part_mask], 0),
                            )
                        },
                    );
                let mut save_res = images
                    .view((-1, DEFAULT_WIDTH as i64, DEFAULT_HEIGHT as i64))
                    .save(format!("{}.{}", images_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving image tensor {}", msg);
                }
                save_res = gts
                    .view((-1, DEFAULT_WIDTH as i64, DEFAULT_HEIGHT as i64))
                    .save(format!("{}.{}", gt_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving gt tensor {}", msg);
                }
                save_res = masks
                    .view((-1, DEFAULT_WIDTH as i64, DEFAULT_HEIGHT as i64))
                    .save(format!("{}.{}", mask_file, pos));
                if let Err(msg) = save_res {
                    error!("Error while saving mask tensor {}", msg);
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

fn load_text_det_values_from_file(
    images_base_dir: &str,
    train: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    debug!("loading text detection file train: {}", train);
    let (images_file, gt_file, mask_file) = if train {
        (
            TEXT_DET_TRAIN_IMAGES_FILE,
            TEXT_DET_TRAIN_GT_FILE,
            TEXT_DET_TRAIN_MASK_FILE,
        )
    } else {
        (
            TEXT_DET_TEST_IMAGES_FILE,
            TEXT_DET_TEST_GT_FILE,
            TEXT_DET_TEST_MASK_FILE,
        )
    };
    if Path::new(images_file).exists()
        && Path::new(gt_file).exists()
        && Path::new(mask_file).exists()
    {
        Ok((
            Tensor::load(images_file)?,
            Tensor::load(gt_file)?,
            Tensor::load(mask_file)?,
        ))
    } else {
        Err(anyhow!("One of the files doesn't exist"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Tensor};

    #[test]
    fn tensor_cat() {
        let device = Device::cuda_if_available();
        let zeros = Tensor::zeros(&[2, 3], (Kind::Float, device));
        let ones = Tensor::ones(&[2, 3], (Kind::Float, device));
        assert_eq!(
            Tensor::cat(&[zeros, ones], 0).view((-1, 2, 3)),
            Tensor::of_slice(&[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).view((2, 2, 3))
        );
    }
}
