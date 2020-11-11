use super::dataset::TextDetectionDataset;
use super::measure_time;
use super::utils::{save_tensor, VALUES_COUNT, VALUES_MAP};
use anyhow::{anyhow, Result};
use geo::{LineString, MultiPolygon, Polygon};
use image::{imageops::FilterType, open, DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::definitions::{HasBlack, HasWhite, Point};
use imageproc::drawing::draw_polygon_mut;
use log::{error, info, trace};
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use tch::vision::dataset::Dataset;
use tch::{Kind, Tensor};

pub const CHAR_REC_IMAGES_PATH: &str = "./images";
pub const CHAR_REC_TRAIN_IMAGES_FILE: &str = "training_images_data_char_rec";
pub const CHAR_REC_TRAIN_LABELS_FILE: &str = "training_labels_data_char_rec";
pub const CHAR_REC_TEST_IMAGES_FILE: &str = "test_images_data_char_rec";
pub const CHAR_REC_TEST_LABELS_FILE: &str = "test_labels_data_char_rec";
pub const TEXT_DET_IMAGES_PATH: &str = "./text-detection-images";
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
pub const MIN_TEXT_SIZE: u32 = 8;

lazy_static! {
    static ref CHAR_REC_FILE_NAME_FORMAT_REGEX: Regex =
        Regex::new(r"(.*)-(upper|lower|num)-([a-zA-z0-9])-img\.png").unwrap();
    static ref TEXT_DET_FILE_NAME_FORMAT_REGEX: Regex = Regex::new(r"img[0-9]+\.jpg").unwrap();
}

pub fn load_values<T: AsRef<Path>>(target_dir: T) -> Result<Dataset> {
    trace!("loading character recognition values");
    let (train_images, train_labels, test_images, test_labels) = measure_time!(
        "loading values",
        || -> Result<(Tensor, Tensor, Tensor, Tensor)> {
            let train_images = Tensor::load(target_dir.as_ref().join(CHAR_REC_TRAIN_IMAGES_FILE))?;
            let train_labels = Tensor::load(target_dir.as_ref().join(CHAR_REC_TRAIN_LABELS_FILE))?;
            let test_images = Tensor::load(target_dir.as_ref().join(CHAR_REC_TEST_IMAGES_FILE))?;
            let test_labels = Tensor::load(target_dir.as_ref().join(CHAR_REC_TEST_LABELS_FILE))?;
            Ok((train_images, train_labels, test_images, test_labels))
        },
        LogType::Debug
    )?;

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: VALUES_COUNT as i64,
    })
}

pub fn load_image_as_tensor<T: AsRef<Path>>(file_path: T) -> Result<Tensor> {
    let path = file_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("File {} doesn't exist", path.display()));
    }
    let image = open(file_path)?.into_luma();
    let dim = image.width() * image.height();
    let images_tensor = convert_image_to_tensor(&image)?
        .view((1, dim as i64))
        .to_kind(Kind::Float)
        / 255.;
    Ok(images_tensor)
}

pub fn generate_char_rec_tensor_files<T: AsRef<Path>>(images_dir: T, target_dir: T) -> Result<()> {
    let train_images_dir = images_dir.as_ref().join("train");
    let test_images_dir = images_dir.as_ref().join("test");
    let training_images_data = load_images(&train_images_dir)?;
    save_tensor(
        &training_images_data,
        target_dir.as_ref().join(CHAR_REC_TRAIN_IMAGES_FILE),
    )?;
    let training_labels_data = load_labels(train_images_dir)?;
    save_tensor(
        &training_labels_data,
        target_dir.as_ref().join(CHAR_REC_TRAIN_LABELS_FILE),
    )?;
    let test_images_data = load_images(&test_images_dir)?;
    save_tensor(
        &test_images_data,
        target_dir.as_ref().join(CHAR_REC_TEST_IMAGES_FILE),
    )?;
    let test_labels_data = load_labels(test_images_dir)?;
    save_tensor(
        &test_labels_data,
        target_dir.as_ref().join(CHAR_REC_TEST_LABELS_FILE),
    )?;
    info!("Successfully generated tensor files!");

    Ok(())
}

fn load_images<T: AsRef<Path>>(dir_path: T) -> Result<Tensor> {
    let dir = dir_path.as_ref();
    if let Ok(files) = fs::read_dir(dir) {
        let mut files_info: Vec<std::fs::DirEntry> = files.map(|f| f.unwrap()).collect();
        files_info.sort_by_key(|a| a.file_name());
        let images_tensor = files_info
            .par_iter()
            .fold(
                || Tensor::of_slice(&[0]),
                |mut p_vec, file| {
                    let filename = file.file_name();
                    if CHAR_REC_FILE_NAME_FORMAT_REGEX
                        .captures(filename.to_str().unwrap())
                        .is_some()
                    {
                        if let Ok(image) = open(&file.path().display().to_string()) {
                            let gray_img = image.into_luma();
                            let image_tensor = convert_image_to_tensor(&gray_img)
                                .unwrap()
                                .view((1, (gray_img.width() * gray_img.height()) as i64));

                            p_vec = if p_vec.numel() == 1 {
                                image_tensor
                            } else {
                                Tensor::cat(&[p_vec, image_tensor], 0)
                            };
                        }
                    }
                    p_vec
                },
            )
            .reduce(
                || Tensor::of_slice(&[0]),
                |acc_vec, partial| {
                    if acc_vec.numel() == 1 {
                        partial
                    } else {
                        Tensor::cat(&[acc_vec, partial], 0)
                    }
                },
            );
        return Ok(images_tensor.to_kind(Kind::Float) / 255.);
    }
    Err(anyhow!("Could not open dir {}", dir.display()))
}

fn load_labels<T: AsRef<Path>>(dir_path: T) -> Result<Tensor> {
    let dir = dir_path.as_ref();
    if let Ok(files) = fs::read_dir(dir) {
        let mut filenames: Vec<String> = files
            .map(|f| f.unwrap().file_name().into_string().unwrap())
            .collect();
        filenames.sort();
        let labels = filenames
            .par_iter()
            .fold(Vec::new, |mut p_vec, filename| {
                if let Some(caps) = CHAR_REC_FILE_NAME_FORMAT_REGEX.captures(filename) {
                    if let Some(letter) = caps[3].chars().next() {
                        p_vec.push(*VALUES_MAP.get(&letter).unwrap());
                    }
                }
                p_vec
            })
            .reduce(Vec::new, |mut acc_vec, mut partial| {
                acc_vec.append(&mut partial);
                acc_vec
            });
        Ok(Tensor::of_slice(&labels).to_kind(Kind::Int64))
    } else {
        Err(anyhow!("Could not open dir {}", dir.display()))
    }
}

pub fn preprocess_image<T: AsRef<Path>>(
    file_path: T,
    target_dim: (u32, u32),
) -> Result<(GrayImage, f64, f64)> {
    let (width, height) = target_dim;
    let rgba_image = open(file_path.as_ref())?.into_rgba();
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
    Ok((
        ImageBuffer::from_vec(width, height, pixel_values).unwrap(),
        adjust_x,
        adjust_y,
    ))
}

fn generate_gt_and_mask_images(
    polygons: &MultiPolygon<u32>,
    adjust_x: f64,
    adjust_y: f64,
    target_dim: (u32, u32),
) -> Result<(GrayImage, GrayImage, Vec<bool>)> {
    let (width, height) = target_dim;
    let mut gt_image = GrayImage::new(width, height);
    let mut mask_temp = DynamicImage::new_luma8(width, height);
    mask_temp.invert();
    let mut mask_image = mask_temp.to_luma();
    let mut ignore_flags = vec![false; polygons.0.len()];
    for (pos, poly) in polygons.0.iter().enumerate() {
        let ext_poly = poly.exterior();
        let (min_x, max_x, min_y, max_y) =
            ext_poly
                .points_iter()
                .fold((std::u32::MAX, 0, std::u32::MAX, 0), |acc, p| {
                    (
                        acc.0.min(p.x()),
                        acc.1.max(p.x()),
                        acc.2.min(p.y()),
                        acc.3.max(p.y()),
                    )
                });
        let poly_width = max_x - min_x;
        let poly_height = max_y - min_y;
        let poly_values = ext_poly
            .points_iter()
            .take(ext_poly.num_coords() - 1)
            .map(|p| {
                Point::new(
                    (p.x() as f64 * adjust_x) as i32,
                    (p.y() as f64 * adjust_y) as i32,
                )
            })
            .collect::<Vec<Point<i32>>>();
        if poly_values.len() < 4 {
            ignore_flags[pos] = true;
            continue;
        }
        if poly_height.min(poly_width) < MIN_TEXT_SIZE {
            draw_polygon_mut(&mut mask_image, &poly_values, Luma::black());
            ignore_flags[pos] = true;
        } else {
            draw_polygon_mut(&mut gt_image, &poly_values, Luma::white());
        }
    }
    Ok((gt_image, mask_image, ignore_flags))
}

fn load_polygons<T: AsRef<Path>>(file_path: T) -> Result<MultiPolygon<u32>> {
    if let Ok(mut file) = File::open(file_path.as_ref()) {
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        let polygons: Vec<Polygon<u32>> = content
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
                Polygon::new(
                    LineString::from(
                        values
                            .chunks_exact(2)
                            .map(|point| (point[0], point[1]))
                            .collect::<Vec<(u32, u32)>>(),
                    ),
                    vec![],
                )
            })
            .collect();
        Ok(MultiPolygon::from(polygons))
    } else {
        Err(anyhow!("didn't find file {}", file_path.as_ref().display()))
    }
}

pub fn load_text_detection_image<T: AsRef<Path>, H: AsRef<Path>>(
    images_base_dir: T,
    file_path: H,
    target_dim: (u32, u32),
) -> Result<SingleImageData> {
    let path = file_path.as_ref();
    let (preprocessed_image, adjust_x, adjust_y) = preprocess_image(path, target_dim)?;

    let extension = path.extension().unwrap().to_str().unwrap();
    let polygons = load_polygons(
        images_base_dir.as_ref().join("gts").join(
            path.strip_prefix(images_base_dir.as_ref().join("images"))?
                .with_extension(format!("{}.txt", extension)),
        ),
    )?;

    let (gt_image, mask_image, ignore_flags) =
        generate_gt_and_mask_images(&polygons, adjust_x, adjust_y, target_dim)?;

    let image_tensor = convert_image_to_tensor(&preprocessed_image)?.to_kind(Kind::Uint8);
    let gt_tensor = (convert_image_to_tensor(&gt_image)? / 255.).to_kind(Kind::Uint8);
    let mask_tensor = (convert_image_to_tensor(&mask_image)? / 255.).to_kind(Kind::Uint8);
    let adjust_tensor = Tensor::of_slice(&[adjust_x, adjust_y]).view((1, 2));

    Ok(SingleImageData::new(
        image_tensor.view((1, target_dim.1 as i64, target_dim.0 as i64)),
        gt_tensor.view((1, target_dim.1 as i64, target_dim.0 as i64)),
        mask_tensor.view((1, target_dim.1 as i64, target_dim.0 as i64)),
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
    let h = size[0] as u32;
    let w = size[1] as u32;
    let numel = tensor.numel();
    let mut pixel_values = vec![0; numel];
    tensor
        .to_kind(Kind::Uint8)
        .view(-1)
        .copy_data(&mut pixel_values, numel);
    Ok(ImageBuffer::from_vec(w, h, pixel_values).unwrap())
}

pub fn load_text_detection_tensor_files<T: AsRef<Path>>(
    target_dir: T,
) -> Result<TextDetectionDataset> {
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
        let mut files_info: Vec<std::fs::DirEntry> = files.map(|f| f.unwrap()).collect();
        files_info.sort_by_key(|a| a.file_name());
        files_info.iter().for_each(|file| {
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
        Err(anyhow!(
            "The directory {} doesn't exist",
            target_dir.as_ref().display()
        ))
    }
}

#[derive(Debug)]
pub struct TextDetDataBatch {
    images: Tensor,
    gts: Tensor,
    masks: Tensor,
    adjs_values: Tensor,
    polygons: Vec<MultiPolygon<u32>>,
    ignore_flags: Vec<Vec<bool>>,
}
impl TextDetDataBatch {
    pub fn new() -> Self {
        Self {
            images: Tensor::of_slice(&[0]),
            gts: Tensor::of_slice(&[0]),
            masks: Tensor::of_slice(&[0]),
            adjs_values: Tensor::of_slice(&[0]),
            polygons: Vec::new(),
            ignore_flags: Vec::new(),
        }
    }
}

pub struct SingleImageData {
    image_tensor: Tensor,
    gt_tensor: Tensor,
    mask_tensor: Tensor,
    adjs_tensor: Tensor,
    polygons: MultiPolygon<u32>,
    ignore_flags: Vec<bool>,
}
impl SingleImageData {
    pub fn new(
        image_tensor: Tensor,
        gt_tensor: Tensor,
        mask_tensor: Tensor,
        adjs_tensor: Tensor,
        polygons: MultiPolygon<u32>,
        ignore_flags: Vec<bool>,
    ) -> Self {
        Self {
            image_tensor,
            gt_tensor,
            mask_tensor,
            adjs_tensor,
            polygons,
            ignore_flags,
        }
    }
}

pub fn generate_text_det_tensor_chunks<T: AsRef<Path>>(
    images_base_dir: T,
    target_dir: T,
    train: bool,
    dim: (u32, u32),
) -> Result<()> {
    let (window_size, data_type) = if train { (40, "train") } else { (10, "test") };
    let base_dir = images_base_dir.as_ref();
    let images_dir = base_dir.join("images").join(data_type);

    if let Ok(files) = fs::read_dir(&images_dir) {
        let mut files_info: Vec<std::fs::DirEntry> = files.map(|f| f.unwrap()).collect();
        files_info.sort_by_key(|a| a.file_name());

        files_info
            .chunks(window_size)
            .enumerate()
            .for_each(|(pos, chunk)| {
                let batch_data = chunk
                    .par_iter()
                    .fold(TextDetDataBatch::new, |mut acc, file| {
                        let filename = file.file_name().into_string().unwrap();
                        if TEXT_DET_FILE_NAME_FORMAT_REGEX
                            .captures(&filename)
                            .is_some()
                        {
                            match load_text_detection_image(base_dir, file.path(), dim) {
                                Ok(image_data) => {
                                    acc.polygons.push(image_data.polygons);
                                    acc.ignore_flags.push(image_data.ignore_flags);
                                    if acc.images.numel() == 1 {
                                        acc.images = image_data.image_tensor;
                                        acc.gts = image_data.gt_tensor;
                                        acc.masks = image_data.mask_tensor;
                                        acc.adjs_values = image_data.adjs_tensor;
                                    } else {
                                        acc.images =
                                            Tensor::cat(&[acc.images, image_data.image_tensor], 0);
                                        acc.gts = Tensor::cat(&[acc.gts, image_data.gt_tensor], 0);
                                        acc.masks =
                                            Tensor::cat(&[acc.masks, image_data.mask_tensor], 0);
                                        acc.adjs_values = Tensor::cat(
                                            &[acc.adjs_values, image_data.adjs_tensor],
                                            0,
                                        );
                                    }
                                }
                                Err(msg) => {
                                    error!("Error while loading single image data: {}", msg);
                                }
                            }
                        }
                        acc
                    })
                    .reduce(TextDetDataBatch::new, |mut acc, mut partial| {
                        if acc.images.numel() == 1 {
                            acc = partial;
                        } else if partial.images.numel() != 1 {
                            acc.polygons.append(&mut partial.polygons);
                            acc.ignore_flags.append(&mut partial.ignore_flags);
                            acc.images = Tensor::cat(&[acc.images, partial.images], 0);
                            acc.gts = Tensor::cat(&[acc.gts, partial.gts], 0);
                            acc.masks = Tensor::cat(&[acc.masks, partial.masks], 0);
                            acc.adjs_values =
                                Tensor::cat(&[acc.adjs_values, partial.adjs_values], 0);
                        }
                        acc
                    });

                save_batch(batch_data, target_dir.as_ref(), pos, train);
            });
        info!(
            "Successfully generated {} tensor files!",
            if train { "train" } else { "test" }
        );
        Ok(())
    } else {
        Err(anyhow!(
            "didn't find text detection images dir: {}",
            images_dir.display()
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

pub fn save_batch<T: AsRef<Path>>(
    batch: TextDetDataBatch,
    target_dir_path: T,
    idx: usize,
    is_train: bool,
) {
    let target_dir = target_dir_path.as_ref();
    let images_file;
    let gt_file;
    let mask_file;
    let adj_file;
    let polys_file;
    let ignore_flags_file;
    if is_train {
        images_file = get_target_filename(TEXT_DET_TRAIN_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TRAIN_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TRAIN_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TRAIN_ADJ_FILE);
        polys_file = get_target_filename(TEXT_DET_TRAIN_POLYS_FILE);
        ignore_flags_file = get_target_filename(TEXT_DET_TRAIN_IGNORE_FLAGS_FILE);
    } else {
        images_file = get_target_filename(TEXT_DET_TEST_IMAGES_FILE);
        gt_file = get_target_filename(TEXT_DET_TEST_GT_FILE);
        mask_file = get_target_filename(TEXT_DET_TEST_MASK_FILE);
        adj_file = get_target_filename(TEXT_DET_TEST_ADJ_FILE);
        polys_file = get_target_filename(TEXT_DET_TEST_POLYS_FILE);
        ignore_flags_file = get_target_filename(TEXT_DET_TEST_IGNORE_FLAGS_FILE);
    };
    if let Err(msg) = save_tensor(
        &batch.images,
        target_dir.join(format!("{}{}", images_file, idx)),
    ) {
        error!("Error while saving image tensor {}", msg);
    }
    if let Err(msg) = save_tensor(&batch.gts, target_dir.join(format!("{}{}", gt_file, idx))) {
        error!("Error while saving gt tensor {}", msg);
    }
    if let Err(msg) = save_tensor(
        &batch.masks,
        target_dir.join(format!("{}{}", mask_file, idx)),
    ) {
        error!("Error while saving mask tensor {}", msg);
    }
    if let Err(msg) = save_tensor(
        &batch.adjs_values,
        target_dir.join(format!("{}{}", adj_file, idx)),
    ) {
        error!("Error while saving adj tensor {}", msg);
    }
    if let Err(msg) = save_batch_polygons(
        &batch.polygons,
        target_dir.join(format!("{}{}", polys_file, idx)),
        true,
    ) {
        error!("Error while saving polygons vec {}", msg);
    }
    if let Err(msg) = save_vec_to_file(
        &batch.ignore_flags,
        target_dir.join(format!("{}{}", ignore_flags_file, idx)),
        true,
    ) {
        error!("Error while saving ignore flags vec {}", msg);
    }
}

fn save_vec_to_file<T: Serialize, H: AsRef<Path>>(
    value: &[T],
    file_path: H,
    overwrite: bool,
) -> Result<()> {
    let path = file_path.as_ref();
    if path.exists() {
        if overwrite {
            fs::remove_file(path)?;
        } else {
            return Err(anyhow!("file {} already exists", path.display()));
        }
    }
    serde_json::to_writer(File::create(path)?, value)?;

    Ok(())
}

fn save_batch_polygons<T: AsRef<Path>>(
    polygons: &[MultiPolygon<u32>],
    file_path: T,
    overwrite: bool,
) -> Result<()> {
    let path = file_path.as_ref();
    if path.exists() {
        if overwrite {
            fs::remove_file(path)?;
        } else {
            return Err(anyhow!("file {} already exists", path.display()));
        }
    }
    serde_json::to_writer(File::create(path)?, polygons)?;

    Ok(())
}

pub fn load_polygons_vec_from_file<T: AsRef<Path>>(file_path: T) -> Result<Vec<MultiPolygon<u32>>> {
    let path = file_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("file {} doesn't exists", path.display()));
    }
    let reader = BufReader::new(File::open(path)?);
    let value = serde_json::from_reader(reader)?;
    Ok(value)
}

pub fn load_ignore_flags_vec_from_file<T: AsRef<Path>>(file_path: T) -> Result<Vec<Vec<bool>>> {
    let path = file_path.as_ref();
    if !path.exists() {
        return Err(anyhow!("file {} doesn't exists", path.display()));
    }
    let file = File::open(path)?;
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
        generate_text_det_tensor_chunks("test_data/text_det", ".", true, (800, 800))?;
        generate_text_det_tensor_chunks("test_data/text_det", ".", false, (800, 800))?;

        // load expected images
        let train_img1 = open("test_data/preprocessed_img224.png")?.to_luma();
        let train_img2 = open("test_data/preprocessed_img55.png")?.to_luma();
        let train_gt_img1 = open("test_data/gt_img224.png")?.to_luma();
        let train_gt_img2 = open("test_data/gt_img55.png")?.to_luma();
        let train_mask_img1 = open("test_data/mask_img224.png")?.to_luma();
        let train_mask_img2 = open("test_data/mask_img55.png")?.to_luma();
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
        .map(|x| {
            let mut s = get_target_filename(x);
            s.push('0');
            s
        })
        .collect();

        let train_images_tensor = Tensor::load(&filenames[0])?;
        assert_eq!(train_images_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_images_tensor
                .get(0)
                .to_kind(Kind::Double)
                .to_string(80)?,
            convert_image_to_tensor(&train_img1)?.to_string(80)?
        );
        assert_eq!(
            train_images_tensor
                .get(1)
                .to_kind(Kind::Double)
                .to_string(80)?,
            convert_image_to_tensor(&train_img2)?.to_string(80)?
        );

        let train_gts_tensor = Tensor::load(&filenames[1])?;
        assert_eq!(train_gts_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_gts_tensor
                .get(0)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&train_gt_img1)? / 255.).to_string(80)?
        );
        assert_eq!(
            train_gts_tensor
                .get(1)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&train_gt_img2)? / 255.).to_string(80)?
        );

        let train_masks_tensor = Tensor::load(&filenames[2])?;
        assert_eq!(train_masks_tensor.size(), [2, 800, 800]);
        assert_eq!(
            train_masks_tensor
                .get(0)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&train_mask_img1)? / 255.).to_string(80)?
        );
        assert_eq!(
            train_masks_tensor
                .get(1)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&train_mask_img2)? / 255.).to_string(80)?
        );

        let train_adj_tensor = Tensor::load(&filenames[3])?;
        assert_eq!(train_adj_tensor.size(), [2, 2]);

        let mut original_dim = (180., 240.);
        let mut resized = (600., 800.);
        assert_eq!(
            train_adj_tensor.get(0).to_string(80)?,
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
                .to_string(80)?
        );

        original_dim = (300., 200.);
        resized = (800., 533.);
        assert_eq!(
            train_adj_tensor.get(1).to_string(80)?,
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
                .to_string(80)?
        );

        let train_polygons_vec = image_ops::load_polygons_vec_from_file(&filenames[4])?;
        assert_eq!(train_polygons_vec.len(), 2);
        assert_eq!(train_polygons_vec[0].0.len(), 3);
        assert_eq!(train_polygons_vec[1].0.len(), 4);

        let train_ignore_flags_vec = image_ops::load_ignore_flags_vec_from_file(&filenames[5])?;
        assert_eq!(train_ignore_flags_vec.len(), 2);
        assert_eq!(train_ignore_flags_vec[0].len(), 3);
        assert_eq!(train_ignore_flags_vec[1].len(), 4);

        let test_images_tensor = Tensor::load(&filenames[6])?;
        assert_eq!(test_images_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_images_tensor
                .get(0)
                .to_kind(Kind::Double)
                .to_string(80)?,
            convert_image_to_tensor(&test_img1)?.to_string(80)?
        );
        assert_eq!(
            test_images_tensor
                .get(1)
                .to_kind(Kind::Double)
                .to_string(80)?,
            convert_image_to_tensor(&test_img2)?.to_string(80)?
        );

        let test_gts_tensor = Tensor::load(&filenames[7])?;
        assert_eq!(test_gts_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_gts_tensor.get(0).to_kind(Kind::Double).to_string(80)?,
            (convert_image_to_tensor(&test_gt_img1)? / 255.).to_string(80)?
        );
        assert_eq!(
            test_gts_tensor.get(1).to_kind(Kind::Double).to_string(80)?,
            (convert_image_to_tensor(&test_gt_img2)? / 255.).to_string(80)?
        );

        let test_masks_tensor = Tensor::load(&filenames[8])?;
        assert_eq!(test_masks_tensor.size(), [2, 800, 800]);
        assert_eq!(
            test_masks_tensor
                .get(0)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&test_mask_img1)? / 255.).to_string(80)?
        );
        assert_eq!(
            test_masks_tensor
                .get(1)
                .to_kind(Kind::Double)
                .to_string(80)?,
            (convert_image_to_tensor(&test_mask_img2)? / 255.).to_string(80)?
        );

        let test_adj_tensor = Tensor::load(&filenames[9])?;
        assert_eq!(test_adj_tensor.size(), [2, 2]);

        let test_polygons_vec = image_ops::load_polygons_vec_from_file(&filenames[10])?;
        assert_eq!(test_polygons_vec.len(), 2);
        assert_eq!(test_polygons_vec[0].0.len(), 3);
        assert_eq!(test_polygons_vec[1].0.len(), 3);

        let test_ignore_flags_vec = image_ops::load_ignore_flags_vec_from_file(&filenames[11])?;
        assert_eq!(test_ignore_flags_vec.len(), 2);
        assert_eq!(test_ignore_flags_vec[0].len(), 3);
        assert_eq!(test_ignore_flags_vec[1].len(), 3);

        original_dim = (200., 200.);
        resized = (800., 800.);
        assert_eq!(
            test_adj_tensor.get(0).to_string(80)?,
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
                .to_string(80)?
        );

        original_dim = (184., 274.);
        resized = (537., 800.);
        assert_eq!(
            test_adj_tensor.get(1).to_string(80)?,
            Tensor::of_slice(&[resized.0 / original_dim.0, resized.1 / original_dim.1])
                .to_string(80)?
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

        let polys_test_vec = vec![
            // image 1
            MultiPolygon::from(vec![
                // poly 1
                Polygon::new(
                    LineString::from(vec![(12, 21), (22, 21), (22, 11), (12, 11)]),
                    vec![],
                ),
                // poly 2
                Polygon::new(
                    LineString::from(vec![(120, 210), (220, 210), (220, 110), (120, 110)]),
                    vec![],
                ),
            ]),
            // image 2
            MultiPolygon::from(vec![
                // poly 3
                Polygon::new(
                    LineString::from(vec![(34, 43), (44, 43), (44, 33), (34, 33)]),
                    vec![],
                ),
                // poly 4
                Polygon::new(
                    LineString::from(vec![(34, 43), (44, 43), (44, 33), (34, 33)]),
                    vec![],
                ),
            ]),
        ];
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
