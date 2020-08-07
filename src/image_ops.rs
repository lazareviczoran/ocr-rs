use magick_rust::bindings::ColorspaceType_GRAYColorspace;
use magick_rust::{magick_wand_genesis, MagickWand};

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Once;

use anyhow::{anyhow, Result};
use regex::Regex;
use tch::vision::dataset::Dataset;
use tch::{Kind, Tensor};

const TRAIN_IMAGES_FILE: &str = "training_images_data";
const TRAIN_LABELS_FILE: &str = "training_labels_data";
const TEST_IMAGES_FILE: &str = "test_images_data";
const TEST_LABELS_FILE: &str = "test_labels_data";

static START: Once = Once::new();

pub const VALUES: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
pub const VALUES_COUNT: usize = VALUES.len();
lazy_static! {
    static ref POS_TO_CHAR: HashMap<usize, char> = {
        let mut m = HashMap::new();
        for (pos, ch) in VALUES.char_indices() {
            m.insert(pos, ch);
        }
        m
    };
    static ref VALUES_MAP: HashMap<char, u8> = {
        let mut m = HashMap::new();
        for (pos, ch) in VALUES.char_indices() {
            m.insert(ch, pos as u8);
        }
        m
    };
    static ref FILE_NAME_FORMAT_REGEX: Regex =
        Regex::new(r"(.*)-(upper|lower|num)-([a-zA-z0-9])-img\.png").unwrap();
}

pub fn load_values() -> Result<Dataset> {
    let train_images = load_values_from_file(TRAIN_IMAGES_FILE)?;
    let train_labels = load_values_from_file(TRAIN_LABELS_FILE)?;
    let test_images = load_values_from_file(TEST_IMAGES_FILE)?;
    let test_labels = load_values_from_file(TEST_LABELS_FILE)?;

    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: VALUES_COUNT as i64,
    })
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
    let files = fs::read_dir(dir_path).unwrap();
    let mut pixels = Vec::new();
    let mut cols = 0;
    let mut rows = 0;
    for file in files {
        let file_info = file.unwrap();
        let filename = file_info.file_name();
        if FILE_NAME_FORMAT_REGEX
            .captures(filename.to_str().unwrap())
            .is_some()
        {
            rows += 1;
            let mut image_pixels =
                get_image_pixel_colors_grayscale(&file_info.path().display().to_string())?;
            cols = image_pixels.len();
            pixels.append(&mut image_pixels);
        }
    }
    println!("Creating tensor for images r-{} c-{}", rows, cols);
    let images_tensor = Tensor::of_slice(&pixels)
        .view((rows as i64, cols as i64))
        .to_kind(Kind::Float)
        / 255.;
    Ok(images_tensor)
}

fn load_labels(dir_path: &str) -> Result<Tensor> {
    let files = fs::read_dir(dir_path).unwrap();
    let mut labels = Vec::new();
    let mut rows = 0;
    for file in files {
        let file_info = file.unwrap();
        let filename = file_info.file_name();
        if let Some(caps) = FILE_NAME_FORMAT_REGEX.captures(filename.to_str().unwrap()) {
            rows += 1;
            let _font = String::from(&caps[1]);
            let _letter_type = String::from(&caps[2]);
            let letter = caps[3].chars().next().unwrap();
            labels.push(*VALUES_MAP.get(&letter).unwrap());
        }
    }
    println!("Creating tensor for labels r-{} c-1", rows);
    let labels_tensor = Tensor::of_slice(&labels).to_kind(Kind::Int64);
    Ok(labels_tensor)
}

fn get_image_pixel_colors_grayscale(file_path: &str) -> Result<Vec<u8>> {
    START.call_once(|| {
        magick_wand_genesis();
    });
    let mut pixels = Vec::new();
    let wand = MagickWand::new();
    wand.read_image(file_path)
        .map_err(|err| anyhow!("Failed to read image {}: {}", file_path, err))?;
    wand.transform_image_colorspace(ColorspaceType_GRAYColorspace)
        .map_err(|err| anyhow!("Failed to convert to gray in image {}: {}", file_path, err))?;
    let cols = wand.get_image_width();
    let rows = wand.get_image_height();
    for i in 0..rows {
        for j in 0..cols {
            // convert to grayscale and store value
            if let Some(pixel_info) = wand.get_image_pixel_color(i as isize, j as isize) {
                let color_values = pixel_info.get_color_as_string().map_err(|err| {
                    anyhow!(
                        "Failed to read pixel {:?} color in file {}: {}",
                        (i, j),
                        file_path,
                        err
                    )
                })?;
                let values_vec: Vec<&str> = color_values
                    .get(5..color_values.len() - 1)
                    .unwrap()
                    .split_terminator(',')
                    .collect();
                pixels.push(values_vec[0].parse::<u8>().unwrap());
            } else {
                return Err(anyhow!(
                    "Couldn't get data for pixel ({}:{}) in {}",
                    i,
                    j,
                    file_path
                ));
            }
        }
    }
    Ok(pixels)
}
