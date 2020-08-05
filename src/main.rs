extern crate rand;
#[macro_use]
extern crate lazy_static;
extern crate coaster as co;
extern crate juice;

use magick_rust::bindings::ColorspaceType_GRAYColorspace;
use magick_rust::{magick_wand_genesis, MagickWand};

use co::frameworks::native::get_native_backend;
// use co::frameworks::cuda::get_cuda_backend;
use co::prelude::*;
use juice::layer::*;
use juice::layers::*;
use juice::solver::*;
use juice::util::*;

use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::io::prelude::*;
use std::rc::Rc;
use std::sync::{Arc, Once, RwLock};
use std::time::Instant;

fn main() -> Result<(), &'static str> {
    let load_instant = Instant::now();
    let (pixels, labels, rows, cols) = load_values("images_data.txt")?;
    println!(
        "Time to load data: {}",
        load_instant.elapsed().as_secs_f64()
    );

    let training_instant = Instant::now();
    println!("Training...");

    create_and_train_nn(pixels, labels, rows, cols);

    println!(
        "Time used for training: {}",
        training_instant.elapsed().as_secs_f64()
    );

    // // Test set
    // let (test_pixels_mat, test_labels, test_rows, test_cols) = load_images("images/test")?;
    // let res = model.predict(&test_pixels_mat).unwrap();

    // let test_instant = Instant::now();
    // println!("Evaluation...");
    // let mut hits = 0;
    // let mut misses = 0;
    // // Evaluation
    // // println!("Got\tExpected");
    // let mut idx = 0;
    // while idx < res.data().len() / VALUES_COUNT {
    //     let pos = idx * VALUES_COUNT;
    //     let data_slice = res.data().get(pos..pos + VALUES_COUNT).unwrap();
    //     let prediction = data_slice
    //         .iter()
    //         .enumerate()
    //         .fold((0, 0.), |mut acc, (pos, x)| {
    //             if *x > acc.1 {
    //                 acc = (pos, *x);
    //             }
    //             acc
    //         });
    //     // println!("{:.2}\t{}", prediction.0, test_labels.data()[idx]);
    //     if prediction.0 == test_labels.data()[idx] as usize {
    //         hits += 1;
    //     } else {
    //         misses += 1;
    //     }
    //     idx += 1;
    // }

    // println!(
    //     "Time used for test: {}",
    //     test_instant.elapsed().as_secs_f64()
    // );

    // println!("Hits: {}, Misses: {}", hits, misses);
    // let hits_f = hits as f64;
    // let total = (hits + misses) as f64;
    // println!("Accuracy: {}%", (hits_f / total) * 100.);

    Ok(())
}

fn create_and_train_nn(
    trn_pixels: Vec<u8>,
    trn_labels: Vec<u8>,
    example_count: usize,
    pixel_count: usize,
) {
    let pixel_dim = 28;

    let mut decoded_images = trn_pixels
        .chunks(pixel_count)
        .enumerate()
        .map(|(ind, pixels)| (trn_labels[ind], pixels.to_vec()));

    let batch_size = 30;
    let learning_rate = 0.03f32;
    let momentum = 0f32;

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[batch_size, pixel_dim, pixel_dim]);
    net_cfg.force_backward = true;

    // net_cfg = add_conv_net(net_cfg, batch_size, pixel_dim);
    net_cfg = add_mlp(net_cfg, batch_size, pixel_count);

    net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

    let mut classifier_cfg = SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[batch_size, VALUES_COUNT]);
    classifier_cfg.add_input("label", &[batch_size, 1]);
    // set up nll loss
    let nll_layer_cfg = NegativeLogLikelihoodConfig {
        num_classes: VALUES_COUNT,
    };
    let nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    // set up backends
    // let backend = Rc::new(get_native_backend());
    let backend = Rc::new(get_native_backend());

    // set up solver
    let mut solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum,
        ..SolverConfig::default()
    };
    solver_cfg.network = LayerConfig::new("network", net_cfg);
    solver_cfg.objective = LayerConfig::new("classifier", classifier_cfg);
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    // set up confusion matrix
    let mut classification_evaluator = ::juice::solver::ConfusionMatrix::new(VALUES_COUNT);
    classification_evaluator.set_capacity(Some(1000));

    let input = SharedTensor::<f32>::new(&[batch_size, pixel_dim, pixel_dim]);
    let inp_lock = Arc::new(RwLock::new(input));

    let label = SharedTensor::<f32>::new(&[batch_size, 1]);
    let label_lock = Arc::new(RwLock::new(label));

    for _ in 0..(example_count / batch_size as usize) {
        // write input
        let mut targets = Vec::new();

        for (batch_n, (label_val, ref input)) in
            decoded_images.by_ref().take(batch_size).enumerate()
        {
            let mut input_tensor = inp_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();
            write_batch_sample(&mut input_tensor, &input, batch_n);
            write_batch_sample(&mut label_tensor, &[label_val], batch_n);
            targets.push(label_val as usize);
        }
        // train the network!
        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = classification_evaluator.get_predictions(&mut infered);
        println!(
            "predictions\n{:?}",
            predictions
                .iter()
                .map(|v| *POS_TO_CHAR.get(v).unwrap())
                .collect::<Vec<char>>()
        );
        println!(
            "expected\n{:?}",
            targets
                .iter()
                .map(|v| *POS_TO_CHAR.get(v).unwrap())
                .collect::<Vec<char>>()
        );

        classification_evaluator.add_samples(&predictions, &targets);
        println!(
            "Last sample: {} | Accuracy {}",
            classification_evaluator.samples().iter().last().unwrap(),
            classification_evaluator.accuracy()
        );
    }
}

fn add_conv_net(
    mut net_cfg: SequentialConfig,
    batch_size: usize,
    pixel_dim: usize,
) -> SequentialConfig {
    net_cfg.add_layer(LayerConfig::new(
        "reshape",
        ReshapeConfig::of_shape(&[batch_size, 1, pixel_dim, pixel_dim]),
    ));
    net_cfg.add_layer(LayerConfig::new(
        "conv",
        ConvolutionConfig {
            num_output: 100,
            filter_shape: vec![5],
            padding: vec![0],
            stride: vec![1],
        },
    ));
    net_cfg.add_layer(LayerConfig::new(
        "pooling",
        PoolingConfig {
            mode: PoolingMode::Max,
            filter_shape: vec![2],
            padding: vec![0],
            stride: vec![2],
        },
    ));
    net_cfg.add_layer(LayerConfig::new(
        "linear1",
        LinearConfig { output_size: 500 },
    ));
    net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
    net_cfg.add_layer(LayerConfig::new(
        "linear2",
        LinearConfig {
            output_size: VALUES_COUNT,
        },
    ));
    net_cfg
}

fn add_mlp(
    mut net_cfg: SequentialConfig,
    batch_size: usize,
    pixel_count: usize,
) -> SequentialConfig {
    net_cfg.add_layer(LayerConfig::new(
        "reshape",
        LayerType::Reshape(ReshapeConfig::of_shape(&[batch_size, pixel_count])),
    ));
    net_cfg.add_layer(LayerConfig::new(
        "linear1",
        LayerType::Linear(LinearConfig { output_size: 500 }),
    ));
    net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
    net_cfg.add_layer(LayerConfig::new(
        "linear1",
        LayerType::Linear(LinearConfig { output_size: 250 }),
    ));
    net_cfg.add_layer(LayerConfig::new("sigmoid2", LayerType::Sigmoid));
    net_cfg.add_layer(LayerConfig::new(
        "linear3",
        LayerType::Linear(LinearConfig {
            output_size: VALUES_COUNT,
        }),
    ));
    net_cfg
}

static START: Once = Once::new();

static VALUES: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
static VALUES_COUNT: usize = VALUES.len();
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
}

fn get_image_pixel_colors_grayscale(file_path: &str) -> Result<Vec<u8>, &'static str> {
    START.call_once(|| {
        magick_wand_genesis();
    });
    let mut pixels = Vec::new();
    let wand = MagickWand::new();
    wand.read_image(file_path)?;
    wand.transform_image_colorspace(ColorspaceType_GRAYColorspace)?;
    let cols = wand.get_image_width();
    let rows = wand.get_image_height();
    for i in 0..rows {
        for j in 0..cols {
            // convert to grayscale and store value
            if let Some(pixel_info) = wand.get_image_pixel_color(i as isize, j as isize) {
                let color_values = pixel_info.get_color_as_string()?;
                let values_vec: Vec<&str> = color_values
                    .get(5..color_values.len() - 1)
                    .unwrap()
                    .split_terminator(',')
                    .collect();
                pixels.push(values_vec[0].parse::<u8>().unwrap());
            } else {
                // let msg = format!("Couldn't get data for pixel ({}:{}) in {}", i, j, file_path);
                return Err("Couldn't get data for pixel");
            }
        }
    }
    Ok(pixels)
}

fn load_images(dir_path: &str) -> Result<(Vec<u8>, Vec<u8>, usize, usize), &'static str> {
    let files = fs::read_dir(dir_path).unwrap();
    let re = Regex::new(r"(.*)-(upper|lower|num)-([a-zA-z0-9])-img\.png").unwrap();
    let mut pixels = Vec::new();
    let mut labels = Vec::new();
    let mut cols = 0;
    let mut rows = 0;
    for file in files {
        let file_info = file.unwrap();
        let filename = file_info.file_name();
        if let Some(caps) = re.captures(filename.to_str().unwrap()) {
            rows += 1;
            let _font = String::from(&caps[1]);
            let _letter_type = String::from(&caps[2]);
            let letter = caps[3].chars().next().unwrap();
            labels.push(*VALUES_MAP.get(&letter).unwrap());
            let mut image_pixels =
                get_image_pixel_colors_grayscale(&file_info.path().display().to_string())?;
            cols = image_pixels.len();
            pixels.append(&mut image_pixels);
        }
    }

    Ok((pixels, labels, rows, cols))
}

fn load_values(file_path: &str) -> Result<(Vec<u8>, Vec<u8>, usize, usize), &'static str> {
    let file_wrapped = fs::File::open(file_path);
    let values;
    if let Ok(mut file) = file_wrapped {
        values = load_values_from_file(&mut file)?;
    } else {
        values = load_images("images/training")?;
        save_values_to_file(file_path, &values.0, &values.1, values.2, values.3)?;
    }
    Ok(values)
}

fn load_values_from_file(
    file: &mut fs::File,
) -> Result<(Vec<u8>, Vec<u8>, usize, usize), &'static str> {
    let mut content = String::new();
    let read_res = file.read_to_string(&mut content);
    if read_res.is_err() {
        return Err("failed to read file with values");
    }
    // matrix values are saved in format
    //
    // rows_count,cols_count
    // val11,val12,...,val1{cols}
    // val21,val22,...,val2{cols}
    // ...
    // val{rows}1,val{rows}2,...val{rows}{cols}
    // label1
    // label2
    // ...
    // label{rows}
    let split_values: Vec<&str> = content.split_terminator('\n').collect();
    let dimensions: Vec<usize> = split_values[0]
        .split_terminator(',')
        .map(|x| x.parse::<usize>().unwrap())
        .collect();
    let rows = dimensions[0];
    let cols = dimensions[1];
    let mut label_values = Vec::with_capacity(VALUES_COUNT * rows);
    let mut image_values = Vec::with_capacity(rows * cols);
    for i in 1..=rows {
        let mut values: Vec<u8> = split_values[i]
            .split_terminator(',')
            .map(|x| x.parse::<u8>().unwrap())
            .collect();
        image_values.append(&mut values);
        label_values.push(split_values[i + rows].parse::<u8>().unwrap());
    }

    Ok((image_values, label_values, rows, cols))
}

fn save_values_to_file(
    file_path: &str,
    pixels: &Vec<u8>,
    labels: &Vec<u8>,
    rows: usize,
    cols: usize,
) -> Result<(), &'static str> {
    let mut content = String::new();
    let w_file_wrapped = fs::File::create(file_path);
    if let Ok(mut w_file) = w_file_wrapped {
        content.push_str(&format!("{},{}\n", rows, cols));
        let mut labels_content = String::new();
        let mut i = 0;
        while i < rows {
            let curr_start = i * cols;
            let row_string = format!("{:?}", pixels.get(curr_start..curr_start + cols).unwrap());
            let row_str = row_string.get(1..row_string.len() - 1).unwrap();
            content.push_str(&row_str.replace(" ", ""));
            content.push('\n');
            labels_content.push_str(&format!("{}\n", labels[i]));
            i += 1;
        }
        content.push_str(&labels_content);
        let write_res = w_file.write(content.as_bytes());
        if write_res.is_err() {
            return Err("Failed to save matrix values to file");
        }
    }
    Ok(())
}
