use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::path::{Path, MAIN_SEPARATOR};
use tch::{nn::VarStore, Tensor};

pub const VALUES: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
pub const VALUES_COUNT: usize = VALUES.len();
pub const VALUES_COUNT_I64: i64 = VALUES_COUNT as i64;

lazy_static! {
    pub static ref POS_TO_CHAR: HashMap<usize, char> = {
        let mut m = HashMap::new();
        for (pos, ch) in VALUES.char_indices() {
            m.insert(pos, ch);
        }
        m
    };
    pub static ref VALUES_MAP: HashMap<char, u8> = {
        let mut m = HashMap::new();
        for (pos, ch) in VALUES.char_indices() {
            m.insert(ch, pos as u8);
        }
        m
    };
}

pub fn topk(tensor: &Tensor, k: i64) -> Vec<(char, f64)> {
    let tensor = match tensor.size().as_slice() {
        [VALUES_COUNT_I64] => tensor.shallow_clone(),
        [1, VALUES_COUNT_I64] => tensor.view((VALUES_COUNT_I64,)),
        [1, 1, VALUES_COUNT_I64] => tensor.view((VALUES_COUNT_I64,)),
        _ => panic!("unexpected tensor shape {:?}", tensor),
    };
    let (values, indexes) = tensor.topk(k, 0, true, true);
    let values = Vec::<f64>::from(values);
    let indexes = Vec::<i64>::from(indexes);
    values
        .iter()
        .zip(indexes.iter())
        .map(|(&value, &index)| (*POS_TO_CHAR.get(&(index as usize)).unwrap(), value))
        .collect()
}

pub fn save_tensor(tensor: &Tensor, path: &str) -> Result<()> {
    if let Some(pos) = path.rfind(MAIN_SEPARATOR) {
        let (dir, _) = path.split_at(pos);
        if !Path::new(dir).exists() {
            create_dir_all(dir)?;
        }
    }
    tensor.save(path)?;
    Ok(())
}

pub fn save_vs(vs: &VarStore, path: &str) -> Result<()> {
    if let Some(pos) = path.rfind(MAIN_SEPARATOR) {
        let (dir, _) = path.split_at(pos);
        if !Path::new(dir).exists() {
            create_dir_all(dir)?;
        }
    }
    vs.save(path)?;
    Ok(())
}

pub fn parse_number<T: std::str::FromStr>(num_str: &str, field: &str) -> Result<T> {
    match num_str.parse::<T>() {
        Ok(val) => Ok(val),
        Err(_msg) => Err(anyhow!("Could not parse {} value: {}", field, num_str)),
    }
}

pub fn parse_dimensions(dims_str: &str) -> Result<(u32, u32)> {
    let values = dims_str.split_terminator('x').collect::<Vec<&str>>();
    if values.len() == 2 {
        Ok((values[0].parse()?, values[1].parse()?))
    } else {
        Err(anyhow!("Could not parse dimensions value: {}", dims_str))
    }
}
