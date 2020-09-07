use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use tch::{kind, Device, IndexOp, TchError, Tensor};

#[derive(Debug)]
pub struct TextDetectionDataset {
  pub train_images: Tensor,
  pub train_gt: Tensor,
  pub train_mask: Tensor,
  pub test_images: Tensor,
  pub test_gt: Tensor,
  pub test_mask: Tensor,
}

impl TextDetectionDataset {
  pub fn train_iter(&self, batch_size: i64) -> TextDetectionIter {
    TextDetectionIter::new(
      &self.train_images,
      &self.train_gt,
      &self.train_mask,
      batch_size,
    )
  }

  pub fn test_iter(&self, batch_size: i64) -> TextDetectionIter {
    TextDetectionIter::new(
      &self.test_images,
      &self.test_gt,
      &self.test_mask,
      batch_size,
    )
  }
}

/// An iterator over a pair of tensors which have the same first dimension
/// size.
/// The typical use case is to iterate over batches. Each batch is a pair
/// containing a (potentially random) slice of each of the two input
/// tensors.
#[derive(Debug)]
pub struct TextDetectionIter {
  xs: Tensor,
  gts: Tensor,
  masks: Tensor,
  batch_index: i64,
  batch_size: i64,
  total_size: i64,
  device: Device,
  return_smaller_last_batch: bool,
}

impl TextDetectionIter {
  /// Returns a new iterator.
  ///
  /// This takes as input two tensors which first dimension must match. The
  /// returned iterator can be used to range over mini-batches of data of
  /// specified size.
  /// An error is returned if `xs` and `ys` have different first dimension
  /// sizes.
  ///
  /// # Arguments
  ///
  /// * `xs` - the features to be used by the model.
  /// * `ys` - the targets that the model attempts to predict.
  /// * `batch_size` - the size of batches to be returned.
  pub fn f_new(
    xs: &Tensor,
    gts: &Tensor,
    masks: &Tensor,
    batch_size: i64,
  ) -> Result<TextDetectionIter, TchError> {
    let total_size = xs.size()[0];
    if gts.size()[0] != total_size {
      return Err(TchError::Shape(format!(
        "different dimension for the gts inputs {:?} {:?}",
        xs, gts
      )));
    }
    if masks.size()[0] != total_size {
      return Err(TchError::Shape(format!(
        "different dimension for the masks inputs {:?} {:?}",
        xs, masks
      )));
    }
    Ok(TextDetectionIter {
      xs: xs.shallow_clone(),
      gts: gts.shallow_clone(),
      masks: masks.shallow_clone(),
      batch_index: 0,
      batch_size,
      total_size,
      device: Device::Cpu,
      return_smaller_last_batch: false,
    })
  }

  /// Returns a new iterator.
  ///
  /// This takes as input two tensors which first dimension must match. The
  /// returned iterator can be used to range over mini-batches of data of
  /// specified size.
  /// Panics if `xs` and `ys` have different first dimension sizes.
  ///
  /// # Arguments
  ///
  /// * `xs` - the features to be used by the model.
  /// * `ys` - the targets that the model attempts to predict.
  /// * `batch_size` - the size of batches to be returned.
  pub fn new(xs: &Tensor, gts: &Tensor, masks: &Tensor, batch_size: i64) -> TextDetectionIter {
    TextDetectionIter::f_new(xs, gts, masks, batch_size).unwrap()
  }

  /// Shuffles the dataset.
  ///
  /// The iterator would still run over the whole dataset but the order in
  /// which elements are grouped in mini-batches is randomized.
  pub fn shuffle(&mut self) -> &mut TextDetectionIter {
    let index = Tensor::randperm(self.total_size, kind::INT64_CPU);
    self.xs = self.xs.index_select(0, &index);
    self.gts = self.gts.index_select(0, &index);
    self.masks = self.masks.index_select(0, &index);
    self
  }

  /// Transfers the mini-batches to a specified device.
  #[allow(clippy::wrong_self_convention)]
  pub fn to_device(&mut self, device: Device) -> &mut TextDetectionIter {
    self.device = device;
    self
  }

  /// When set, returns the last batch even if smaller than the batch size.
  pub fn return_smaller_last_batch(&mut self) -> &mut TextDetectionIter {
    self.return_smaller_last_batch = true;
    self
  }
}

impl Iterator for TextDetectionIter {
  type Item = (Tensor, Tensor, Tensor);

  fn next(&mut self) -> Option<Self::Item> {
    let start = self.batch_index * self.batch_size;
    let size = std::cmp::min(self.batch_size, self.total_size - start);
    if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
      None
    } else {
      self.batch_index += 1;
      Some((
        self.xs.i(start..start + size).to_device(self.device),
        self.gts.i(start..start + size).to_device(self.device),
        self.masks.i(start..start + size).to_device(self.device),
      ))
    }
  }
}
