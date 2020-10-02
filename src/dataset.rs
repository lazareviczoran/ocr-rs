#[derive(Debug)]
pub struct TextDetectionDataset {
  pub train_images: Vec<String>,
  pub train_gt: Vec<String>,
  pub train_mask: Vec<String>,
  pub train_adj: Vec<String>,
  pub train_polys: Vec<String>,
  pub train_ignore_flags: Vec<String>,
  pub test_images: Vec<String>,
  pub test_gt: Vec<String>,
  pub test_mask: Vec<String>,
  pub test_adj: Vec<String>,
  pub test_polys: Vec<String>,
  pub test_ignore_flags: Vec<String>,
}
