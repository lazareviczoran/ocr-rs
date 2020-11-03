use std::mem::drop;
use tch::{nn, nn::Conv2D, nn::ConvTranspose2D, nn::FuncT, nn::ModuleT, Tensor};

fn conv2d(p: nn::Path, c_in: i64, c_out: i64, ksize: i64, padding: i64, stride: i64) -> Conv2D {
    let conv2d_cfg = nn::ConvConfig {
        stride,
        padding,
        bias: false,
        ..Default::default()
    };
    nn::conv2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn conv_transpose2d(
    p: nn::Path,
    c_in: i64,
    c_out: i64,
    ksize: i64,
    padding: i64,
    stride: i64,
) -> ConvTranspose2D {
    let conv2d_cfg = nn::ConvTransposeConfig {
        stride,
        padding,
        ..Default::default()
    };
    nn::conv_transpose2d(&p, c_in, c_out, ksize, conv2d_cfg)
}

fn downsample(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    if stride != 1 || c_in != c_out {
        nn::seq_t()
            .add(conv2d(&p / "0", c_in, c_out, 1, 0, stride))
            .add(nn::batch_norm2d(&p / "1", c_out, Default::default()))
    } else {
        nn::seq_t()
    }
}

fn basic_block(p: nn::Path, c_in: i64, c_out: i64, stride: i64) -> impl ModuleT {
    let conv1 = conv2d(&p / "conv1", c_in, c_out, 3, 1, stride);
    let bn1 = nn::batch_norm2d(&p / "bn1", c_out, Default::default());
    let conv2 = conv2d(&p / "conv2", c_out, c_out, 3, 1, 1);
    let bn2 = nn::batch_norm2d(&p / "bn2", c_out, Default::default());
    let downsample = downsample(&p / "downsample", c_in, c_out, stride);
    nn::func_t(move |xs, train| {
        (xs.apply_t(&conv1, train)
            .apply_t(&bn1, train)
            .relu()
            .apply_t(&conv2, train)
            .apply_t(&bn2, train)
            + xs.apply_t(&downsample, train))
        .relu()
    })
}

fn basic_layer(p: nn::Path, c_in: i64, c_out: i64, stride: i64, cnt: i64) -> impl ModuleT {
    let mut layer = nn::seq_t().add(basic_block(&p / "0", c_in, c_out, stride));
    for block_index in 1..cnt {
        layer = layer.add(basic_block(&p / &block_index.to_string(), c_out, c_out, 1))
    }
    layer
}

fn resnet(p: &nn::Path, c1: i64, c2: i64, c3: i64, c4: i64) -> FuncT<'static> {
    let inner_in = 256;
    let inner_out = inner_in / 4;
    let conv1 = conv2d(p / "conv1", 1, 64, 7, 3, 2);
    let bn1 = nn::batch_norm2d(p / "bn1", 64, Default::default());
    let layer1 = basic_layer(p / "layer1", 64, 64, 1, c1);
    let layer2 = basic_layer(p / "layer2", 64, 128, 2, c2);
    let layer3 = basic_layer(p / "layer3", 128, 256, 2, c3);
    let layer4 = basic_layer(p / "layer4", 256, 512, 2, c4);

    let in5 = conv2d(p / "in5", 512, inner_in, 1, 0, 1);
    let in4 = conv2d(p / "in4", 256, inner_in, 1, 0, 1);
    let in3 = conv2d(p / "in3", 128, inner_in, 1, 0, 1);
    let in2 = conv2d(p / "in2", 64, inner_in, 1, 0, 1);

    let out5 = nn::seq()
        .add(conv2d(p / "out5", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 8, size[3] * 8], None, None)
        });
    let out4 = nn::seq()
        .add(conv2d(p / "out4", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 4, size[3] * 4], None, None)
        });
    let out3 = nn::seq()
        .add(conv2d(p / "out3", inner_in, inner_out, 3, 1, 1))
        .add_fn(move |xs| {
            let size = xs.size();
            xs.upsample_nearest2d(&[size[2] * 2, size[3] * 2], None, None)
        });
    let out2 = conv2d(p / "out2", inner_in, inner_out, 3, 1, 1);

    // binarization
    let bin_conv1 = conv2d(p / "bin_conv1", inner_in, inner_out, 3, 1, 1);
    let bin_bn1 = nn::batch_norm2d(p / "bin_bn1", inner_out, Default::default());
    let bin_conv_tr1 = conv_transpose2d(p / "bin_conv_tr1", inner_out, inner_out, 2, 0, 2);
    let bin_bn2 = nn::batch_norm2d(p / "bin_bn2", inner_out, Default::default());
    let bin_conv_tr2 = conv_transpose2d(p / "bin_conv_tr2", inner_out, 1, 2, 0, 2);

    nn::func_t(move |xs, train| {
        let x1 = xs
            .apply_t(&conv1, train)
            .apply_t(&bn1, train)
            .relu()
            .max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[1, 1], false)
            .apply_t(&layer1, train);
        let x2 = x1.apply_t(&layer2, train);
        let x_in2 = x1.apply_t(&in2, train);
        drop(x1);
        let x3 = x2.apply_t(&layer3, train);
        let x_in3 = x2.apply_t(&in3, train);
        drop(x2);
        let x4 = x3.apply_t(&layer4, train);
        let x_in4 = &x3.apply_t(&in4, train);
        drop(x3);
        let x_in5 = x4.apply_t(&in5, train);
        drop(x4);

        let x_in3_size = x_in3.size();
        let p2 = (x_in3.upsample_nearest2d(&[x_in3_size[2] * 2, x_in3_size[3] * 2], None, None)
            + x_in2)
            .apply_t(&out2, train);
        let x_in4_size = x_in4.size();
        let p3 = (x_in4.upsample_nearest2d(&[x_in4_size[2] * 2, x_in4_size[3] * 2], None, None)
            + x_in3)
            .apply_t(&out3, train);
        let x_in5_size = x_in5.size();
        let p4 = (x_in5.upsample_nearest2d(&[x_in5_size[2] * 2, x_in5_size[3] * 2], None, None)
            + x_in4)
            .apply_t(&out4, train);
        let p5 = x_in5.apply_t(&out5, train);

        let fuse = Tensor::cat(&[p5, p4, p3, p2], 1);

        // binarization
        fuse.apply_t(&bin_conv1, train)
            .apply_t(&bin_bn1, train)
            .relu()
            .apply_t(&bin_conv_tr1, train)
            .apply_t(&bin_bn2, train)
            .relu()
            .apply_t(&bin_conv_tr2, train)
            .sigmoid()
    })
}

pub fn resnet18(p: &nn::Path) -> FuncT<'static> {
    resnet(p, 2, 2, 2, 2)
}
