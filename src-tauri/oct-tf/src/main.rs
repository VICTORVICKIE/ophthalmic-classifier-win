use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use tensorflow::eager::{self, raw_ops, TensorHandle, ToTensorHandle};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

const CHANNEL_IDX: i32 = 2;
const GRAY: u64 = 1;
const SIZE: [i32; 2] = [256, 256];
const TOTAL_CLASSES: usize = 9;
const CLASSES: [&str; TOTAL_CLASSES] = [
    "ARMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "GC", "MH", "NORMAL",
];

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    dir: String,

    #[arg(short, long)]
    name: String,

    #[arg(short, long)]
    image: String,
}

#[derive(Debug, Copy, Clone)]
struct ModelDetail<'a> {
    name: &'a str,
    dir: &'a str,
    input: &'a str,
    output: &'a str,
}

impl Into<PathBuf> for ModelDetail<'_> {
    fn into(self) -> PathBuf {
        return Path::new(self.dir).join(self.name);
    }
}

#[derive(Debug)]
struct Model<'a> {
    bundle: &'a SavedModelBundle,
    graph: &'a Graph,
    detail: &'a ModelDetail<'a>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Prediction<'a> {
    pub model: &'a str,
    pub prediction: &'a str,
    pub probability: f32,
    pub classes: [&'a str; TOTAL_CLASSES],
    pub probabilities: [f32; TOTAL_CLASSES],
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Response<'a> {
    pub success: bool,
    #[serde(borrow)]
    pub result: Option<Prediction<'a>>,
    pub message: String,
}

impl<'a> Response<'a> {
    fn failure(msg: String) -> Self {
        return Self {
            success: false,
            result: None,
            message: msg,
        };
    }
}

fn load_img(ctx: &eager::Context, img_path: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
    let img_path = img_path.to_handle(&ctx)?;
    let buf = raw_ops::read_file(&ctx, &img_path)?;
    let img = raw_ops::decode_image(&ctx, &buf)?;
    let img = if img.dim(CHANNEL_IDX)? == GRAY {
        raw_ops::concat(&ctx, &CHANNEL_IDX, &[&img, &img, &img])?
    } else {
        img
    };
    let cast2float = raw_ops::Cast::new().DstT(tensorflow::DataType::Float);
    let img = cast2float.call(&ctx, &img)?;
    let img = raw_ops::div(&ctx, &img, &255f32)?;
    let batch = raw_ops::expand_dims(&ctx, &img, &0)?;
    let batch = raw_ops::resize_bicubic(&ctx, &batch, &SIZE)?;

    return Ok(unsafe { batch.resolve()?.into_tensor() });
}

fn predict<'a>(
    ctx: &'a eager::Context,
    model: &'a Model<'a>,
    input: &'a Tensor<f32>,
) -> Result<TensorHandle<'a>, Box<dyn Error>> {
    let signature = model
        .bundle
        .meta_graph_def()
        .get_signature("serving_default")?;

    let x_info = signature.get_input(model.detail.input)?;

    let op_x = model
        .graph
        .operation_by_name_required(&x_info.name().name)?;

    let output_info = signature.get_output(model.detail.output)?;

    let op_output = &model
        .graph
        .operation_by_name_required(&output_info.name().name)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&op_x, 0, input);
    let token_output = args.request_fetch(op_output, 0);
    let session = &model.bundle.session;
    session.run(&mut args)?;

    // Check the output.
    let fetch = args.fetch::<f32>(token_output)?;
    let output = fetch.into_handle(&ctx)?;
    let output = raw_ops::mul(&ctx, &output, &100f32)?;

    return Ok(output);
}

fn predictor<'a>(image_path: &'a str, model_name: &'a str, model_dir: &'a str) -> Response<'a> {
    let models = HashMap::from([
        (
            "VGG16",
            ModelDetail {
                name: "vgg",
                dir: model_dir,
                input: "vgg_input",
                output: "vgg_output",
            },
        ),
        (
            "CUSTOM",
            ModelDetail {
                name: "custom",
                dir: model_dir,
                input: "custom_input",
                output: "custom_output",
            },
        ),
    ]);

    let opts = eager::ContextOptions::new();
    let mut graph = Graph::new();
    let ctx = match eager::Context::new(opts) {
        Ok(ctx) => ctx,
        Err(e) => return Response::failure(format!("Failed to create tensorflow context: {}", e)),
    };

    let model_detail = match models.get(model_name) {
        Some(m) => m.to_owned(),
        None => {
            return Response::failure(format!("Failed to get tensorflow model: {}", model_name))
        }
    };

    let model_path: PathBuf = model_detail.into();

    let bundle =
        match SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_path) {
            Ok(b) => b,
            Err(e) => return Response::failure(format!("Failed to load tensorflow model: {}", e)),
        };

    let model = &Model {
        bundle: &bundle,
        graph: &graph,
        detail: &model_detail,
    };

    let x = match load_img(&ctx, image_path) {
        Ok(x) => x,
        Err(e) => return Response::failure(format!("Failed to load input image: {}", e)),
    };

    let output = match predict(&ctx, model, &x) {
        Ok(o) => o,
        Err(e) => return Response::failure(format!("Failed to predict input image: {}", e)),
    };

    let arg_max_tensor = match raw_ops::arg_max(&ctx, &output, &1) {
        Ok(a) => match a.resolve::<i64>() {
            Ok(a) => a,
            Err(e) => return Response::failure(format!("Failed to predict input image: {}", e)),
        },
        Err(e) => return Response::failure(format!("Failed to predict input image: {}", e)),
    };

    let arg_max = match arg_max_tensor.to_vec().get(0) {
        Some(a) => a.to_owned() as usize,
        None => return Response::failure(format!("Failed to predict input image: {}", 0)),
    };

    let probabilities: [f32; 9] = match output.resolve::<f32>() {
        Ok(p) => match p.to_vec().try_into() {
            Ok(p) => p,
            Err(e) => return Response::failure(format!("Failed to predict input image: {:?}", e)),
        },
        Err(e) => return Response::failure(format!("Failed to predict input image: {}", e)),
    };

    let prediction = Prediction {
        classes: CLASSES,
        model: model_name,
        prediction: CLASSES[arg_max],
        probability: probabilities[arg_max],
        probabilities,
    };

    return Response {
        success: true,
        result: Some(prediction),
        message: format!("Predicted: {}", CLASSES[arg_max]),
    };
}

fn main() {
    let args = Args::parse();
    let response = predictor(&args.image, &args.name, &args.dir);
    let stdout = serde_json::to_string(&response).expect("Response -> String");
    print!("{}", stdout);
}
