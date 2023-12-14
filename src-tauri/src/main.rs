// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(dead_code, unused)]
use chrono::Utc;
use chrono_tz::Asia::Kolkata;

use std::{
    collections::HashMap,
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tauri::{
    api::process::{Command, CommandEvent},
    Manager,
};

const TOTAL_CLASSES: usize = 9;

#[derive(Debug, Default, Serialize, Deserialize, Copy, Clone)]
pub struct Prediction<'a> {
    pub model: &'a str,
    pub prediction: &'a str,
    pub probability: f32,
    pub classes: [&'a str; TOTAL_CLASSES],
    pub probabilities: [f32; TOTAL_CLASSES],
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy)]
pub struct Response<'a> {
    pub success: bool,
    pub result: Option<Prediction<'a>>,
    pub message: &'a str,
}

impl<'a> Response<'a> {
    fn failure(msg: &'a str) -> Self {
        return Self {
            success: false,
            result: None,
            message: msg,
        };
    }
}

fn emit_prediction(app: &tauri::AppHandle, event: &CommandEvent) {
    let window = app
        .get_window("main")
        .ok_or_else(|| log(&app, &format!("Unable tot get main window")))
        .unwrap();
    match event {
        CommandEvent::Stdout(res) => {
            log(app, &format!("[RESPONSE] {}", res.trim()));
            window
                .emit("prediction", Some(res))
                .map_err(|err| {
                    log(
                        &app,
                        &format!("[ERROR] Failed to emit prediction event: {}", err),
                    )
                })
                .unwrap();
        }
        CommandEvent::Stderr(tf_log) => log(app, &format!("[TF LOG] {}", tf_log.trim())),
        CommandEvent::Error(e) => log(app, &format!("[ERROR] {}\n{}", e.trim(), "-".repeat(100))),
        CommandEvent::Terminated(signal) => log(
            app,
            &format!("[TERMINATED] {:?}\n{}", signal, "-".repeat(100)),
        ),
        _ => log(
            app,
            &format!("[ERROR] Unknown Event Encountered\n{}", "-".repeat(100)),
        ),
    }
}

#[derive(Debug, Clone)]
struct Resource(Arc<str>);

impl Resource {
    fn new(app: &tauri::AppHandle, p: &str) -> Result<Self, Box<dyn Error>> {
        let path = app
            .path_resolver()
            .resource_dir()
            .and_then(|p| dunce::canonicalize(p).ok());

        let path = match path.as_ref().and_then(|f| f.to_str()) {
            Some(p) => p,
            None => return Err(format!("[ERROR] Unable to resolve path: {}", p).into()),
        };

        let path: PathBuf = [path, p].iter().collect();
        let path = match path.to_str() {
            Some(p) => p,
            None => return Err(format!("[ERROR] Unable to resolve path: {}", p).into()),
        };
        return Ok(Self(Arc::from(path)));
    }

    fn path(&self) -> &str {
        &self.0
    }

    fn join(&mut self, p: &str) -> Result<Self, Box<dyn Error>> {
        let path: PathBuf = [self.path(), p].iter().collect();
        let path = match path.to_str() {
            Some(p) => p,
            None => return Err(format!("[ERROR] Unable to join path: {}", p).into()),
        };
        return Ok(Self(Arc::from(path)));
    }
}

fn log(app: &tauri::AppHandle, msg: &str) {
    let mut log_dir = match Resource::new(&app, "logs") {
        Ok(l) => l,
        Err(_) => return,
    };
    if !Path::new(&log_dir.path()).exists() {
        fs::create_dir(log_dir.path());
    }

    let ist = Utc::now().with_timezone(&Kolkata);

    let date = &ist.format("ophthalmic.classifier.%Y%m%d.log").to_string();
    let time = &ist.format("[%Y-%m-%d %H:%M:%S]").to_string();

    let log = match log_dir.join(date) {
        Ok(l) => l,
        Err(_) => return,
    };

    let mut f = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(log.path())
    {
        Ok(f) => f,
        Err(_) => return,
    };

    match writeln!(f, "{} {}", time, msg) {
        Ok(_) => return,
        Err(_) => return,
    }
}

#[tauri::command]
fn predictor<'a>(app: tauri::AppHandle, model: &'a str, file: &'a str) {
    let window = app
        .get_window("main")
        .ok_or_else(|| log(&app, &format!("[ERROR] Unable tot get main window")))
        .unwrap();
    let model_dir = Resource::new(&app, "models")
        .map_err(|err| {
            log(
                &app,
                &format!(
                    "[ERROR] Unable to resolve models directory/folder!: {}",
                    err
                ),
            )
        })
        .unwrap();

    let (mut rx, _) = Command::new_sidecar("oct-tf")
        .map_err(|err| {
            log(
                &app,
                &format!("[ERROR] Failed to create `oct-tf` binary command: {}", err),
            )
        })
        .unwrap()
        .args(["-d", model_dir.path(), "-n", model, "-i", file])
        .spawn()
        .map_err(|err| log(&app, &format!("[ERROR] Failed to spawn sidecar: {}", err)))
        .unwrap();

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            emit_prediction(&app, &event)
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![predictor])
        .run(tauri::generate_context!())
        .expect("Error while running tauri application");
}
