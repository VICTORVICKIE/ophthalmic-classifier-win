// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(dead_code, unused)]

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Write},
    mem,
    path::PathBuf,
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
    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Response -> String")
    }
}
fn log(file: &str, msg: &str) {
    let mut log_file = match OpenOptions::new().append(true).create(true).open(file) {
        Ok(f) => f,
        Err(_) => return,
    };

    writeln!(log_file, "{}", msg);
}

fn emit_prediction(window: &tauri::Window, event: &CommandEvent) {
    let log_dir = window.app_handle().path_resolver().app_log_dir();
    let log_dir = log_dir
        .as_ref()
        .and_then(|f| f.to_str())
        .expect("App Log Dir");
    println!("{}", log_dir);

    match event {
        CommandEvent::Stdout(res) => {
            println!("{}", res);
            window
                .emit("prediction", Some(res))
                .expect("failed to emit prediction event");
        }
        CommandEvent::Stderr(tf_log) => {
            println!("{}", tf_log);
        }
        CommandEvent::Error(e) => {
            println!("{}", e);
        }
        CommandEvent::Terminated(signal) => {
            println!("{:?}", signal);
        }
        _ => {}
    }
}

fn emit_failure(window: &tauri::Window, msg: &str) {
    window
        .emit("prediction", Some(Response::failure(msg).to_json()))
        .expect("failed to emit prediction event");
}

#[tauri::command]
fn predictor<'a>(app: tauri::AppHandle, model: &'a str, file: &'a str) {
    let window = app.get_window("main").expect("Main Window");
    let env = HashMap::from([("TF_CPP_MIN_LOG_LEVEL".to_owned(), "3".to_owned())]);

    let model_path = app
        .path_resolver()
        .resolve_resource("models")
        .and_then(|p| dunce::canonicalize(p).ok());

    let model_dir = model_path
        .as_ref()
        .and_then(|f| f.to_str())
        .expect("$RESOURCE/model");

    let (mut rx, _) = Command::new_sidecar("oct-tf")
        .expect("failed to create `oct-tf` binary command")
        .args(["-d", model_dir, "-n", model, "-i", file])
        // .envs(env)
        .spawn()
        .expect("Failed to spawn sidecar");

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            emit_prediction(&window, &event)
        }
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![predictor])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
