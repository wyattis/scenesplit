// Hide the console window in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analysis;
mod app;
mod export;
mod ffmpeg;
mod player;
mod scenes;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Scene Split")
            .with_inner_size([1000.0, 760.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("Scene Split", options, Box::new(|cc| Ok(Box::new(app::App::new(cc)))))
}
