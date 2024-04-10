use wasm_bindgen::prelude::*;
use anaglyph_rs::anaglyph::{left_right_to_anaglyph_offset, AnaglyphType, Offset};
use image::{buffer::ConvertBuffer, imageops, RgbImage, RgbaImage};
extern crate web_sys;
use web_sys::console;
use std::io::Cursor;
use image::io::Reader as ImageReader;

pub struct Timer<'a> {
    name: &'a str,
}

impl<'a> Timer<'a> {
    pub fn new(name: &'a str) -> Timer<'a> {
        console::time_with_label(name);
        Timer { name }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        console::time_end_with_label(self.name);
    }
}

#[wasm_bindgen]
#[derive(Default)] 
pub struct Anaglyph {
    left_image: RgbImage,
    right_image: RgbImage,
    anaglyph_type: Option<AnaglyphType>
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct AnaglyphResult {
    image_vec: Vec<u8>,
    pub height: u32,
    pub width: u32
}
#[wasm_bindgen]
impl Anaglyph {
    pub fn new() -> Self {
        Self { ..Default::default() }
    }

    pub fn set_left_image(&mut self, width: u32, height: u32, left_image: Vec<u8>) -> bool {
        
        self.left_image =  RgbImage::from_raw(width, height, left_image).unwrap() as RgbImage;
        true
    }

    pub fn set_right_image(&mut self, width: u32, height: u32, right_image: Vec<u8>) -> bool {
        self.right_image = RgbImage::from_raw(width, height, right_image).unwrap().convert();
        true
    }

    pub fn set_right_image_raw(&mut self, right_image: Vec<u8>) {
        let _timer = Timer::new("Load raw image");
        self.right_image = ImageReader::new(Cursor::new(right_image)).with_guessed_format().unwrap().decode().unwrap().to_rgb8();
    }

    pub fn set_left_image_raw(&mut self, left_image: Vec<u8>) {
        let _timer = Timer::new("Load raw image");
        self.left_image = ImageReader::new(Cursor::new(left_image)).with_guessed_format().unwrap().decode().unwrap().to_rgb8();
    }

    pub fn set_stereoscopic_image(&mut self, width: u32, height: u32, stereo_image: Vec<u8>) -> bool {
        let stereoscopic_image = RgbImage::from_raw(width, height, stereo_image);

        if width % 2 != 0 {
            return false;
        }

        self.left_image = imageops::crop_imm(stereoscopic_image.as_ref().unwrap(), 0, 0, width / 2, height).to_image();

        self.right_image = imageops::crop_imm(&stereoscopic_image.unwrap(), width / 2, 0, width, height).to_image();
        
        true
    }

    pub fn to_anaglyph(&self, anaglyph_type: Option<String>, offset_x: i32, offset_y: i32) -> AnaglyphResult {

        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color
        };
        let _timer = Timer::new("To Anaglyph");

        let offset: Offset = Offset {
            x: offset_x,
            y: offset_y
        };

        let anaglyph_image = left_right_to_anaglyph_offset(&self.left_image, &self.right_image, a_type, offset);

        AnaglyphResult {
            image_vec: (anaglyph_image.convert() as RgbaImage).into_vec(),
            height: anaglyph_image.height(),
            width: anaglyph_image.width()
        }
        
    }

    pub fn get_width(&self) -> u32 {
        self.left_image.width()
    }

    pub fn get_height(&self) -> u32 {
        self.left_image.height()
    }

    
}

#[wasm_bindgen]
impl AnaglyphResult {
    pub fn get_image(&self) -> Vec<u8> {
        self.image_vec.clone()
    }
}
