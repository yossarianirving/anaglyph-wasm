use wasm_bindgen::prelude::*;
use anaglyph_rs::anaglyph::{AnaglyphType, left_right_to_anaglyph};
use image::{buffer::ConvertBuffer, imageops, DynamicImage, RgbaImage};

#[wasm_bindgen]
#[derive(Default)] 
pub struct Anaglyph {
    left_image: Option<RgbaImage>,
    right_image: Option<RgbaImage>,
    anaglyph_type: Option<AnaglyphType>
}

#[wasm_bindgen]
impl Anaglyph {
    pub fn new() -> Self {
        Self { ..Default::default() }
    }

    pub fn set_left_image(&mut self, width: u32, height: u32, left_image: Vec<u8>) -> bool {
        
        self.left_image =  Some(RgbaImage::from_raw(width, height, left_image).unwrap().convert() as RgbaImage) ;
        self.left_image.is_some()
    }

    pub fn set_right_image(&mut self, width: u32, height: u32, right_image: Vec<u8>) -> bool {
        self.right_image = Some(RgbaImage::from_raw(width, height, right_image).unwrap().convert() as RgbaImage);
        self.right_image.is_some()
    }

    pub fn set_stereoscopic_image(&mut self, width: u32, height: u32, stereo_image: Vec<u8>) -> bool {
        let stereoscopic_image = RgbaImage::from_raw(width, height, stereo_image);

        if width % 2 != 0 {
            return false;
        }

        self.left_image = Some(imageops::crop_imm(stereoscopic_image.as_ref().unwrap(), 0, 0, width / 2, height).to_image());

        self.right_image = Some(imageops::crop_imm(&stereoscopic_image.unwrap(), width / 2, 0, width, height).to_image());
        
        true
    }

    pub fn to_anaglyph(&self, anaglyph_type: Option<String>) -> Vec<u8> {

        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color
        };

        // refactor in future to reduce clone
        let anaglyph_image = left_right_to_anaglyph(&self.left_image.as_ref().unwrap(), &self.right_image.as_ref().unwrap(), a_type);

        anaglyph_image.to_vec()
    }

    pub fn get_width(&self) -> u32 {
        match &self.left_image {
            Some(i) => i.width(),
            None => 0
        }
    }

    pub fn get_height(&self) -> u32 {
        match &self.left_image {
            Some(i) => i.height(),
            None => 0
        }
    }

    
}
