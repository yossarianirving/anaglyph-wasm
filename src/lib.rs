use anaglyph_rs::anaglyph::{
    left_right_to_anaglyph_offset, AnaglyphType, BlenderChoice, Offset, VideoDirection,
};
use anaglyph_rs::gif::convert_gif_buffer_to_anaglyph;
#[cfg(target_arch = "wasm32")]
use anaglyph_rs::gif::convert_gif_buffer_to_anaglyph_async;
use image::{buffer::ConvertBuffer, imageops, RgbImage, RgbaImage};
use wasm_bindgen::prelude::*;
extern crate web_sys;
use image::ImageReader;
use std::io::Cursor;
use web_sys::console;

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
    gif: Vec<u8>,
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct AnaglyphResult {
    image_vec: Vec<u8>,
    pub height: u32,
    pub width: u32,
}
#[wasm_bindgen]
impl Anaglyph {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn set_left_image(&mut self, width: u32, height: u32, left_image: Vec<u8>) -> bool {
        self.left_image = RgbImage::from_raw(width, height, left_image).unwrap() as RgbImage;
        true
    }

    pub fn set_right_image(&mut self, width: u32, height: u32, right_image: Vec<u8>) -> bool {
        self.right_image = RgbImage::from_raw(width, height, right_image)
            .unwrap()
            .convert();
        true
    }

    pub fn set_right_image_raw(&mut self, right_image: Vec<u8>) {
        let _timer = Timer::new("Load raw image");
        self.right_image = ImageReader::new(Cursor::new(right_image))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
    }

    pub fn set_left_image_raw(&mut self, left_image: Vec<u8>) {
        let _timer = Timer::new("Load raw image");
        self.left_image = ImageReader::new(Cursor::new(left_image))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .to_rgb8();
    }

    pub fn set_stereoscopic_image(
        &mut self,
        width: u32,
        height: u32,
        stereo_image: Vec<u8>,
    ) -> bool {
        let stereoscopic_image = RgbImage::from_raw(width, height, stereo_image);

        if width % 2 != 0 {
            return false;
        }

        self.left_image = imageops::crop_imm(
            stereoscopic_image.as_ref().unwrap(),
            0,
            0,
            width / 2,
            height,
        )
        .to_image();

        self.right_image =
            imageops::crop_imm(&stereoscopic_image.unwrap(), width / 2, 0, width, height)
                .to_image();

        true
    }

    pub fn set_gif(&mut self, gif: Vec<u8>) {
        self.gif = gif.clone();
    }

    pub fn to_anaglyph(
        &self,
        anaglyph_type: Option<String>,
        offset_x: i32,
        offset_y: i32,
    ) -> Result<AnaglyphResult, JsValue> {
        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color,
        };
        let _timer = Timer::new("To Anaglyph");

        let offset: Offset = Offset {
            x: offset_x,
            y: offset_y,
        };

        let anaglyph_image =
            left_right_to_anaglyph_offset(&self.left_image, &self.right_image, a_type, offset)
                .map_err(|e| JsValue::from_str(&format!("anaglyph conversion failed: {e}")))?;

        Ok(AnaglyphResult {
            image_vec: (anaglyph_image.convert() as RgbaImage).into_vec(),
            height: anaglyph_image.height(),
            width: anaglyph_image.width(),
        })
    }

    /// Convert the stored gif into an anaglyph gif, returned as raw gif
    /// bytes.
    ///
    /// With the `gpu` feature enabled, frame blending runs on the GPU via
    /// WebGPU and falls back to a CPU loop when no usable adapter is
    /// available (the reason is logged to the console); without it, blending
    /// stays on the CPU. The method is async so the JS API is stable across
    /// both builds: callers must `await` the returned promise.
    // wasm-only upstream: the async gif blend in anaglyph-rs exists solely
    // for the browser build; native callers use the synchronous path.
    #[cfg(target_arch = "wasm32")]
    pub async fn gif_to_anaglyph_async(
        &self,
        anaglyph_type: Option<String>,
        video_direction: Option<String>,
    ) -> Result<Vec<u8>, JsValue> {
        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color,
        };
        let direction = match video_direction.as_deref() {
            Some("clockwise") => VideoDirection::Clockwise,
            Some("counter-clockwise") => VideoDirection::CounterClockwise,
            Some(_) | None => VideoDirection::Clockwise,
        };

        let _timer = Timer::new("To Anaglyph");

        convert_gif_buffer_to_anaglyph_async(&self.gif, direction, a_type)
            .await
            .map_err(|e| JsValue::from_str(&format!("gif conversion failed: {e}")))
    }

    /// Same conversion as [`Self::gif_to_anaglyph_async`] but always on the
    /// CPU and fully synchronous (blocks the event loop until done). Used by
    /// the benchmark page to compare CPU vs WebGPU from the same build.
    pub fn gif_to_anaglyph_cpu(
        &self,
        anaglyph_type: Option<String>,
        video_direction: Option<String>,
    ) -> Result<Vec<u8>, JsValue> {
        if self.gif.is_empty() {
            return Err(JsValue::from_str("no gif data uploaded"));
        }
        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color,
        };
        let direction = match video_direction.as_deref() {
            Some("clockwise") => VideoDirection::Clockwise,
            Some("counter-clockwise") => VideoDirection::CounterClockwise,
            Some(_) | None => VideoDirection::Clockwise,
        };

        convert_gif_buffer_to_anaglyph(&self.gif, direction, a_type, BlenderChoice::Cpu)
            .map_err(|e| JsValue::from_str(&format!("gif conversion failed: {e}")))
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

/// Per-frame anaglyph blender for video conversion, backing the WebCodecs
/// pipeline in `js/video-pipeline.js`.
///
/// The JS side owns decode/encode (hardware `VideoDecoder` / `VideoEncoder`)
/// and the stereo pairing window; this struct only blends one frame pair at a
/// time: dense RGBA in, dense RGBA out, matching the gif blend in
/// `anaglyph-rs`. With the `gpu` feature the blend runs through WebGPU
/// (`GpuBlender::blend_rgba_async`) and falls back to a single-threaded CPU
/// loop when no usable adapter exists; the chosen path is logged and exposed
/// via [`Self::is_gpu`] so the UI can report it.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct VideoBlender {
    gpu: Option<anaglyph_rs::gpu::GpuBlender>,
    matrix: &'static [[f32; 9]; 2],
    /// Expected byte length of every input frame (`4 * width * height`).
    frame_len: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl VideoBlender {
    /// Create a blender for `width` x `height` RGBA frames. Async because the
    /// WebGPU adapter request must not block the event loop; it resolves even
    /// when no GPU is found (the blender then runs on the CPU).
    ///
    /// A static factory rather than an `async` constructor: wasm-bindgen
    /// deprecates async constructors because they emit invalid TypeScript.
    /// Use `await VideoBlender.new(...)`.
    ///
    /// `backend` optionally forces `"cpu"` (skips the GPU probe entirely);
    /// anything else, including omission, prefers WebGPU with a CPU fallback.
    /// Benchmarks use this to compare both paths in one build.
    pub async fn new(
        width: u32,
        height: u32,
        anaglyph_type: Option<String>,
        backend: Option<String>,
    ) -> Result<VideoBlender, JsValue> {
        let a_type = match anaglyph_type.as_deref() {
            Some("color") => AnaglyphType::Color,
            Some("half-color") => AnaglyphType::HalfColor,
            Some("grayscale") => AnaglyphType::GrayScale,
            Some("optimized") => AnaglyphType::Optimized,
            Some("true") => AnaglyphType::True,
            Some(_) | None => AnaglyphType::Color,
        };
        let matrix = anaglyph_rs::anaglyph::anaglyph_type_to_matrix(a_type);
        let frame_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(4))
            .ok_or_else(|| JsValue::from_str("frame dimensions overflow"))?;
        let force_cpu = backend.as_deref() == Some("cpu");

        #[cfg(feature = "gpu")]
        let gpu = if force_cpu {
            console::log_1(&"anaglyph: CPU blending forced".into());
            None
        } else {
            match anaglyph_rs::gpu::GpuBlender::new_async(width, height, matrix).await {
                Ok(blender) => {
                    console::log_1(&"anaglyph: blending video frames on WebGPU".into());
                    Some(blender)
                }
                Err(e) => {
                    console::warn_1(
                        &format!("anaglyph: WebGPU unavailable ({e}), using CPU blending").into(),
                    );
                    None
                }
            }
        };
        #[cfg(not(feature = "gpu"))]
        let gpu = {
            let _ = force_cpu;
            console::warn_1(&"anaglyph: this build has no gpu feature, using CPU blending".into());
            None
        };

        Ok(VideoBlender {
            gpu,
            matrix,
            frame_len,
        })
    }

    /// Blend one stereo pair of dense RGBA frames (`4 * width * height` bytes
    /// each) and return the blended frame. Left/right assignment (which eye
    /// sees which decoded frame) is the caller's choice, so auto-direction can
    /// swap them per pair.
    pub async fn blend(&mut self, left: &[u8], right: &[u8]) -> Result<Vec<u8>, JsValue> {
        if left.len() != self.frame_len || right.len() != self.frame_len {
            return Err(JsValue::from_str(
                "frame size mismatch: expected width*height*4 bytes per frame",
            ));
        }

        #[cfg(feature = "gpu")]
        if let Some(blender) = &mut self.gpu {
            return blender
                .blend_rgba_async(left, right)
                .await
                .map_err(|e| JsValue::from_str(&format!("gpu blend failed: {e}")));
        }

        Ok(cpu_blend_rgba(left, right, self.matrix))
    }

    /// Whether frames blend on WebGPU (false means the CPU fallback is in
    /// use).
    pub fn is_gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

/// CPU blend of two dense RGBA frames with the anaglyph matrix, mirroring the
/// (private) reference in `anaglyph_rs::gif`: fully transparent input pixels
/// stay transparent, everything else comes out opaque. Single-threaded, like
/// every other CPU fallback in the browser build.
///
/// Uses WebAssembly SIMD when the build enables `simd128` (the default for
/// `wasm32-unknown-unknown` since Rust 1.77; build with
/// `RUSTFLAGS=-Ctarget-cpu=mvp` to get the scalar path for A/B benchmarks).
/// The SIMD loop is bit-identical to the scalar one — `bench/e2e-video.mjs`
/// checks both paths against each other in the browser.
#[cfg(target_arch = "wasm32")]
fn cpu_blend_rgba(left: &[u8], right: &[u8], matrix: &[[f32; 9]; 2]) -> Vec<u8> {
    debug_assert_eq!(left.len(), right.len());
    let mut out = vec![0u8; left.len()];
    let count = left.len() / 4;

    #[cfg(target_feature = "simd128")]
    let scalar_from = {
        let chunks = count / 4;
        if chunks > 0 {
            blend_rgba_simd(&mut out, left, right, matrix, chunks);
        }
        chunks * 4
    };
    #[cfg(not(target_feature = "simd128"))]
    let scalar_from = 0;
    blend_rgba_scalar(&mut out, left, right, matrix, scalar_from, count);
    out
}

/// Scalar reference loop over pixels `from..to` (also the tail of the SIMD
/// path). Matches `combine_slices` semantics exactly, including the
/// round-half-away-from-zero quantizer.
#[cfg(target_arch = "wasm32")]
fn blend_rgba_scalar(
    out: &mut [u8],
    left: &[u8],
    right: &[u8],
    matrix: &[[f32; 9]; 2],
    from: usize,
    to: usize,
) {
    use anaglyph_rs::anaglyph::combine_slices;
    for i in from..to {
        let (l, r) = (&left[i * 4..i * 4 + 4], &right[i * 4..i * 4 + 4]);
        if l[3] == 0 && r[3] == 0 {
            continue;
        }
        combine_slices(&l[..3], &r[..3], &mut out[i * 4..i * 4 + 3], matrix);
        out[i * 4 + 3] = 255;
    }
}

/// SIMD blend of the first `chunks * 4` pixels, 4 pixels per 128-bit
/// register.
///
/// Each channel of 4 pixels is spread into the 32-bit lanes of one register,
/// then each output channel is the same matrix dot product the scalar loop
/// computes — same operation order, so results are bit-identical (wasm has no
/// FMA contraction). Quantization mirrors `to_u8`: `(v + 0.5)` then a
/// saturating truncation toward zero saturates negatives to 0 and anything
/// past 255 to 255 through the narrowing steps. Fully-transparent pixels
/// (alpha 0 in both eyes) zero the whole output pixel.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn blend_rgba_simd(out: &mut [u8], left: &[u8], right: &[u8], m: &[[f32; 9]; 2], chunks: usize) {
    use core::arch::wasm32::*;

    // Matrix coefficients broadcast across the 4-pixel lane groups.
    let mc = |m: &[[f32; 9]; 2], eye: usize, idx: usize| f32x4_splat(m[eye][idx]);
    let (l00, l01, l02) = (mc(m, 0, 0), mc(m, 0, 1), mc(m, 0, 2));
    let (l10, l11, l12) = (mc(m, 0, 3), mc(m, 0, 4), mc(m, 0, 5));
    let (l20, l21, l22) = (mc(m, 0, 6), mc(m, 0, 7), mc(m, 0, 8));
    let (r00, r01, r02) = (mc(m, 1, 0), mc(m, 1, 1), mc(m, 1, 2));
    let (r10, r11, r12) = (mc(m, 1, 3), mc(m, 1, 4), mc(m, 1, 5));
    let (r20, r21, r22) = (mc(m, 1, 6), mc(m, 1, 7), mc(m, 1, 8));

    let low_byte = u32x4_splat(0xff);
    let alpha_lane = i32x4_splat(255);
    let i_zero = i32x4_splat(0);

    // De-interleave one channel of 4 RGBA pixels into the low byte of each
    // 32-bit lane (`k` = 0 red, 1 green, 2 blue, 3 alpha). The shuffled bytes
    // are masked down to the lane's low byte before the f32 conversion.
    macro_rules! channel_f32 {
        ($px:expr, $k:literal) => {{
            let spread = match $k {
                0 => u8x16_shuffle::<0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12>($px, $px),
                1 => u8x16_shuffle::<1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13>($px, $px),
                2 => u8x16_shuffle::<2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14>(
                    $px, $px,
                ),
                _ => u8x16_shuffle::<3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15>(
                    $px, $px,
                ),
            };
            f32x4_convert_u32x4(v128_and(spread, u32x4_splat(0xff)))
        }};
    }
    macro_rules! channel_alpha {
        ($px:expr) => {{
            let spread =
                u8x16_shuffle::<3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15>($px, $px);
            v128_and(spread, low_byte)
        }};
    }
    // `to_u8` on 4 lanes: round half away from zero via +0.5 then a truncating
    // saturate; the i16/u8 narrowing clamps like the scalar branches do
    // (negatives -> 0, over-255 -> 255).
    #[inline(always)]
    fn quantize(v: v128) -> v128 {
        i32x4_trunc_sat_f32x4(f32x4_add(v, f32x4_splat(0.5)))
    }
    // One output channel: `(left dot row) + (right dot row)` with the exact
    // association order of the scalar `dot3` calls (float adds are not
    // associative and the loops must agree bit for bit).
    #[inline(always)]
    fn channel_out(
        (lr, lg, lb): (v128, v128, v128),
        (rr, rg, rb): (v128, v128, v128),
        a: v128,
        b: v128,
        c: v128,
        d: v128,
        e: v128,
        f: v128,
    ) -> v128 {
        let dl = f32x4_add(
            f32x4_add(f32x4_mul(lr, a), f32x4_mul(lg, b)),
            f32x4_mul(lb, c),
        );
        let dr = f32x4_add(
            f32x4_add(f32x4_mul(rr, d), f32x4_mul(rg, e)),
            f32x4_mul(rb, f),
        );
        quantize(f32x4_add(dl, dr))
    }

    for i in 0..chunks {
        let o = i * 16; // 4 pixels * 4 bytes
        let (ls, rs) = unsafe {
            (
                v128_load(left.as_ptr().add(o) as *const v128),
                v128_load(right.as_ptr().add(o) as *const v128),
            )
        };

        let lp = (
            channel_f32!(ls, 0),
            channel_f32!(ls, 1),
            channel_f32!(ls, 2),
        );
        let rp = (
            channel_f32!(rs, 0),
            channel_f32!(rs, 1),
            channel_f32!(rs, 2),
        );

        // Alpha == 0 in both eyes -> the whole output pixel stays zero.
        let transparent = v128_and(
            i32x4_eq(channel_alpha!(ls), u32x4_splat(0)),
            i32x4_eq(channel_alpha!(rs), u32x4_splat(0)),
        );

        let or_ = channel_out(lp, rp, l00, l01, l02, r00, r01, r02);
        let og = channel_out(lp, rp, l10, l11, l12, r10, r11, r12);
        let ob = channel_out(lp, rp, l20, l21, l22, r20, r21, r22);

        // Clear pixels transparent in both eyes, force alpha 255 elsewhere.
        let or_ = v128_bitselect(i_zero, or_, transparent);
        let og = v128_bitselect(i_zero, og, transparent);
        let ob = v128_bitselect(i_zero, ob, transparent);
        let oa = v128_bitselect(i_zero, alpha_lane, transparent);

        // Planar [R0..3, G0..3, B0..3, A0..3] then interleaved back to RGBA.
        let rg = i16x8_narrow_i32x4(or_, og);
        let ba = i16x8_narrow_i32x4(ob, oa);
        let planar = u8x16_narrow_i16x8(rg, ba);
        let pixels =
            u8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(planar, planar);
        unsafe { v128_store(out.as_mut_ptr().add(o) as *mut v128, pixels) };
    }
}

/// Whether this build blends CPU-side frames with WebAssembly SIMD (false:
/// scalar fallback, e.g. a `target-cpu=mvp` build).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn blend_is_simd() -> bool {
    cfg!(all(target_feature = "simd128"))
}

/// Blend a dense RGBA pair on the CPU with the matrix for `anaglyph_type`,
/// forcing either the SIMD or the scalar reference loop. Exists so the e2e
/// suite can prove both loops are byte-identical from the browser; regular
/// callers use [`VideoBlender::blend`].
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn cpu_blend_debug(
    left: &[u8],
    right: &[u8],
    anaglyph_type: Option<String>,
    scalar: bool,
) -> Result<Vec<u8>, JsValue> {
    if left.len() != right.len() {
        return Err(JsValue::from_str("left and right frames differ in size"));
    }
    let a_type = match anaglyph_type.as_deref() {
        Some("color") => AnaglyphType::Color,
        Some("half-color") => AnaglyphType::HalfColor,
        Some("grayscale") => AnaglyphType::GrayScale,
        Some("optimized") => AnaglyphType::Optimized,
        Some("true") => AnaglyphType::True,
        Some(_) | None => AnaglyphType::Color,
    };
    let matrix = anaglyph_rs::anaglyph::anaglyph_type_to_matrix(a_type);
    let mut out = vec![0u8; left.len()];
    let count = left.len() / 4;
    if scalar {
        blend_rgba_scalar(&mut out, left, right, &matrix, 0, count);
    } else {
        out = cpu_blend_rgba(left, right, &matrix);
    }
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
use anaglyph_rs::motion::{trend_to_direction, MotionSample};

/// Per-frame camera drift estimator for auto-direction pairing. Wraps the
/// native `MotionEstimator` and consumes the same dense RGBA frames the
/// pipeline already produces for [`VideoBlender`], so decoded frames feed
/// motion analysis without a separate format conversion on the JS side.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct MotionAnalyzer {
    inner: anaglyph_rs::motion::MotionEstimator,
    /// Reusable dense RGB24 buffer; the estimator expects RGB24.
    rgb: Vec<u8>,
    frame_len: usize,
    last_reliable: bool,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl MotionAnalyzer {
    /// New estimator for `width` x `height` frames. Every
    /// [`Self::observe_rgba`] call must supply exactly `width * height * 4`
    /// bytes.
    pub fn new(width: u32, height: u32) -> Result<MotionAnalyzer, JsValue> {
        let rgb_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| JsValue::from_str("frame dimensions overflow"))?;
        Ok(MotionAnalyzer {
            inner: anaglyph_rs::motion::MotionEstimator::new(width as usize, height as usize),
            rgb: vec![0; rgb_len],
            frame_len: rgb_len / 3 * 4,
            last_reliable: false,
        })
    }

    /// Match the next frame (dense RGBA) against its predecessor and return
    /// the signed horizontal drift in source pixels per frame, positive when
    /// the image content moves right. Check [`Self::is_reliable`] before
    /// trusting the value.
    pub fn observe_rgba(&mut self, rgba: &[u8]) -> Result<f32, JsValue> {
        if rgba.len() != self.frame_len {
            return Err(JsValue::from_str(
                "frame size mismatch: expected width*height*4 bytes",
            ));
        }
        pack_rgb24(&mut self.rgb, rgba);
        let sample = self.inner.observe(&self.rgb);
        self.last_reliable = sample.reliable;
        Ok(sample.dx)
    }

    /// Whether the last [`Self::observe_rgba`] sample was trustworthy
    /// (false: scene cut, motion beyond the search range, textureless input,
    /// or the very first frame).
    pub fn is_reliable(&self) -> bool {
        self.last_reliable
    }
}

/// Pack dense RGBA into dense RGB24 (drop alpha). Uses wasm SIMD when the
/// build enables `simd128`: 4 pixels per register, one shuffle plus two
/// partial lane stores, remainder scalar. Byte-identical to the scalar loop.
#[cfg(target_arch = "wasm32")]
fn pack_rgb24(out: &mut [u8], rgba: &[u8]) {
    let count = rgba.len() / 4;

    #[cfg(target_feature = "simd128")]
    {
        use core::arch::wasm32::*;
        for i in 0..count / 4 {
            let o = i * 16;
            let pixels = unsafe { v128_load(rgba.as_ptr().add(o) as *const v128) };
            // RGBA×4 -> [R G B]×4 in the low 12 bytes.
            let packed =
                u8x16_shuffle::<0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0>(pixels, pixels);
            unsafe {
                let dst = out.as_mut_ptr().add(i * 12);
                v128_store64_lane::<0>(packed, dst as *mut u64);
                v128_store32_lane::<2>(packed, dst.add(8) as *mut u32);
            }
        }
    }

    let done = count / 4 * 4;
    let mut j = done * 3;
    for chunk in rgba[done * 4..].chunks_exact(4) {
        out[j..j + 3].copy_from_slice(&chunk[..3]);
        j += 3;
    }
}

/// Camera orbit direction tracker for `direction: "auto"`: wraps the native
/// [`anaglyph_rs::motion::DirectionTracker`] and exposes the trend-to-eye-
/// pairing mapping, so the JS pairing window gates and labels its output
/// exactly like the native CLI's auto mode.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Tracker {
    inner: anaglyph_rs::motion::DirectionTracker,
    reverse: bool,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Tracker {
    /// `reverse` flips the detected pairing, for rigs whose physical
    /// rotation runs opposite to the convention (the CLI's `auto-reverse`).
    #[wasm_bindgen(constructor)]
    pub fn new(reverse: bool) -> Tracker {
        Tracker {
            inner: anaglyph_rs::motion::DirectionTracker::new(Default::default()),
            reverse,
        }
    }

    /// Record the drift sample of frame `index` (0-based decode order) at
    /// `time_secs` (source presentation time, seconds).
    pub fn push(&mut self, index: usize, time_secs: f64, dx: f32, reliable: bool) {
        self.inner
            .push(index, time_secs, &MotionSample { dx, reliable });
    }

    /// Highest frame index whose pairing can no longer change; pairs whose
    /// newer frame is at or below it may be emitted. `-1` while nothing is
    /// confirmed yet.
    pub fn confirmed_through(&self) -> f64 {
        self.inner.confirmed_through().map_or(-1.0, |c| c as f64)
    }

    /// Eye pairing for the frames following `confirmed_through()`, as
    /// `"clockwise"` or `"counter-clockwise"` (the un-reversed base pairing
    /// while no trend has formed).
    pub fn current_direction(&self) -> String {
        direction_name(trend_to_direction(self.inner.current_trend(), self.reverse))
    }

    /// End of stream: everything held back commits to the current trend.
    pub fn finish(&mut self) {
        self.inner.finish();
    }

    /// Drain the reversals confirmed since the last call, packed as f64
    /// quads `[frame, time_secs, from_clockwise, to_clockwise]` with the
    /// direction slots 1 for clockwise, 0 for counter-clockwise (already
    /// mapped through `reverse`).
    pub fn take_reversals(&mut self) -> Vec<f64> {
        let mut out = Vec::new();
        for r in self.inner.take_reversals() {
            out.push(r.frame as f64);
            out.push(r.time);
            out.push(direction_bit(trend_to_direction(
                Some(r.from),
                self.reverse,
            )));
            out.push(direction_bit(trend_to_direction(Some(r.to), self.reverse)));
        }
        out
    }
}

#[cfg(target_arch = "wasm32")]
fn direction_name(direction: VideoDirection) -> String {
    match direction {
        VideoDirection::Clockwise => "clockwise".to_string(),
        VideoDirection::CounterClockwise => "counter-clockwise".to_string(),
    }
}

#[cfg(target_arch = "wasm32")]
fn direction_bit(direction: VideoDirection) -> f64 {
    match direction {
        VideoDirection::Clockwise => 1.0,
        VideoDirection::CounterClockwise => 0.0,
    }
}
