import init, { Anaglyph } from "./pkg/anaglyph_wasm.js";
var anaglyph;
async function run() {
  await init();

  // And afterwards we can use all the functionality defined in wasm.
  anaglyph = Anaglyph.new();
  console.log("Worker initialized");
  self.postMessage({ type: "ready" });
}
run();

self.onmessage = function (e) {
  switch (e.data.type) {
    case "uploadLeftImage":
      handleLeftImageUpload(e);
      break;
    case "uploadRightImage":
      handleRightImageUpload(e);
      break;
    case "uploadGif":
      handleGifUpload(e);
      break;
    case "anaglyph-submit":
      handleAnaglyphSubmit(e);
      break;
    case "anaglyph-submit-gif":
      handleAnaglyphSubmitGif(e);
      break;
    default:
      break;
  }
};

async function handleLeftImageUpload(e) {
  // e.data.image is a file object, convert it to an array buffer using blob
  var leftImage = new Uint8ClampedArray(await e.data.image.arrayBuffer());
  anaglyph.set_left_image_raw(leftImage);
  postMessage({ type: "left-image-loaded" });
}

async function handleRightImageUpload(e) {
  // e.data.image is a file object, convert it to an array buffer using blob
  var rightImage = new Uint8ClampedArray(await e.data.image.arrayBuffer());
  anaglyph.set_right_image_raw(rightImage);
  postMessage({ type: "right-image-loaded" });
}

async function handleGifUpload(e) {
  var gifData = new Uint8Array(await e.data.gif.arrayBuffer());
  anaglyph.set_gif(gifData);
  postMessage({ type: "gif-loaded" });
}

function handleAnaglyphSubmit(e) {
  var offset = e.data.offset;
  var anaglyphType = e.data.anaglyphType;
  var x = offset.x;
  var y = offset.y;
  var result = anaglyph.to_anaglyph(anaglyphType, x, y);
  var anaglyph_image = new Uint8ClampedArray(result.get_image());
  var anaglyphHeight = result.height;
  var anaglyphWidth = result.width;
  var canvas = new OffscreenCanvas(anaglyphWidth, anaglyphHeight);
  var ctx = canvas.getContext("2d");
  var imageData = new ImageData(anaglyph_image, anaglyphWidth, anaglyphHeight);
  ctx.putImageData(imageData, 0, 0);
  var imageBitmap = canvas.transferToImageBitmap();
  postMessage({ type: "anaglyph-result", image: imageBitmap }, [imageBitmap]);
}

function handleAnaglyphSubmitGif(e) {
  var anaglyphType = e.data.anaglyphType;
  var videoDirection = e.data.videoDirection;
  var resultGif = anaglyph.gif_to_anaglyph(anaglyphType, videoDirection);
  var blob = new Blob([new Uint8Array(resultGif)], { type: "image/gif" });
  var url = URL.createObjectURL(blob);
  postMessage({ type: "anaglyph-result-gif", url: url });
}
