import init, { Anaglyph } from './pkg/anaglyph_wasm.js';
var anaglyph;
async function run() {

  await init();

  // And afterwards we can use all the functionality defined in wasm.
  anaglyph = Anaglyph.new();
  console.log("Worker initialized");
  self.postMessage({type: "ready"});
}
run();

self.onmessage = function(e) {
  switch (e.data.type) {
    case 'uploadLeftImage':
      handleLeftImageUpload(e);
      break;
    
    case 'uploadRightImage':
      handleRightImageUpload(e);
      break;
    case 'anaglyph-submit':
      handleAnaglyphSubmit(e);
      break;
    default:
      break;
  }

}

async function handleLeftImageUpload(e) {
  // e.data.image is a file object, convert it to an array buffer using blob
  var leftImage = new Uint8ClampedArray(await e.data.image.arrayBuffer());
  anaglyph.set_left_image_raw(leftImage);
  postMessage({type: "left-image-loaded"});
}

async function handleRightImageUpload(e) {
  // e.data.image is a file object, convert it to an array buffer using blob
  var rightImage = new Uint8ClampedArray(await e.data.image.arrayBuffer());
  anaglyph.set_right_image_raw(rightImage);
  postMessage({type: "right-image-loaded"});
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
  var ctx = canvas.getContext('2d');
  var imageData = new ImageData(anaglyph_image, anaglyphWidth, anaglyphHeight);
  ctx.putImageData(imageData, 0, 0);
  var imageBitmap = canvas.transferToImageBitmap();
  postMessage({type: "anaglyph-result", image: imageBitmap}, [imageBitmap]);
}