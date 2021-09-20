import {BrowserFftFeatureExtractor} from '@tensorflow-models/speech-commands/dist/browser_fft_extractor';
import {normalize} from '@tensorflow-models/speech-commands/dist/browser_fft_utils';
import * as speechCommands from '@tensorflow-models/speech-commands';

const worker = new Worker('worker.js')

const recognizer = speechCommands.create(
  'BROWSER_FFT',
  null,
  'http://localhost:8080/model/model.json',  // URL to the custom model's model.json
  'http://localhost:8080/model/metadata.json'  // URL to the custom model's metadata.json
);

recognizer.ensureModelLoaded().then(() => {
  // shape: (4) [1, 43, 232, 1]
  const shape = recognizer.modelInputShape();
  console.log(shape);

  function recognizeOnline() {
    var audioDataExtractor;
    return new Promise((resolve, reject) => {
      const spectrogramCallback = async (x) => {
        const normalizedX = normalize(x);
        worker.postMessage({data: {
          data: await normalizedX.data(),
          shape: normalizedX.shape
        }});
        normalizedX.dispose();
        return false;
      };
      audioDataExtractor = new BrowserFftFeatureExtractor({
        numFramesPerSpectrogram: shape[1],
        columnTruncateLength: shape[2],
        suppressionTimeMillis: 0,
        spectrogramCallback,
        overlapFactor: 0.25
      });
      audioDataExtractor.start();
    });
  }

  worker.addEventListener("message", async (event) => {
      const { result, wordLabels } = event.data;
      if (result.indexOf(Math.max.apply(Math, result)) > 0) {
          document.getElementById('euhh').style.display = 'block';
          //document.getElementById('beep').play();
          setTimeout(function() {
              document.getElementById('euhh').style.display = 'none';
          }, 1500);
          console.log(result);
      }
    });
  recognizeOnline().then(spectrogramData => worker.postMessage({data: spectrogramData}))
});
