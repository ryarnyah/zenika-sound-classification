import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

const recognizer = speechCommands.create(
    'BROWSER_FFT',
    null,
    'http://localhost:8080/model/model.json',  // URL to the custom model's model.json
    'http://localhost:8080/model/metadata.json'  // URL to the custom model's metadata.json
);

recognizer.ensureModelLoaded()
    .then(() => {

        // See the array of words that the recognizer is trained to recognize.
        console.log(recognizer.wordLabels());

        // `listen()` takes two arguments:
        // 1. A callback function that is invoked anytime a word is recognized.
        // 2. A configuration object with adjustable fields such a
        //    - includeSpectrogram
        //    - probabilityThreshold
        //    - includeEmbedding
        self.addEventListener("message", (event) => {
            if (!!event.data) {
                const spectrogramData = event.data;
                var y = recognizer.model.predict(tf.tensor(spectrogramData.data.data, spectrogramData.data.shape))
                y.data().then(scores => {
                    const maxScore = Math.max(...scores);
                    if (maxScore > 0.9) {
                        self.postMessage({result: scores, wordLabels: recognizer.wordLabels()});
                    }
                });
            }
        });
    });
