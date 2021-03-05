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
        recognizer.listen(result => {
            // - result.scores contains the probability scores that correspond to
            //   recognizer.wordLabels().
            // - result.spectrogram contains the spectrogram of the recognized word.
            if (result.scores.indexOf(Math.max.apply(Math, result.scores)) > 0) {
                document.getElementById('euhh').style.display = 'block';
                //document.getElementById('beep').play();
                setTimeout(function() {
                    document.getElementById('euhh').style.display = 'none';
                }, 1500);
                console.log(result.scores);
            }
        }, {
            includeSpectrogram: true,
            probabilityThreshold: 0.9,
            overlapFactor: 0.2
        });
    });
