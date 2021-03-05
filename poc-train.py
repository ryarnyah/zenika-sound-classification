import glob
import json
import os
import random

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tensorflow as tf
import tensorflowjs as tfjs
import tqdm

print(tf.__version__)
print(tfjs.__version__)

# mkdir -p tfjs-sc-model
# curl -o tfjs-sc-model/metadata.json -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/metadata.json
# curl -o tfjs-sc-model/model.json -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/model.json
# curl -o tfjs-sc-model/group1-shard1of2 -fSsL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/group1-shard1of2
# curl -o tfjs-sc-model/group1-shard2of2 -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/group1-shard2of2
# curl -o tfjs-sc-model/sc_preproc_model.tar.gz -fSsL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz
# cd tfjs-sc-model/ && tar xzvf sc_preproc_model.tar.gz
# mkdir -p speech_commands_v0.02
# curl -o speech_commands_v0.02/speech_commands_v0.02.tar.gz -fSsL http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# cd  speech_commands_v0.02 && tar xzf speech_commands_v0.02.tar.gz

preproc_model_path = 'tfjs-sc-model/sc_preproc_model'
preproc_model = tf.keras.models.load_model(preproc_model_path)
print(preproc_model.summary())
print(preproc_model.input_shape)

# Only 1s
TARGET_SAMPLE_TIME = 1.0
# Target sampling rate. It is required by the audio preprocessing model.
TARGET_SAMPLE_RATE = 44100
# The specific audio tensor length expected by the preprocessing model.
EXPECTED_WAVEFORM_LEN = preproc_model.input_shape[-1]

# Where the Speech Commands v0.02 dataset has been downloaded.
DATA_ROOT = "final-data"

WORDS = ("_background_noise_", "Euhh")

def resample_wavs(dir_path, target_sample_rate=44100):
  """Resample the .wav files in an input directory to given sampling rate.

  The resampled waveforms are written to .wav files in the same directory with
  file names that ends in "_44100hz.wav".

  44100 Hz is the sample rate required by the preprocessing model. It is also
  the most widely supported sample rate among web browsers and mobile devices.
  For example, see:
  https://developer.mozilla.org/en-US/docs/Web/API/AudioContextOptions/sampleRate
  https://developer.android.com/ndk/guides/audio/sampling-audio

  Args:
    dir_path: Path to a directory that contains .wav files.
    target_sapmle_rate: Target sampling rate in Hz.
  """
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  resampled_suffix = "_%shz.wav" % target_sample_rate
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if wav_path.endswith(resampled_suffix) or 'split' not in wav_path:
      continue
    sample_rate, xs = wavfile.read(wav_path)
    xs = xs.astype(np.float32)
    xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
    resampled_path = os.path.splitext(wav_path)[0] + resampled_suffix
    wavfile.write(resampled_path, target_sample_rate, xs)


def add_noise(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if 'data_aug' in wav_path or 'hz.wav' in wav_path:
      continue
    rate, wav = wavfile.read(wav_path)
    wav_n = wav + 0.009 * np.random.normal(0, 1, len(wav))
    wavfile.write(wav_path + '-data_aug_noise.wav', rate, wav_n)


def timeshift(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if 'data_aug' in wav_path or 'hz.wav' in wav_path:
      continue
    rate, wav = wavfile.read(wav_path)
    wav_n = np.roll(wav, int(rate / 10))
    wavfile.write(wav_path + '-data_aug-timeshift.wav', rate, wav_n)


def split_to_time(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for wav in wav_paths:
    if 'split' in wav:
      continue
    rate, data = wavfile.read(wav)
    batches = int(len(data) / (TARGET_SAMPLE_TIME * rate))
    for i in range(batches):
      wavfile.write(wav + '-data_aug-split' + '-' + str(i) + '.wav', rate, data[i * rate: (i+1) * rate])

@tf.function
def read_wav(filepath):
  file_contents = tf.io.read_file(filepath)
  audio = tf.audio.decode_wav(
    file_contents,
    desired_channels=1,
    desired_samples=TARGET_SAMPLE_RATE).audio
  return tf.expand_dims(tf.squeeze(audio, axis=-1), 0)

@tf.function
def filter_by_waveform_length(waveform, label):
  return tf.size(waveform) > EXPECTED_WAVEFORM_LEN


@tf.function
def crop_and_convert_to_spectrogram(waveform, label):
  cropped = tf.slice(waveform, begin=[0, 0], size=[1, EXPECTED_WAVEFORM_LEN])
  return tf.squeeze(preproc_model(cropped), axis=0), label

@tf.function
def spectrogram_elements_finite(spectrogram, label):
  return tf.math.reduce_all(tf.math.is_finite(spectrogram))


def get_dataset(input_wav_paths, labels):
  """Get a tf.data.Dataset given input .wav files and their labels.

  The returned dataset emits 2-tuples of `(spectrogram, label)`, wherein
  - `spectrogram` is a tensor of dtype tf.float32 and shape [43, 232, 1].
    It is z-normalized (i.e., have a mean of ~0.0 and variance of ~1.0).
  - `label` is a tensor of dtype tf.int32 and shape [] (scalar).

  Args:
    input_wav_paths: Input audio .wav file paths as a list of string.
    labels: integer labels (class indices) of the input .wav files. Must have
      the same lengh as `input_wav_paths`.

  Returns:
    A tf.data.Dataset object as described above.
  """
  ds = tf.data.Dataset.from_tensor_slices(input_wav_paths)
  # Read audio waveform from the .wav files.
  ds = ds.map(read_wav)
  ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))
  # Keep only the waveforms longer than `EXPECTED_WAVEFORM_LEN`.
  ds = ds.filter(filter_by_waveform_length)
  # Crop the waveforms to `EXPECTED_WAVEFORM_LEN` and convert them to
  # spectrograms using the preprocessing layer.
  ds = ds.map(crop_and_convert_to_spectrogram)
  # Discard examples that contain infinite or NaN elements.
  ds = ds.filter(spectrogram_elements_finite)
  return ds

# Resample data
for word in WORDS:
  word_dir = os.path.join(DATA_ROOT, word)
  assert os.path.isdir(word_dir)
  # data augmentation
  print('data augmentation for %s' % word)
  add_noise(word_dir)
  timeshift(word_dir)
  split_to_time(word_dir)
  resample_wavs(word_dir, target_sample_rate=TARGET_SAMPLE_RATE)

input_wav_paths_and_labels = []
for i, word in enumerate(WORDS):
  wav_paths = glob.glob(os.path.join(DATA_ROOT, word, "*_%shz.wav" % TARGET_SAMPLE_RATE))
  print("Found %d examples for class %s" % (len(wav_paths), word))
  labels = [i] * len(wav_paths)
  input_wav_paths_and_labels.extend(zip(wav_paths, labels))
random.shuffle(input_wav_paths_and_labels)

input_wav_paths, labels = ([t[0] for t in input_wav_paths_and_labels],
                           [t[1] for t in input_wav_paths_and_labels])
dataset = get_dataset(input_wav_paths, labels)

# The amount of data we have is relatively small. It fits into typical host RAM
# or GPU memory. For better training performance, we preload the data and
# put it into numpy arrays:
# - xs: The audio features (normalized spectrograms).
# - ys: The labels (class indices).
print(
    "Loading dataset and converting data to numpy arrays. "
    "This may take a few minutes...")
xs_and_ys = list(dataset)
xs = np.stack([item[0] for item in xs_and_ys])
ys = np.stack([item[1] for item in xs_and_ys])
print("Done.")

tfjs_model_json_path = 'tfjs-sc-model/model.json'

# Load the Speech Commands model. Weights are loaded along with the topology,
# since we train the model from scratch. Instead, we will perform transfer
# learning based on the model.
orig_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=True)

# Remove the top Dense layer and add a new Dense layer of which the output
# size fits the number of sound classes we care about.
model = tf.keras.Sequential(name="TransferLearnedModel")
for layer in orig_model.layers[:-1]:
  model.add(layer)
model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax"))

# Freeze all but the last layer of the model. The last layer will be fine-tuned
# during transfer learning.
for layer in model.layers[:-1]:
  layer.trainable = False

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
print(model.summary())

# Train the model.
model.fit(xs, ys, batch_size=256, validation_split=0.3, shuffle=True, epochs=60)

# Convert the model to TensorFlow.js Layers model format.

tfjs_model_dir = "tfjs-model"
tfjs.converters.save_keras_model(model, tfjs_model_dir)

# Create the metadata.json file.
metadata = {"words": ["_background_noise_"] + list(WORDS[1:]), "frameSize": model.input_shape[-2]}
with open(os.path.join(tfjs_model_dir, "metadata.json"), "w") as f:
  json.dump(metadata, f)
