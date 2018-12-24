import os
from pathlib import Path
from typing import Optional
import numpy as np
import librosa #Library for audio data

def load_wav_file(files:list, path:Path, sample_rate:Optional[int]=None,
                  offset:float=0.0, duration:Optional[int]=None) -> np.ndarray:
  '''
  Reads .wav files into numpy array
  '''
  samples_all = []
  file_counter = 1
  for f in files:
    print(f'Reading .wav {file_counter}/{len(files)}')
    full_path = path / f
    if full_path.suffix == '.wav':
      audio_samples = librosa.core.load(full_path, sr=sample_rate,
                                        offset=offset, duration=duration)
      samples_all.append(audio_samples[0])
    file_counter += 1
  return np.array(samples_all)

def get_labels(files:list) -> np.ndarray:
  '''
  Get classification labels from file names
  '''
  labels = []
  for f in files:
    if 'cat' in f:
      labels.append('cat')
    elif 'dog' in f:
      labels.append('dog')
  return np.array(labels)

if __name__ == '__main__':
    INPUT_PATH = Path('data/raw/cats_dogs')
    OUTPUT_PATH = Path('data/interim')
    files_list = os.listdir(INPUT_PATH)
    X_all = load_wav_file(files_list, INPUT_PATH)
    labels = get_labels(files_list)

    np.save(OUTPUT_PATH / 'wav_samples.npy', X_all)
    np.save(OUTPUT_PATH / 'labels.npy', labels)
    np.save(OUTPUT_PATH / 'filenames.npy',
            np.array(files_list))
