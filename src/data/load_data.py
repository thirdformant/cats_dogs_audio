import sys, os
import logging
import click
import librosa

def load_wav_files(filenames, raw_audio, duration):
    wav_files = []
    sr = []
    file_name = []
    for f in filenames:
        audio_samples = librosa.core.load(os.path.join(raw_audio, f),
                                          sr=None, offset=0.0,
                                          duration=duration)
        wav_files.append(audio_samples[0])
        sr.append(audio_samples[1])
        file_name.append(os.path.splitext(f)[0])
    return wav_files, sr, file_name

def load_data(input_filepath, duration=None):
    # Get list of filenames in raw data directory
    RAW_AUDIO = os.path.join(input_filepath, "cats_dogs")
    X_filenames = os.listdir(RAW_AUDIO)

    # if filename contains 'cat' y=0 else y=1 (dog)
    y = [0 if 'cat' in f else 1 for f in X_filenames]

    X_all, X_sr, file_name = load_wav_files(X_filenames, RAW_AUDIO, duration)

    # Just in case something goes wrong
    # Ensure the length of the all lists is equal
    assert len(y) == len(X_filenames) == len(X_all) == len(X_sr)
    return X_all, X_sr, y, file_name

def cat_dog_split(X, y):
    X_cats = [_x for _x, _y in zip(X, y) if _y == 0]
    X_dogs = [_x for _x, _y in zip(X, y) if _y == 1]
    return X_cats, X_dogs
