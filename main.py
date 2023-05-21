import numpy as np
import sounddevice as sd
import scipy.io.wavfile
import scipy.fftpack
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time
import os
import copy

CONCERT_PITCH = 440
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
BASIC_NOTES_AND_PITCHES = {
    "E4": 329.64,
    "B3": 246.94,
    "G3": 196,
    "D3": 146.83,
    "A2": 110,
    "E2": 82.41,
}

def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch


def record_audio_and_save(sample_dur, sample_freq):
    print("Grab the guitar!")
    time.sleep(1)

    # Actual Recording
    my_recording = sd.rec(sample_dur * sample_freq, samplerate=sample_freq, channels=1, dtype='float64')
    print("Recording audio")
    sd.wait()

    # Play the audio
    sd.play(my_recording, sample_freq)
    print("Playing audio")
    sd.wait()

    # Save the recording in a wav file
    scipy.io.wavfile.write('example1.wav', sample_freq, my_recording)

    return my_recording


def plot_signal_time():
    sampleFreq, myRecording = scipy.io.wavfile.read("example1.wav")
    sampleDur = len(myRecording) / sampleFreq
    timeX = np.arange(0, sampleDur, 1 / sampleFreq)

    plt.plot(timeX, myRecording)
    plt.ylabel('x(k)')
    plt.xlabel('time[s]')
    plt.show()


SELECTION = "G3"


def callback(indata, frames, time, status):
    if not hasattr(callback, "window_samples"):
        callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
    if not hasattr(callback, "noteBuffer"):
        callback.noteBuffer = ["1", "2"]
    if not hasattr(callback, "selection"):
        callback.selection_pitch = BASIC_NOTES_AND_PITCHES[SELECTION]

    if status:
        print(status)
        return
    if any(indata):
        callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0]))  # append new samples
        callback.window_samples = callback.window_samples[len(indata[:, 0]):]  # remove old samples

        # skip if signal power is too low
        signal_power = (np.linalg.norm(callback.window_samples, ord=2) ** 2) / len(callback.window_samples)
        if signal_power < POWER_THRESH:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Closest note: ...")
            return

        # avoid spectral leakage by multiplying the signal with a hann window
        hann_samples = callback.window_samples * HANN_WINDOW
        magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

        # supress mains hum, set everything below 62Hz to zero
        for i in range(int(62 / DELTA_FREQ)):
            magnitude_spec[i] = 0

        # calculate average energy per frequency for the octave bands
        # and suppress everything below it
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
            avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2) / (
                        ind_end - ind_start)
            avg_energy_per_freq = avg_energy_per_freq ** 0.5
            for i in range(ind_start, ind_end):
                magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[
                                                             i] > WHITE_NOISE_THRESH * avg_energy_per_freq else 0

        # interpolate spectrum
        mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                                  magnitude_spec)
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

        hps_spec = copy.deepcopy(mag_spec_ipol)

        # calculate the HPS
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        max_ind = np.argmax(hps_spec)
        max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        callback.noteBuffer.insert(0, closest_note)  # note that this is a ringbuffer
        callback.noteBuffer.pop()

        os.system('cls' if os.name == 'nt' else 'clear')
        if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
            print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")
            if callback.selection_pitch - 1 <= max_freq <= callback.selection_pitch + 1:
                print("Tuned")
            elif max_freq < callback.selection_pitch:
                print("Tune Up")
            elif max_freq > callback.selection_pitch:
                print("Tune Down")
        else:
            print(f"Closest note: ...")

    else:
        print('no input')


def plot_dft():
    sample_freq, my_rec = scipy.io.wavfile.read("example1.wav")
    sampleDur = len(my_rec) / sample_freq
    time_x = np.arange(0, sample_freq / 2, sample_freq / len(my_rec))
    abs_freq_spectrum = abs(fft(my_rec))
    print(abs_freq_spectrum)

    plt.plot(time_x, abs_freq_spectrum[:len(my_rec) // 2])
    plt.ylabel('|X(n)|')
    plt.xlabel('frequency[Hz]')
    plt.show()


SAMPLE_FREQ = 48000   # Sampling frequency of the recording
SAMPLE_DUR = 4  # Duration of the recoding
WINDOW_SIZE = 48000   # window size of the DFT in samples
WINDOW_STEP = 12000   # step size of window

HANN_WINDOW = np.hanning(WINDOW_SIZE)
NUM_HPS = 5           # max number of harmonic product spectrums
POWER_THRESH = 1e-6 # tuning is activated if the signal power exceeds this threshold
WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
WHITE_NOISE_THRESH = 0.2 # everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT

OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

windowSamples = [0 for _ in range(WINDOW_SIZE)]

if __name__ == '__main__':
    # print(find_closest_note(1000))

    # Test and visualize the recording of a note
    # print(sd.default.device)
    #
    # recording = record_audio_and_save(SAMPLE_DUR, SAMPLE_FREQ)
    #
    # plot_signal_time()
    # plot_dft()

    # Start the microphone input stream
    try:
        print("Starting HPS guitar tuner...")
        with sd.InputStream(channels=1, callback=callback,
                            blocksize=WINDOW_STEP,
                            samplerate=SAMPLE_FREQ):
            while True:
                time.sleep(0.5)
    except Exception as e:
        print(str(e))
