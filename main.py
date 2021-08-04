import pyaudio
import wave
from scipy.fft import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from datetime import datetime
from pynput import mouse, keyboard
#import threading
from collections import namedtuple
import time
# This might seem mad. Why use threading functions from a UI library? Simply: I'm used to them. Wanted a threading solution I liked. I like this one.


class FishingListener:
    def __init__(self, device_name):
        """Duration: Time to listen for, in milliseconds"""
        super().__init__()

        self.p = pyaudio.PyAudio()
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.fs = 48000
        self.filename = "output.wav"

        self.device = 0
        for i in range(self.p.get_device_count()):
            if self.p.get_device_info_by_index(i)["name"] == device_name:
                self.device = i

        self.folder_output_path = "./fishing_log"  # The folder for the listener to save its reports to.
        self.stop_command = True
        self.on = True
        self.stream = None

        self.mouse = mouse.Controller()
        self.aim_position = (0, 0)  # Where to position the mouse when fishing.

        try:
            shutil.rmtree(self.folder_output_path)  # Clear log folder.
        except FileNotFoundError:
            pass
        p = Path(f"{self.folder_output_path}")
        p.mkdir(parents=True, exist_ok=True)

    def listen(self):
        """Enters the listening loop."""
        interval_size = 0.125  # How large the chunks of audio we analyse are. I want to increase this for performance, eventually.
        curr_audio = []
        prev_audio = []
        read_count = int(self.fs / self.chunk * interval_size)

        l = keyboard.Listener(
            on_press=self.on_press
        )
        l.start()

        # Open a stream
        # TODO: How do I handle the analysis lagging behind the audio stream?
        # FIXME: Starting this stream causes some audio on my PC to stop working.
        self.stream = self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True,
            input_device_index=self.device
        )

        # Load the first chunk of "previous audio"
        for i in range(read_count):
            prev_audio += self.stream.read(self.chunk)
        prev_audio = self.convert_bytes(prev_audio)

        # FIXME: Fishing sound is subtly pitch shifted each time it is played by a couple hz. This causes false negatives sometimes.
        #  I could solve this by trying to locate the first position were the given peak thresholds match up.
        #  However, this might increase the computational load a lot more, and I haven't properly assessed how much I have to work with.
        while True:
            if self.stop_command:
                self.on = not self.on
                print(f"Now: {self.on}")
                if self.on:
                    self.stream.start_stream()
                    self.aim_position = self.mouse.position
                else:
                    self.stream.stop_stream()
                prev_audio = []
                curr_audio = []
                self.stop_command = False

            if not self.on:
                time.sleep(0.1)
                continue

            # Read the given amount of time from the stream
            for i in range(read_count):
                curr_audio += self.stream.read(self.chunk)
            curr_audio = self.convert_bytes(curr_audio)
            audio = prev_audio + curr_audio

            fishing_sound = self.analyse_fishing(audio)
            # If it doesn't work, find better checks, don't add more. I want to keep the number of checks minimal so code doesn't become spaghetti.
            if fishing_sound.found:
                self.stream.stop_stream()
                time_now = datetime.now().strftime("%H-%M-%S.%f")
                self.heard_sound()  # Reel line in, cast again.
                self.save_analysis(time_now, fishing_sound)
                self.save_audio(time_now, audio)
                self.stream.start_stream()
                curr_audio = []  # Clear audio cache

            prev_audio = curr_audio
            curr_audio = []
        self.stream.close()

    def on_press(self, key):
        if key == keyboard.Key.f2:
            self.stop_command = True

    def stop_listen(self):
        self.stop = True

    def heard_sound(self):
        pos_before = self.mouse.position
        # Reel line in
        self.mouse.position = self.aim_position
        self.click()
        self.mouse.position = pos_before
        time.sleep(0.5)

        # Cast line
        self.mouse.position = self.aim_position
        self.click()
        self.mouse.position = pos_before
        time.sleep(0.7)

    def analyse_fishing(self, audio):
        """
        Analyse PCM audio and return an object with the data.
        """
        fishing_sound = namedtuple("fishing_sound", "xf yf found peak1 peak2 dip1")
        yf = rfft(audio)  # Gets the transform. Imaginary numbers.
        xf = rfftfreq(len(audio), 1 / self.fs)  # This calculates the "grouping" of frequencies. Might be able to mess with this for my ranges

        # TODO: Refine my tests further.
        peak1 = get_range_sum(xf, yf, 230, 240)
        peak2 = get_range_sum(xf, yf, 260, 330)
        dip1 = get_range_dip(xf, yf, 240, 270)
        found = True if self.threshold_check(peak1, peak2) else False

        return fishing_sound(xf, yf, found, peak1, peak2, dip1)

    def save_analysis(self, name, analysis):
        """
        Saves the a graph of the analysis.
        """
        # Create a graph of the audio
        plt.title(f"{name}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.axvspan(220, 250, color="mediumaquamarine")  # Colour the analysed bands
        plt.axvspan(270, 330, color="mediumaquamarine")
        plt.axvspan(240, 270, color="indianred")
        plt.axis([150, 600, 0, 400000])  # Limit axis to ranges we care about.
        plt.plot(analysis.xf, np.abs(analysis.yf))
        plt.savefig(f'{self.folder_output_path}/{name}.png')
        plt.close()

    def save_audio(self, name, PCM_data):
        """Saves the given data to a WAV file.
        name: Name of file to save to
        PCM_data: Uncompressed audio data to be saved."""
        # Convert PCM data to bytes.
        _bytes = b''
        for number in PCM_data:
            _bytes += number.to_bytes(2, "little", signed=True)

        wf = wave.open(f"{self.folder_output_path}/{name}.wav", "wb")
        wf.setframerate(44100)
        wf.setsampwidth(2)
        wf.setnchannels(1)
        wf.writeframes(_bytes)
        wf.close()

    def click(self):
        self.mouse.press(mouse.Button.left)
        time.sleep(0.06)  # About 3 frames in a 60 fps game.
        self.mouse.release(mouse.Button.left)

    @staticmethod
    def convert_bytes(_bytes):
        new_frame = []
        for i in range(len(_bytes) // 2):
            sample = _bytes[i * 2:i * 2 + 2]
            try:
                int_value = int.from_bytes(sample, byteorder="little", signed=True)
            except Exception as e:
                print(sample)
                raise e
            new_frame.append(int_value)
        return new_frame

    def parse_file_intervals_overlap(self, file, interval=0.25):
        wf = wave.open(file, "rb")
        rate = wf.getframerate()
        num_samples = int(rate * (interval / 2))  # Read the audio in chunks of this size.
        previous_audio = wf.readframes(num_samples)

        i = 0
        while True:
            if i > 7:
                break

            from_time = (interval / 2) * i
            to_time = (interval / 2) * (i + 2)
            curr_audio = wf.readframes(num_samples)
            if not curr_audio:  # End of file
                break

            audio = previous_audio + curr_audio
            audio = FishingListener.convert_bytes(audio)

            fishing_sound = self.analyse_fishing(audio)
            # I want to keep the number of checks minimal so code doesn't become spaghetti.
            # If it doesn't work, find better checks, don't add more.
            if fishing_sound.found:
                name = f"{from_time}-{to_time}"
                self.save_analysis(name, fishing_sound)
                self.save_audio(name, audio)
                print(f"{name}   {round(fishing_sound.peak1, 2)}   {round(fishing_sound.peak2, 2)}   {round(fishing_sound.dip1, 2)}")

            previous_audio = curr_audio
            i += 1

    def threshold_check(self, peak1, peak2):
        """This is a simple "volume" check, used to filter out low levels such as noise. Prebaked thresholds.
        Should be replaced in the future with a more advanced check.

        Arguments:
            Each peak is the sum of the predefined range.
        Returns: True if all thresholds are correct.
        """
        if not peak1 > 200000:
            return False
        if not peak2 > 500000:
            return False
        if not peak2 > peak1:
            return False

        return True

    def relative_check(self, peak1, dip1):
        """Runs through the checks that describe shapes relative to each other."""
        compar_size = dip1 / peak1
        if not compar_size < 0.75:  # Our "dip" is at most 75% the first peak.
            return False
        return True


class Trott:
    def __init__(self, audio_device_name):
        self.ears = {"fishing": FishingListener(audio_device_name)}

    def update(self):
        self.ears["Fishing"].test()


def get_range(xf, yf, low, high):
    """Gets the values in a range of frequencies.

    Arguments:
        xf: A list of frequencies, generated by rfftfreq
        yf: The amplitudes of frequencies, generated by rfft
        low: Lower bound of frequency to capture
        high: Upper bound of frequency to capture

    returns: Array of values within frquencies.
    """
    x1 = -1
    for i in range(len(xf)):
        if xf[i] >= low:
            x1 = i
            break
    if x1 == -1:
        raise ValueError

    x2 = -1
    for i in range(len(xf) - x1):
        if xf[i + x1] >= high:
            x2 = i + x1
            break
    if x2 == -1:
        raise ValueError

    return [abs(x) for x in yf[x1:x2]]


def get_range_peak(xf, yf, low, high):
    """Gets the total amplitude of a given range of frequencies.

        Arguments:
            xf: A list of frequencies, generated by rfftfreq
            yf: The amplitudes of frequencies, generated by rfft
            low: Lower bound of frequency to capture
            high: Upper bound of frequency to capture

        returns: The loudest volume found in the given range
        """
    values = get_range(xf, yf, low, high)
    return np.amax(values)


def get_range_dip(xf, yf, low, high):
    """Gets the total amplitude of a given range of frequencies.

    Arguments:
        xf: A list of frequencies, generated by rfftfreq
        yf: The amplitudes of frequencies, generated by rfft
        low: Lower bound of frequency to capture
        high: Upper bound of frequency to capture

    returns: The loudest volume found in the given range
    """
    values = get_range(xf, yf, low, high)
    return np.amin(values)


def get_range_sum(xf, yf, low, high):
    """Gets the total amplitude of a given range of frequencies.

    Arguments:
        xf: A list of frequencies, generated by rfftfreq
        yf: The amplitudes of frequencies, generated by rfft
        low: Lower bound of frequency to capture
        high: Upper bound of frequency to capture

    returns: The amplitude of the range of frequencies
    """
    # TODO: I could change this to a "get average of range" function. This would avoid issues with bitrates or range sizes.
    x1 = -1
    for i in range(len(xf)):
        if xf[i] >= low:
            x1 = i
            break
    if x1 == -1:
        raise ValueError

    x2 = -1
    for i in range(len(xf) - x1):
        if xf[i+x1] >= high:
            x2 = i + x1
            break
    if x2 == -1:
        raise ValueError

    return np.sum([abs(x) for x in yf[x1:x2]])


def get_range_average(xf, yf, low, high):
    """Gets the total amplitude of a given range of frequencies.

    Arguments:
        xf: A list of frequencies, generated by rfftfreq
        yf: The amplitudes of frequencies, generated by rfft
        low: Lower bound of frequency to capture
        high: Upper bound of frequency to capture

    returns: The amplitude of the range of frequencies
    """
    # TODO: I could change this to a "get average of range" function. This would avoid issues with bitrates or range sizes.
    x1 = -1
    for i in range(len(xf)):
        if xf[i] >= low:
            x1 = i
            break
    if x1 == -1:
        raise ValueError

    x2 = -1
    for i in range(len(xf) - x1):
        if xf[i+x1] >= high:
            x2 = i + x1
            break
    if x2 == -1:
        raise ValueError

    return np.mean([abs(x) for x in yf[x1:x2]])


# TODO: I need to find a way to compare the values of my graph to its surroundings, to detect when a spike has happened.
#  I could set specific number thresholds. This would be easiest, but will also break fastest.
#  I could compare to the last "frame". This scales with volume, but is complex.
#  I could compare the "shape", E.G peak 1 is about 2.5x bigger than peak 2.
#  I could create a normalized standard for the frequency, similar to "last frame".


if __name__ == "__main__":
    #TerrariaBot("Line In (High Definition Audio ")
    l = FishingListener("Line In (High Definition Audio ")
    #l.parse_file_intervals_overlap("Clean_fishing_example.wav")
    l.listen()
