import pyaudio
import wave
from scipy.fft import rfft, rfftfreq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


class FishingListener:
    def __init__(self, duration, device_name):
        """Duration: Time to listen for, in milliseconds"""
        self.duration = duration/1000
        self.p = pyaudio.PyAudio()
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.fs = 44100
        self.filename = "output.wav"

        self.device = 0
        for i in range(self.p.get_device_count()):
            if self.p.get_device_info_by_index(i)["name"] == device_name:
                self.device = i

        self.folder_output_path = "./fishing_log"  # The folder for the listener to save its reports to.

    def listen(self):
        # Open a stream
        stream = self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True,
            input_device_index=self.device
        )
        frames = []

        # Record for duration time
        for i in range(0, int(self.fs / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        # Convert from bytes to PCM
        # Values are stored as signed 16 bit integers, in little endian format. This parses them.
        new_frame = []
        for y in range(len(frames)):
            new_frame += self.convert_bytes(frames[y])

        return new_frame

        # Save the recorded data as a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    @staticmethod
    def convert_bytes(_bytes):
        new_frame = []
        for i in range(len(_bytes) // 2):
            sample = _bytes[i * 2:i * 2 + 2]
            int_value = int.from_bytes(sample, byteorder="little", signed=True)
            new_frame.append(int_value)
        return new_frame

    def parse_intervals_overlap(self, file, interval=0.25):
        try:
            shutil.rmtree(self.folder_output_path)  # Clear log folder.
        except FileNotFoundError:
            pass

        wf = wave.open(file, "rb")
        rate = wf.getframerate()
        num_samples = int(rate * (interval / 2))  # Read the audio in chunks of this size.
        previous_audio = wf.readframes(num_samples)

        i = 0
        while True:
            from_time = (interval / 2) * i
            to_time = (interval / 2) * (i + 2)

            curr_audio = wf.readframes(num_samples)
            if not curr_audio:  # End of file
                break
            audio = previous_audio + curr_audio
            audio = FishingListener.convert_bytes(audio)

            yf = rfft(audio)  # Gets the transform. Imaginary numbers.
            xf = rfftfreq(len(audio), 1 / wf.getframerate())  # This calculates the "grouping" of frequencies. Might be able to mess with this for my ranges

            peak1 = get_range_sum(xf, yf, 230, 240)
            peak2 = get_range_sum(xf, yf, 260, 330)
            dip1 = get_range_sum(xf, yf, 245, 255)
            # I want to keep the number of checks minimal so code doesn't become spaghetti.
            # If it doesn't work, find better checks, don't add more.
            if self.threshold_check(peak1, peak2) and self.relative_check(peak1, dip1):
                # If we find the fishing sound, create a graph of this sound so I can validate it.
                plt.axis([150, 600, 0, 400000])
                plt.title(f"{from_time} to {to_time}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude")
                plt.plot(xf, np.abs(yf))
                # Save the audio and analysis of the fishing data.
                self.save_audio(f"{from_time}-{to_time}", audio)
                plt.savefig(f'./{self.folder_output_path}/{from_time}-{to_time}.png')
                plt.close()

            previous_audio = curr_audio
            i += 1

    def threshold_check(self, peak1, peak2):
        """This is a simple "volume" check, used to filter out low levels such as noise. Prebaked thresholds.
        Should be replaced in the future with a more advanced check.

        Arguments:
            Each peak is the sum of the predefined range.
        Returns: True if all thresholds are correct.
        """
        if not peak1 > 250000:
            return False
        if not peak2 > 700000:
            return False

        return True

    def relative_check(self, peak1, dip1):
        """Runs through the checks that describe shapes relative to each other."""
        compar_size = dip1 / peak1
        if not compar_size < 0.75:  # Our "dip" is at most 75% the first peak.
            return False
        return True

    def save_audio(self, name, PCM_data):
        """Saves the given data to a WAV file.
        name: Name of file to save to
        PCM_data: Uncompressed audio data to be saved."""
        # Create dir
        Path(f"{self.folder_output_path}/{name}").parent.mkdir(parents=True, exist_ok=True)

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


def get_range_sum(xf, yf, low, high):
    """Gets the total amplitude of a given range of frequencies.

    Arguments:
        xf: A list of frequencies, generated by rfftfreq
        yf: The amplitudes of frequencies, generated by rfft
        low: Lower bound of frequency to capture
        high: Upper bound of frequency to capture

    returns: The amplitude of the range of frequencies
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
        if xf[i+x1] >= high:
            x2 = i + x1
            break
    if x2 == -1:
        raise ValueError

    return np.sum([abs(x) for x in yf[x1:x2]])


# TODO: For error checking, make the bot log a graph of, and an audio sample of anything that returns a hit.
#  This would allow me to analyse any false positives.

#l = Listener(2000, "Line In (High Definition Audio ")
#frames = l.listen()
#exit()


# peak 1: 220-240
# peak 2: 260-330

# dip 1: 240-260

# TODO: I need to find a way to compare the values of my graph to its surroundings, to detect when a spike has happened.
#  I could set specific number thresholds. This would be easiest, but will also break fastest.
#  I could compare to the last "frame". This scales with volume, but is complex.
#  I could compare the "shape", E.G peak 1 is about 2.5x bigger than peak 2.
#  I could create a normalized standard for the frequency, similar to "last frame".

l = FishingListener(2500, "Line In (High Definition Audio ")
l.parse_intervals_overlap("terr_example_audio.wav")
exit()


# Read audio from wav file.
wf = wave.open("fishing_sound.wav", "rb")
print(wf.getframerate())
frames = wf.readframes(99999999)
frames = Listener.convert_bytes(frames)

yf = rfft(frames)  # Gets the transform. This returns complex numbers.
xf = rfftfreq(len(frames), 1/wf.getframerate())  # This calculates the "grouping" of frequencies. Might be able to mess with this for my ranges.

plt.plot(xf[:100], np.abs(yf)[:100])
plt.show()

exit()


# Number of sample points

N = 600

# sample spacing

T = 1.0 / 800.0

x = np.linspace(0.0, N*T, N, endpoint=False)

y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

yf = fft(y)

xf = fftfreq(N, T)[:N//2]

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

plt.grid()

plt.show()
