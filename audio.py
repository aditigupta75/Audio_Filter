import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter, butter, firwin
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

,j
def design_filter(filter_type='low', cutoff=4000, fs=44100, order=101, use_fir=True):
    if use_fir:
        taps = firwin(order, cutoff, fs=fs, pass_zero=(filter_type == 'low'))
        return taps, None
    else:
        b, a = butter(N=6, Wn=cutoff, fs=fs, btype=filter_type)
        return b, a

def apply_filter(audio, b, a=None):
    return lfilter(b, [1.0] if a is None else a, audio)

def amplify(audio, gain=2.0):
    return np.clip(audio * gain, -1.0, 1.0)

def highpass_noise_removal(audio, fs=44100, cutoff=100):
    b, a = butter(4, cutoff, btype='high', fs=fs)
    return lfilter(b, a, audio)

class AudioApp:
    def __init__(self, master):
        self.master = master
        master.title("Audio Amplifier & Noise Cleaner")

        # ====== Controls Section ======
        # Gain
        tk.Label(master, text="Gain (x):").grid(row=0, column=0)
        self.gain_slider = tk.Scale(master, from_=1, to=10, orient='horizontal', resolution=0.1)
        self.gain_slider.set(2.0)
        self.gain_slider.grid(row=0, column=1)

        # Cutoff
        tk.Label(master, text="Cutoff Frequency (Hz):").grid(row=1, column=0)
        self.cutoff_slider = tk.Scale(master, from_=100, to=10000, orient='horizontal', resolution=100)
        self.cutoff_slider.set(4000)
        self.cutoff_slider.grid(row=1, column=1)

        # Duration
        tk.Label(master, text="Mic Record Duration (s):").grid(row=2, column=0)
        self.duration_slider = tk.Scale(master, from_=1, to=20, orient='horizontal')
        self.duration_slider.set(5)
        self.duration_slider.grid(row=2, column=1)

        # Filter Type
        self.filter_type = tk.StringVar(value="low")
        tk.Label(master, text="Filter Type:").grid(row=3, column=0)
        tk.OptionMenu(master, self.filter_type, "low", "high").grid(row=3, column=1)

        # Use FIR
        self.use_fir = tk.BooleanVar(value=True)
        tk.Checkbutton(master, text="Use FIR Filter", variable=self.use_fir).grid(row=4, column=0, columnspan=2)

        # Noise Reduction
        self.noise_reduction = tk.BooleanVar(value=True)
        tk.Checkbutton(master, text="Noise Reduction", variable=self.noise_reduction).grid(row=5, column=0, columnspan=2)

        # Buttons
        tk.Button(master, text="Run Real-Time", command=self.run_realtime).grid(row=6, column=0)
        tk.Button(master, text="Process File", command=self.process_file).grid(row=6, column=1)
        tk.Button(master, text="Record from Mic", command=self.record_from_mic).grid(row=6, column=2)

        # Status
        self.status_label = tk.Label(master, text="Ready", fg="green")
        self.status_label.grid(row=7, column=0, columnspan=3)

        # Visualization
        self.fig, (self.ax_wave, self.ax_fft) = plt.subplots(2, 1, figsize=(5, 3))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=3)

    def update_status(self, message, color="blue"):
        self.status_label.config(text=message, fg=color)
        self.master.update_idletasks()

    def update_plot(self, audio_chunk, fs):
        self.ax_wave.clear()
        self.ax_fft.clear()

        time = np.linspace(0, len(audio_chunk)/fs, num=len(audio_chunk))
        self.ax_wave.plot(time, audio_chunk)
        self.ax_wave.set_title("Waveform")
        self.ax_wave.set_xlabel("Time [s]")

        freqs = np.fft.rfftfreq(len(audio_chunk), 1/fs)
        fft_vals = np.abs(np.fft.rfft(audio_chunk))
        self.ax_fft.plot(freqs, fft_vals)
        self.ax_fft.set_title("FFT Spectrum")
        self.ax_fft.set_xlabel("Frequency [Hz]")

        self.canvas.draw()

    def get_user_params(self):
        return {
            "cutoff": self.cutoff_slider.get(),
            "gain": self.gain_slider.get(),
            "duration": self.duration_slider.get(),
            "filter_type": self.filter_type.get(),
            "use_fir": self.use_fir.get(),
            "noise_reduction": self.noise_reduction.get()
        }

    def run_realtime(self):
        params = self.get_user_params()
        threading.Thread(target=self.realtime_process_visual, args=(params,), daemon=True).start()

    def realtime_process_visual(self, params):
        fs = 44100
        b, a = design_filter(params["filter_type"], params["cutoff"], fs=fs, use_fir=params["use_fir"])

        def callback(indata, outdata, frames, time, status):
            in_audio = indata[:, 0]
            if params["noise_reduction"]:
                in_audio = highpass_noise_removal(in_audio, fs)
            filtered = apply_filter(in_audio, b, a)
            amplified = amplify(filtered, params["gain"])
            outdata[:, 0] = amplified
            self.update_plot(amplified, fs)

        try:
            self.update_status("Running real-time processing...", "orange")
            with sd.Stream(channels=1, callback=callback, samplerate=fs, blocksize=1024):
                sd.sleep(1000000)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_status("Error occurred", "red")

    def process_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath:
            return
        try:
            params = self.get_user_params()
            audio, fs = sf.read(filepath)
            if audio.ndim > 1:
                audio = audio[:, 0]

            if params["noise_reduction"]:
                audio = highpass_noise_removal(audio, fs)

            b, a = design_filter(params["filter_type"], params["cutoff"], fs, use_fir=params["use_fir"])
            filtered = apply_filter(audio, b, a)
            amplified = amplify(filtered, params["gain"])

            out_path = "processed_output.wav"
            sf.write(out_path, amplified, fs)
            self.update_plot(amplified, fs)

            if messagebox.askyesno("Success", f"Processed file saved as {out_path}\nPlay it now?"):
                sd.play(amplified, fs)

            self.update_status("File processed and saved!", "green")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_status("Error occurred", "red")

    def record_from_mic(self):
        try:
            params = self.get_user_params()
            fs = 44100
            self.update_status(f"Recording for {params['duration']} seconds...", "orange")

            recording = sd.rec(int(params["duration"] * fs), samplerate=fs, channels=1)
            sd.wait()
            audio = recording.flatten()

            if params["noise_reduction"]:
                audio = highpass_noise_removal(audio, fs)

            b, a = design_filter(params["filter_type"], params["cutoff"], fs, use_fir=params["use_fir"])
            filtered = apply_filter(audio, b, a)
            amplified = amplify(filtered, params["gain"])

            out_path = "mic_output.wav"
            sf.write(out_path, amplified, fs)
            self.update_plot(amplified, fs)

            if messagebox.askyesno("Done", f"Saved as {out_path}\nPlay it now?"):
                sd.play(amplified, fs)

            self.update_status("Mic recording processed and saved!", "green")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_status("Error occurred", "red")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
