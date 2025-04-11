import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import threading
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🦻 Smart Hearing Aid Console")
        self.root.geometry("1000x720")
        self.root.configure(bg='#eef2f7')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Segoe UI', 12, 'bold'))
        style.configure('TButton', font=('Segoe UI', 11), padding=6)
        style.configure('TLabel', font=('Segoe UI', 11))

        self.notebook = ttk.Notebook(root)
        self.tab_realtime = ttk.Frame(self.notebook)
        self.tab_file = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_realtime, text='🎧 Real-Time')
        self.notebook.add(self.tab_file, text='📂 File Processing')
        self.notebook.add(self.tab_settings, text='⚙️ Settings')
        self.notebook.pack(fill='both', expand=True)

        self.setup_realtime_tab()
        self.setup_file_tab()
        self.setup_settings_tab()

    def setup_realtime_tab(self):
        frame = self.tab_realtime

        ttk.Label(frame, text="🔊 Real-Time Audio Enhancer", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        self.gain_slider = self.make_slider(frame, "Gain", 1, 6, 3.0, row=1)

        self.noise_gate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Enable Noise Gate", variable=self.noise_gate_var).grid(row=2, column=0, columnspan=2, pady=5)

        self.record_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Record Output", variable=self.record_var).grid(row=3, column=0, columnspan=2, pady=5)

        self.fig, self.ax_wave = plt.subplots(figsize=(7, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, padx=10, pady=10, columnspan=2)

        ttk.Button(frame, text="▶ Start Real-Time Audio", command=self.realtime_process).grid(row=5, column=0, columnspan=2, pady=20)

    def setup_file_tab(self):
        frame = self.tab_file

        ttk.Label(frame, text="📁 Process Audio File", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        self.file_gain_slider = self.make_slider(frame, "Gain", 1, 6, 3.0, row=1)

        ttk.Button(frame, text="📂 Select Audio File", command=self.process_file).grid(row=2, column=0, columnspan=2, pady=10)

        self.fig_file, self.ax_file_wave = plt.subplots(figsize=(7, 3))
        self.canvas_file = FigureCanvasTkAgg(self.fig_file, master=frame)
        self.canvas_file.get_tk_widget().grid(row=3, column=0, padx=10, pady=10, columnspan=2)

    def setup_settings_tab(self):
        frame = self.tab_settings
        ttk.Label(frame, text="⚙️ Settings Panel", font=("Segoe UI", 14, "bold")).pack(pady=20)
        ttk.Label(frame, text="More customization options coming soon!").pack()

    def make_slider(self, frame, label, minval, maxval, default, row):
        ttk.Label(frame, text=label + f" ({default})").grid(row=row, column=0, sticky='w', padx=10)
        slider = ttk.Scale(frame, from_=minval, to=maxval, orient='horizontal')
        slider.set(default)
        slider.grid(row=row, column=1, sticky='ew', padx=10)
        return slider

    def design_iir_filter(self, filter_type='band', cutoff=[150, 3500], fs=44100):
        b, a = butter(N=6, Wn=cutoff, fs=fs, btype=filter_type)
        return b, a

    def noise_gate(self, signal, threshold=0.015):
        return np.where(np.abs(signal) < threshold, 0, signal)

    def spectral_subtraction(self, audio, noise_estimate):
        spectrum = np.fft.rfft(audio)
        noise_spectrum = np.fft.rfft(noise_estimate)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        clean_magnitude = np.maximum(magnitude - np.abs(noise_spectrum), 0)
        cleaned_signal = np.fft.irfft(clean_magnitude * np.exp(1j * phase))
        return cleaned_signal

    def normalize_audio(self, audio):
        return audio / (np.max(np.abs(audio)) + 1e-6)

    def process_audio(self, audio, b, a, gain, use_gate=False):
        audio = self.normalize_audio(audio)
        filtered = lfilter(b, a, audio)
        if use_gate:
            filtered = self.noise_gate(filtered)
        amplified = np.clip(filtered * gain, -1.0, 1.0)
        return amplified

    def realtime_process(self):
        fs = 44100
        blocksize = 512
        b, a = self.design_iir_filter()
        self.realtime_buffer = np.zeros(fs)
        self.recorded_audio = []
        noise_profile = np.zeros(blocksize)

        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print("Status:", status)
            audio_in = indata[:, 0]
            gain = float(self.gain_slider.get())
            use_gate = self.noise_gate_var.get()
            record = self.record_var.get()
            cleaned = self.spectral_subtraction(audio_in, noise_profile)
            audio_out = self.process_audio(cleaned, b, a, gain, use_gate=use_gate)
            outdata[:, 0] = audio_out
            self.realtime_buffer = np.roll(self.realtime_buffer, -frames)
            self.realtime_buffer[-frames:] = audio_out
            if record:
                self.recorded_audio.append(audio_out.copy())

        def update_plot():
            if not self.root.winfo_exists():
                return
            self.ax_wave.clear()
            self.ax_wave.plot(self.realtime_buffer, color='blue')
            self.ax_wave.set_title("Live Waveform")
            self.ax_wave.set_ylim([-1.1, 1.1])
            try:
                self.canvas.draw()
            except tk.TclError:
                return
            self.root.after(100, update_plot)

        def stream_thread():
            with sd.Stream(channels=1, callback=audio_callback, samplerate=fs, blocksize=blocksize, latency='low'):
                sd.sleep(1000000)
                if self.record_var.get():
                    full_audio = np.concatenate(self.recorded_audio)
                    sf.write("realtime_recorded.wav", full_audio, fs)
                    print("Saved recorded audio.")

        threading.Thread(target=stream_thread, daemon=True).start()
        self.root.after(100, update_plot)

    def process_file(self):
        file_path = filedialog.askopenfilename(filetypes=[["WAV files", "*.wav"]])
        if not file_path:
            return
        try:
            audio, fs = sf.read(file_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            b, a = self.design_iir_filter()
            gain = self.file_gain_slider.get()
            processed_audio = self.process_audio(audio, b, a, gain, use_gate=True)
            sf.write("processed_output.wav", processed_audio, fs)
            messagebox.showinfo("Success", "File processed and saved as 'processed_output.wav'")

            self.ax_file_wave.clear()
            self.ax_file_wave.plot(processed_audio, color='green')
            self.ax_file_wave.set_title("Processed File Waveform")
            self.ax_file_wave.set_ylim([-1.1, 1.1])
            self.canvas_file.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == '__main__':
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
