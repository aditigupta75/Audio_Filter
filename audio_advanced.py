import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import threading
from scipy.signal import butter, lfilter

class SimpleHearingAid:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Hearing Aid")
        self.root.geometry("500x300")
        
        self.frame = ttk.Frame(root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Simple controls
        ttk.Label(self.frame, text="Hearing Aid Controls", font=("Arial", 14)).pack(pady=10)
        
        # Volume control
        ttk.Label(self.frame, text="Volume:").pack(anchor="w")
        self.gain_var = tk.DoubleVar(value=2.0)
        ttk.Scale(self.frame, from_=0.5, to=5.0, variable=self.gain_var).pack(fill="x")
        
        # Clarity control
        ttk.Label(self.frame, text="Speech Clarity:").pack(anchor="w")
        self.clarity_var = tk.DoubleVar(value=0.7)
        ttk.Scale(self.frame, from_=0.0, to=1.0, variable=self.clarity_var).pack(fill="x")
        
        # Start/Stop button
        self.is_processing = False
        self.button = ttk.Button(self.frame, text="Start Hearing Aid", command=self.toggle_processing)
        self.button.pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.frame, textvariable=self.status_var).pack()
        
    def toggle_processing(self):
        if self.is_processing:
            self.is_processing = False
            self.button.config(text="Start Hearing Aid")
            self.status_var.set("Stopped")
        else:
            self.is_processing = True
            self.button.config(text="Stop Hearing Aid")
            self.status_var.set("Processing audio...")
            threading.Thread(target=self.process_audio, daemon=True).start()
    
    def design_bandpass_filter(self, lowcut=300, highcut=5000, fs=44100):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return b, a
    
    def process_audio(self):
        fs = 44100
        blocksize = 1024
        b, a = self.design_bandpass_filter()
        
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print(f"Status: {status}")
            
            audio_in = indata[:, 0] if indata.ndim > 1 else indata[:]
            gain = self.gain_var.get()
            clarity = self.clarity_var.get()
            
            # Apply bandpass filter
            filtered = lfilter(b, a, audio_in)
            
            # Apply speech clarity enhancement
            if clarity > 0:
                speech_b, speech_a = butter(2, [1000/22050, 3000/22050], btype='band')
                enhanced = lfilter(speech_b, speech_a, filtered)
                # Blend original and enhanced based on clarity setting
                filtered = (1-clarity) * filtered + clarity * enhanced * 1.5
            
            # Apply gain and prevent clipping
            processed = np.clip(filtered * gain, -0.99, 0.99)
            
            # Output to all channels
            if outdata.ndim > 1:
                for i in range(outdata.shape[1]):
                    outdata[:, i] = processed
            else:
                outdata[:] = processed
        
        try:
            with sd.Stream(channels=1, callback=audio_callback, 
                          samplerate=fs, blocksize=blocksize,
                          latency='low'):
                while self.is_processing and self.root.winfo_exists():
                    sd.sleep(100)
        except Exception as e:
            print(f"Error: {e}")
            self.is_processing = False
            self.status_var.set(f"Error: {str(e)}")
            self.button.config(text="Start Hearing Aid")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleHearingAid(root)
    root.mainloop()
