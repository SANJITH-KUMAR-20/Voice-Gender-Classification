import torchaudio

def process_audio(file_path):
  audio, sr = torchaudio.load(file_path)
  spectrogram = torchaudio.transforms.MelSpectrogram()(audio)
  spectrogram = spectrogram[0,:128,:300]
  return spectrogram
