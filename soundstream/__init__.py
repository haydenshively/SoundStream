import torch
import torchaudio
from huggingface_hub import hf_hub_download

from soundstream.soundstream import SoundStream


def _infer_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def from_pretrained(
    repo_id='haydenshively/SoundStream',
    filename='soundstream_variant_naturalspeech2.pt',
    device=None,
):
    if device is None:
        device = _infer_device()

    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model_naturalspeech2 = SoundStream(
        n_q=16,
        codebook_size=1024,
        D=256,
        C=58,
        strides=(2, 4, 5, 5),
    )
    model_naturalspeech2.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    return model_naturalspeech2


def load(waveform_path):
    # Load audio from file
    waveform, sample_rate = torchaudio.load(waveform_path)

    # Resample to the frequency the model was trained on
    resampler = torchaudio.transforms.Resample(
        sample_rate,
        16000,
        dtype=waveform.dtype
    )
    waveform = resampler(waveform)
    # Combine channels to get mono audio
    waveform = waveform.mean(dim=0, keepdim=True)

    return torch.unsqueeze(waveform, dim=0)
