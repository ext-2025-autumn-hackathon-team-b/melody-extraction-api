import base64
import io

import numpy as np
from scipy.io import wavfile


def waveform_to_wav_bytes(signal: np.ndarray, sr: int) -> bytes:
    """float32のモノラル信号を16bit PCM WAVのバイト列へ。

    Args:
        signal: 音声信号（float32, -1.0〜1.0想定）
        sr: サンプリングレート

    Returns:
        WAVファイルのバイト列
    """
    y16 = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, y16)
    return buf.getvalue()


def bytes_to_data_url(data: bytes, mime: str = "audio/wav") -> str:
    """バイト列をdata URLにエンコードして返す。

    Cloud Run上でのインスタンス跨ぎによるメディア配信不整合を回避するため、
    音声バイト列をページ内に直接埋め込む用途に使います。

    Args:
        data: エンコードするバイト列
        mime: MIMEタイプ

    Returns:
        data URL文字列
    """
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
