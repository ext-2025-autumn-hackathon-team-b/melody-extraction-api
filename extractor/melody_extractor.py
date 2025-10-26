import io
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import librosa
from scipy.signal import medfilt
import mido
import pretty_midi


@dataclass
class ExtractionParams:
    """メロディ抽出のパラメータを保持するデータクラス。"""

    sample_rate: int = 44100
    fmin_hz: float = 65.41  # C2
    fmax_hz: float = 523.25  # C5
    frame_length: int = 2048
    hop_length: int = 256
    voiced_threshold: float = 0.1
    min_note_sec: float = 0.08
    midi_tempo_bpm: int = 120
    smooth_f0: bool = True
    smooth_kernel_size: int = 7
    merge_notes: bool = True
    gap_threshold: float = 0.05
    octave_shift: int = 0


class MelodyExtractor:
    """音声からメロディを抽出し、MIDI化するクラス。"""

    def __init__(self, params: Optional[ExtractionParams] = None):
        """
        Args:
            params: 抽出パラメータ。Noneの場合はデフォルト値を使用。
        """
        self.params = params or ExtractionParams()

    def extract_melody(
        self, audio: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
        """音声からメロディ（f0系列とノートイベント）を抽出。

        Args:
            audio: モノラル音声信号（正規化済み推奨）

        Returns:
            Tuple[f0系列, ノートイベントリスト]
            ノートイベント: List[(midi_note, onset_sec, duration_sec)]
        """
        # 正規化
        if audio.size == 0:
            raise ValueError("音声データが空です")

        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        # pYINによるf0推定
        f0, voiced_flags, voiced_probs = librosa.pyin(
            audio,
            fmin=self.params.fmin_hz,
            fmax=self.params.fmax_hz,
            sr=self.params.sample_rate,
            frame_length=self.params.frame_length,
            hop_length=self.params.hop_length,
        )

        # 無声化
        f0[(voiced_probs < self.params.voiced_threshold) | ~voiced_flags] = np.nan

        # 平滑化
        if self.params.smooth_f0:
            f0 = self._smooth_f0(f0, kernel_size=self.params.smooth_kernel_size)

        # オクターブ移調
        if self.params.octave_shift != 0:
            factor = 2.0 ** float(self.params.octave_shift)
            f0 = np.where(~np.isnan(f0), f0 * factor, np.nan)

        # ノート化
        events = self._f0_to_midi_events(f0)

        # ブレ吸収
        if self.params.merge_notes and len(events) > 0:
            events = self._merge_close_notes(events)

        return f0, events

    def _smooth_f0(self, f0: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """NaNを保ったまま有声区間ごとにメディアンフィルタで平滑化。

        Args:
            f0: 周波数系列（NaNは無声区間）
            kernel_size: メディアンフィルタのカーネルサイズ（大きいほど強く平滑化）
        """
        f0_smoothed = f0.copy()
        if f0_smoothed.size == 0:
            return f0_smoothed

        voiced_mask = ~np.isnan(f0_smoothed)
        if not np.any(voiced_mask):
            return f0_smoothed

        idx = np.where(voiced_mask)[0]
        splits: List[Tuple[int, int]] = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                splits.append((start, prev))
                start = i
            prev = i
        splits.append((start, prev))

        for s, e in splits:
            segment = f0_smoothed[s : e + 1]
            segment_len = e - s + 1
            # セグメントの長さに応じてカーネルサイズを調整
            k = min(kernel_size, segment_len)
            if k < 3:
                k = 3 if segment_len >= 3 else segment_len
            if k % 2 == 0:  # 奇数にする
                k = k - 1
            if k < 3:
                continue
            f0_smoothed[s : e + 1] = medfilt(segment, kernel_size=k)

        return f0_smoothed

    def _f0_to_midi_events(self, f0: np.ndarray) -> List[Tuple[int, float, float]]:
        """f0系列をノートイベントに変換。

        Returns:
            List[(midi_note, onset_sec, duration_sec)]
        """
        midi_float = librosa.hz_to_midi(f0)  # NaNはNaNのまま
        midi_rounded = np.round(midi_float)

        events: List[Tuple[int, float, float]] = []
        n_frames = len(midi_rounded)

        def frame_to_time(fr: int) -> float:
            return fr * self.params.hop_length / self.params.sample_rate

        run_note: Optional[int] = None
        run_start = 0

        for i in range(n_frames + 1):
            cur = midi_rounded[i] if i < n_frames else np.nan
            if np.isnan(cur):
                cur_note = None
            else:
                cur_note = int(np.clip(cur, 0, 127))

            if run_note is None and cur_note is not None:
                run_note = cur_note
                run_start = i
            elif run_note is not None and (cur_note != run_note):
                onset = frame_to_time(run_start)
                offset = frame_to_time(i)
                dur = offset - onset
                if dur >= self.params.min_note_sec:
                    events.append((run_note, onset, dur))
                run_note = cur_note
                run_start = i
            else:
                pass

        return events

    def _merge_close_notes(
        self, events: List[Tuple[int, float, float]]
    ) -> List[Tuple[int, float, float]]:
        """同じ音高で近接するノートを結合してブレを吸収。

        Args:
            events: [(midi_note, onset_sec, duration_sec)]

        Returns:
            マージ後のイベントリスト
        """
        if len(events) == 0:
            return events

        # onsetでソート
        sorted_events = sorted(events, key=lambda x: x[1])
        merged: List[Tuple[int, float, float]] = []

        current_note, current_onset, current_duration = sorted_events[0]
        current_end = current_onset + current_duration

        for i in range(1, len(sorted_events)):
            next_note, next_onset, next_duration = sorted_events[i]
            next_end = next_onset + next_duration

            # 同じ音高で、間隔がthreshold以下なら結合
            if (
                next_note == current_note
                and (next_onset - current_end) <= self.params.gap_threshold
            ):
                # 現在のノートを延長（次のノートの終わりまで）
                current_end = max(current_end, next_end)
                current_duration = current_end - current_onset
            else:
                # 現在のノートを確定して次へ
                merged.append((current_note, current_onset, current_duration))
                current_note = next_note
                current_onset = next_onset
                current_duration = next_duration
                current_end = next_end

        # 最後のノートを追加
        merged.append((current_note, current_onset, current_duration))

        return merged

    def events_to_midi_bytes(self, events: List[Tuple[int, float, float]]) -> bytes:
        """ノートイベントをMIDIファイルのバイト列に変換。

        Args:
            events: [(midi_note, onset_sec, duration_sec)]

        Returns:
            MIDIファイルのバイト列
        """
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        tempo = mido.bpm2tempo(self.params.midi_tempo_bpm)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        cur_time_ticks = 0
        for note, onset, dur in events:
            on_ticks = mido.second2tick(onset, mid.ticks_per_beat, tempo)
            dur_ticks = mido.second2tick(dur, mid.ticks_per_beat, tempo)

            delta_on = int(round(on_ticks - cur_time_ticks))
            if delta_on < 0:
                delta_on = 0
            track.append(mido.Message("note_on", note=note, velocity=90, time=delta_on))
            track.append(
                mido.Message(
                    "note_off", note=note, velocity=64, time=int(round(dur_ticks))
                )
            )

            cur_time_ticks = on_ticks + dur_ticks

        buf = io.BytesIO()
        mid.save(file=buf)
        return buf.getvalue()

    def events_to_piano_wav(self, events: List[Tuple[int, float, float]]) -> np.ndarray:
        """MIDIノートイベントをピアノ音源で合成。

        Args:
            events: [(midi_note, onset_sec, duration_sec)]

        Returns:
            合成された音声波形（float32）
        """
        # PrettyMIDIオブジェクトを作成
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.params.midi_tempo_bpm)

        # ピアノ音源（Acoustic Grand Piano = 0）
        piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
        piano = pretty_midi.Instrument(program=piano_program)

        # ノートイベントを追加
        for note_number, onset, duration in events:
            note = pretty_midi.Note(
                velocity=90,
                pitch=int(note_number),
                start=onset,
                end=onset + duration,
            )
            piano.notes.append(note)

        pm.instruments.append(piano)

        # オーディオ合成
        audio = pm.fluidsynth(fs=self.params.sample_rate)

        return audio

    def resynthesize_from_f0(self, f0: np.ndarray) -> np.ndarray:
        """フレーム単位のf0からサイン波で簡易再合成。

        Args:
            f0: 周波数系列（NaNは無声区間）

        Returns:
            再合成された音声波形（float32）
        """
        f0_playback = np.nan_to_num(f0, nan=0.0)
        waveform = []
        phase = 0.0
        for freq in f0_playback:
            t = np.arange(self.params.hop_length) / self.params.sample_rate
            angular = 2.0 * np.pi * freq
            chunk = np.sin(angular * t + phase)
            phase = (
                angular * self.params.hop_length / self.params.sample_rate + phase
            ) % (2.0 * np.pi)
            if freq == 0.0:
                chunk[:] = 0.0
            waveform.append(chunk)
        out = np.concatenate(waveform).astype(np.float32)
        out *= 0.5
        return out
