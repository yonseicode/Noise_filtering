#주기를 fft를 이용한 지배 주파수를 이용해서 반주기를 구한다. 반주기가 지배 주파수의 역수라고 하고, 반주기의 90~110%이내에 최대, 최소쌍이 잡혀야 그 반주기는 살아남음.

import sys
import os
import math
import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import iirnotch, filtfilt, stft
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QComboBox,
    QGroupBox, QTableWidget, QTableWidgetItem, QSplitter,
    QTabWidget, QDoubleSpinBox, QSpinBox, QFormLayout,
    QMessageBox, QProgressBar, QListWidgetItem, QAbstractItemView,
    QHeaderView, QDialog, QTextEdit, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ──────────────────────────────────────────────
# Unit conversion constants
# ──────────────────────────────────────────────
UNIT_SCALE = {'m': 1e0, 'mm': 1e3, 'um': 1e6, 'nm': 1e9, 'pm': 1e12}
UNIT_LABEL = {'m': 'm', 'mm': 'mm', 'um': 'um', 'nm': 'nm', 'pm': 'pm'}

# ──────────────────────────────────────────────
# Help text English
# ──────────────────────────────────────────────
HELP_TEXT_EN = """
<html><body style="font-family:Arial;font-size:10pt;line-height:1.6;">
<h2 style="color:#2e86de;">Analysis Results — Field Descriptions</h2>
<p>Results are split into four tabs.</p><hr>

<h3 style="color:#e74c3c;">Tab 1 | Signal &amp; Noise</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Noise scale factor</b></td>
      <td>RMS(measured)/RMS(noise). Noise is scaled by this factor before subtraction.</td></tr>
  <tr><td><b>RMS – Raw measured</b></td>
      <td>RMS of the original signal (device + noise).</td></tr>
  <tr><td><b>RMS – Noise (scaled)</b></td>
      <td>RMS of noise after scale-factor correction.</td></tr>
  <tr><td><b>RMS – Filtered signal</b></td>
      <td>RMS of the noise-removed signal (pure device vibration).</td></tr>
  <tr><td><b>Noise reduction</b></td>
      <td>RMS(raw) − RMS(filtered). Positive = noise removed.</td></tr>
  <tr><td><b>Noise peak #N</b></td>
      <td>N-th dominant noise frequency detected from noise FFT.</td></tr>
</table><br>

<h3 style="color:#27ae60;">Tab 2 | Amplitude</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Cycles detected</b></td>
      <td>Total number of complete cycles found in the signal.</td></tr>
  <tr><td><b>Amplitude – global max</b></td>
      <td>Largest amplitude across all cycles. max(|y|) per cycle, then max overall.</td></tr>
  <tr><td><b>Amplitude – mean</b></td><td>Mean of per-cycle amplitudes.</td></tr>
  <tr><td><b>Amplitude – median</b></td><td>Median of per-cycle amplitudes.</td></tr>
  <tr><td><b>Amplitude – min</b></td><td>Smallest amplitude across all cycles.</td></tr>
  <tr><td><b>Amplitude – std</b></td><td>Std dev of per-cycle amplitudes.</td></tr>
</table><br>

<h3 style="color:#f39c12;">Tab 3 | Peak-to-Peak</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>P-P – global max</b></td><td>Largest P-P across all cycles.</td></tr>
  <tr><td><b>P-P – mean</b></td><td>Mean of per-cycle P-P values.</td></tr>
  <tr><td><b>P-P – median</b></td><td>Median of per-cycle P-P values.</td></tr>
  <tr><td><b>P-P – min</b></td><td>Smallest P-P across all cycles.</td></tr>
  <tr><td><b>P-P – std</b></td><td>Std dev of per-cycle P-P values.</td></tr>
  <tr><td><b>P-P range</b></td><td>max P-P − min P-P. Spread of amplitude variation.</td></tr>
</table><br>

<h3 style="color:#8e44ad;">Tab 4 | Frequency</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Est. dominant frequency</b></td>
      <td>FFT-based dominant frequency of the filtered signal.</td></tr>
  <tr><td><b>Est. dominant period</b></td>
      <td>1 / dominant frequency.</td></tr>
  <tr><td><b>Noise frequency #N</b></td>
      <td>N-th noise peak frequency detected from noise FFT.</td></tr>
</table><br>

<h3 style="color:#555;">Filter Method Reference</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>Method</th><th>Principle</th><th>Best for</th></tr>
  <tr><td><b>Wiener</b></td>
      <td>H[k]=SNR[k]/(1+SNR[k]) per FFT bin.</td>
      <td>Broadband overlapping noise.</td></tr>
  <tr><td><b>Notch</b></td>
      <td>IIR band-reject at each noise-peak frequency.</td>
      <td>Tonal noise (50/60 Hz harmonics).</td></tr>
  <tr><td><b>STFT</b></td>
      <td>Wiener mask per time-frequency cell.</td>
      <td>Non-stationary noise.</td></tr>
</table>
</body></html>
"""

# ──────────────────────────────────────────────
# Help text Korean
# ──────────────────────────────────────────────
HELP_TEXT_KO = """
<html><body style="font-family:Arial;font-size:10pt;line-height:1.6;">
<h2 style="color:#2e86de;">분석 결과 항목 설명</h2>
<p>결과는 4개의 탭으로 나뉩니다.</p><hr>

<h3 style="color:#e74c3c;">Tab 1 | Signal &amp; Noise</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Noise scale factor</b></td>
      <td>RMS(측정)/RMS(노이즈) 비율. 노이즈를 이 계수로 스케일한 뒤 제거합니다.</td></tr>
  <tr><td><b>RMS – Raw measured</b></td>
      <td>필터링 전 원본 신호(기기+노이즈)의 RMS.</td></tr>
  <tr><td><b>RMS – Noise (scaled)</b></td>
      <td>스케일 보정 후 노이즈 신호의 RMS.</td></tr>
  <tr><td><b>RMS – Filtered signal</b></td>
      <td>노이즈 제거 후 순수 기기 진동 신호의 RMS.</td></tr>
  <tr><td><b>Noise reduction</b></td>
      <td>RMS(원본) - RMS(필터 후). 양수이면 노이즈가 제거된 것.</td></tr>
  <tr><td><b>Noise peak #N</b></td>
      <td>노이즈 FFT에서 감지된 N번째 주요 노이즈 주파수.</td></tr>
</table><br>

<h3 style="color:#27ae60;">Tab 2 | Amplitude (진폭)</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Cycles detected</b></td>
      <td>신호에서 감지된 완전한 주기의 총 개수.</td></tr>
  <tr><td><b>Amplitude – global max</b></td>
      <td>전체 주기 중 가장 큰 진폭.</td></tr>
  <tr><td><b>Amplitude – mean</b></td><td>주기별 진폭의 평균값.</td></tr>
  <tr><td><b>Amplitude – median</b></td><td>주기별 진폭의 중앙값.</td></tr>
  <tr><td><b>Amplitude – min</b></td><td>전체 주기 중 가장 작은 진폭.</td></tr>
  <tr><td><b>Amplitude – std</b></td><td>주기별 진폭의 표준편차.</td></tr>
</table><br>

<h3 style="color:#f39c12;">Tab 3 | Peak-to-Peak</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>P-P – global max</b></td><td>전체 주기 중 가장 큰 P-P 값.</td></tr>
  <tr><td><b>P-P – mean</b></td><td>주기별 P-P의 평균값.</td></tr>
  <tr><td><b>P-P – median</b></td><td>주기별 P-P의 중앙값.</td></tr>
  <tr><td><b>P-P – min</b></td><td>전체 주기 중 가장 작은 P-P 값.</td></tr>
  <tr><td><b>P-P – std</b></td><td>주기별 P-P의 표준편차.</td></tr>
  <tr><td><b>P-P range</b></td><td>최대 P-P - 최소 P-P. 진폭 변동 범위.</td></tr>
</table><br>

<h3 style="color:#8e44ad;">Tab 4 | Frequency (주파수)</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Est. dominant frequency</b></td>
      <td>필터링된 신호의 FFT 기반 지배 주파수.</td></tr>
  <tr><td><b>Est. dominant period</b></td>
      <td>지배 주파수의 역수 (1/f).</td></tr>
  <tr><td><b>Noise frequency #N</b></td>
      <td>노이즈 FFT에서 감지된 N번째 노이즈 주파수.</td></tr>
</table><br>

<h3 style="color:#555;">필터 방법 요약</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%"
 style="border-collapse:collapse;">
  <tr style="background:#dce8f7;"><th>방법</th><th>원리</th><th>적합한 상황</th></tr>
  <tr><td><b>Wiener</b></td><td>H[k]=SNR[k]/(1+SNR[k]).</td><td>광대역 노이즈.</td></tr>
  <tr><td><b>Notch</b></td><td>노이즈 피크 주파수에 IIR 노치 필터.</td><td>특정 주파수 톤 노이즈.</td></tr>
  <tr><td><b>STFT</b></td><td>시간-주파수 셀마다 Wiener 마스크.</td><td>비정상 노이즈.</td></tr>
</table>
</body></html>
"""

# ──────────────────────────────────────────────
# Number formatter: no scientific notation
# ──────────────────────────────────────────────
def fmt(value, decimals=4):
    try:
        if value is None:
            return "N/A"
        v = float(value)
        if not math.isfinite(v):
            return "N/A"
        if v == 0.0:
            return "0." + "0" * decimals
        abs_v = abs(v)
        if abs_v >= 1.0:
            return f"{v:.{decimals}f}"
        else:
            mag   = -int(math.floor(math.log10(abs_v)))
            total = mag + decimals
            return f"{v:.{total}f}"
    except Exception:
        return "N/A"

# ──────────────────────────────────────────────
# File parser
# ──────────────────────────────────────────────
def parse_vibro_file(filepath):
    domain     = 'time'
    x_unit     = 's'
    y_unit     = 'm'
    data_lines = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('Signal:'):
            domain = 'fft' if 'FFT' in stripped else 'time'
        if stripped.startswith('[') and ']' in stripped:
            parts = stripped.split()
            units = [p.strip('[]') for p in parts if p.startswith('[')]
            if len(units) >= 1:
                x_unit = units[0]
            if len(units) >= 2:
                y_unit = units[1]
            data_start = i + 1
            break

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            vals = stripped.split()
            if len(vals) >= 2:
                data_lines.append((float(vals[0]), float(vals[1])))
        except ValueError:
            continue

    if not data_lines:
        raise ValueError(f"Cannot parse data: {filepath}")

    arr = np.array(data_lines)
    return {
        'domain':   domain,
        'x':        arr[:, 0],
        'y':        arr[:, 1],
        'x_unit':   x_unit,
        'y_unit':   y_unit,
        'filename': os.path.basename(filepath),
        'filepath': filepath,
    }

# ──────────────────────────────────────────────
# Vibration analysis engine
# ──────────────────────────────────────────────
class VibrationAnalyzer:
    def __init__(self):
        self.noise_time = None
        self.noise_fft  = None

    def set_noise(self, time_data, fft_data):
        self.noise_time = time_data
        self.noise_fft  = fft_data

    @staticmethod
    def rms(y):
        return float(np.sqrt(np.mean(np.asarray(y, dtype=float) ** 2)))

    def noise_scale_factor(self, measured_y):
        if self.noise_time is None:
            return 1.0
        rms_noise    = self.rms(self.noise_time['y'])
        rms_measured = self.rms(measured_y)
        if rms_noise == 0:
            return 1.0
        return float(rms_measured / rms_noise)

    def detect_noise_peaks(self, threshold_factor=3.0):
        if self.noise_fft is None:
            return []
        freqs     = self.noise_fft['x']
        amps      = self.noise_fft['y']
        threshold = threshold_factor * np.mean(amps)
        peaks, _  = scipy_signal.find_peaks(
            amps,
            height=threshold,
            distance=max(1, len(freqs) // 200))
        return [{'freq': float(freqs[p]),
                 'amplitude': float(amps[p])} for p in peaks]

    @staticmethod
    def estimate_fs(time_x):
        arr   = np.asarray(time_x, dtype=float)
        diffs = np.diff(arr)
        valid = diffs[diffs > 0]
        if len(valid) == 0:
            return 1.0
        return float(1.0 / np.mean(valid))

    # ── Wiener Filter ────────────────────────
    def apply_wiener(self, measured_time, scale_factor=None, gain_floor=0.05):
        y_meas = measured_time['y'].copy().astype(float)
        n      = len(y_meas)

        if self.noise_time is None:
            sf = scale_factor if scale_factor is not None else 1.0
            return y_meas, float(sf)

        if scale_factor is None:
            scale_factor = self.noise_scale_factor(y_meas)

        Y_meas  = np.fft.rfft(y_meas)
        y_noise = self.noise_time['y'].astype(float) * float(scale_factor)
        if len(y_noise) >= n:
            y_noise = y_noise[:n]
        else:
            y_noise = np.pad(y_noise, (0, n - len(y_noise)))

        Y_noise  = np.fft.rfft(y_noise)
        P_noise  = np.abs(Y_noise) ** 2
        P_meas   = np.abs(Y_meas)  ** 2
        P_signal = np.maximum(P_meas - P_noise, 0.0)
        SNR      = P_signal / (P_noise + 1e-30)
        H_wiener = SNR / (1.0 + SNR)
        H_wiener = np.maximum(H_wiener, gain_floor)
        Y_clean  = Y_meas * H_wiener
        y_clean  = np.fft.irfft(Y_clean, n=n)
        return y_clean, float(scale_factor)

    # ── Notch Filter ─────────────────────────
    # returns (y_clean, applied_freqs)
    def apply_notch(self, measured_time, threshold_factor=3.0, q_factor=30.0):
        if self.noise_fft is None:
            raise ValueError("No noise FFT data loaded.")
        y_meas  = measured_time['y'].copy().astype(float)
        fs      = self.estimate_fs(measured_time['x'])
        peaks   = self.detect_noise_peaks(threshold_factor)
        applied = []
        for peak in peaks:
            f0 = peak['freq']
            if f0 <= 0 or f0 >= fs / 2:
                continue
            try:
                b, a   = iirnotch(f0, q_factor, fs)
                y_meas = filtfilt(b, a, y_meas)
                applied.append(f0)
            except Exception:
                continue
        return y_meas, applied

    # ── STFT Denoising ───────────────────────
    # returns (y_clean, f, t, Zmeas, Zclean, scale_factor)
    def apply_stft_denoise(self, measured_time, scale_factor=None,
                           nperseg=256, threshold_factor=3.0):
        if self.noise_time is None:
            raise ValueError("No noise data loaded.")
        y_meas  = measured_time['y'].copy().astype(float)
        y_noise = self.noise_time['y'].copy().astype(float)
        fs      = self.estimate_fs(measured_time['x'])
        n       = min(len(y_meas), len(y_noise))
        y_meas  = y_meas[:n]
        y_noise = y_noise[:n]
        nperseg = min(nperseg, n)
        if scale_factor is None:
            scale_factor = self.noise_scale_factor(y_meas)
        y_noise_scaled = y_noise * float(scale_factor)

        f, t, Zxx_meas  = stft(y_meas,        fs=fs, nperseg=nperseg)
        _, _, Zxx_noise = stft(y_noise_scaled, fs=fs, nperseg=nperseg)

        noise_power = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)
        meas_power  = np.abs(Zxx_meas) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = np.where(
                noise_power > 0,
                np.maximum(
                    (meas_power - noise_power) / (meas_power + 1e-30),
                    0.0),
                1.0)
        Zxx_clean  = Zxx_meas * mask
        _, y_clean = scipy_signal.istft(Zxx_clean, fs=fs, nperseg=nperseg)
        y_clean    = y_clean[:n]
        return y_clean, f, t, np.abs(Zxx_meas), np.abs(Zxx_clean), float(scale_factor)

    # ── Amplitude & Peak-to-Peak : ver5 ──────
    def compute_amplitude_stats(self, y_clean, fs,
                                 peak_prominence_factor=0.1):
        from scipy.signal import find_peaks

        empty = {
            'amplitude_global': None, 'amplitude_mean': None,
            'amplitude_median': None, 'amplitude_min': None,
            'amplitude_std': None,
            'peak_to_peak_global': None, 'peak_to_peak_mean': None,
            'peak_to_peak_median': None, 'peak_to_peak_min': None,
            'peak_to_peak_max': None, 'peak_to_peak_std': None,
            'peaks_pos': np.array([], dtype=int),
            'peaks_neg': np.array([], dtype=int),
            'dominant_freq_est': None,
            'mean_period': None,
            'num_cycles': 0,
            'cycle_amplitudes': [], 'cycle_pp': [],
            'cycle_boundaries': [], 'cycle_corrected': [],
            'cycle_times': [],
        }

        # fs 배열 방어
        if not np.isscalar(fs):
            fs = self.estimate_fs(fs)
        fs = float(fs)

        n = len(y_clean)
        if n < 10 or fs <= 0:
            return empty

        # ── Step 1: Dominant frequency ───────────────────────────────────
        freqs    = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(y_clean))
        spectrum[0] = 0.0
        dom_idx  = int(np.argmax(spectrum))
        if dom_idx == 0:
            return empty
        f_dom  = float(freqs[dom_idx])
        T_est  = fs / f_dom        # samples per period
        half_T = T_est * 0.5
        dist_min = max(1, int(half_T * 0.9))

        # ── Step 2: find_peaks ───────────────────────────────────────────
        prom = peak_prominence_factor * float(np.std(y_clean))
        peak_idx,   _ = find_peaks( y_clean, distance=dist_min, prominence=prom)
        trough_idx, _ = find_peaks(-y_clean, distance=dist_min, prominence=prom)

        if len(peak_idx) < 1 or len(trough_idx) < 1:
            return empty

        # ── Step 3: Pair peaks ↔ troughs ────────────────────────────────
        max_first = peak_idx[0] < trough_idx[0]
        cycles = []
        pi, ti = 0, 0

        while pi < len(peak_idx) and ti < len(trough_idx):
            if max_first:
                p = int(peak_idx[pi])
                cands = trough_idx[trough_idx > p]
                if len(cands) == 0:
                    break
                t_idx = int(cands[0])
                spacing = t_idx - p
                if half_T * 0.9 <= spacing <= half_T / 0.9:
                    cycles.append((p, t_idx))
                ti = int(np.searchsorted(trough_idx, t_idx)) + 1
                pi += 1
            else:
                t_idx = int(trough_idx[ti])
                cands = peak_idx[peak_idx > t_idx]
                if len(cands) == 0:
                    break
                p = int(cands[0])
                spacing = p - t_idx
                if half_T * 0.9 <= spacing <= half_T / 0.9:
                    cycles.append((p, t_idx))
                pi = int(np.searchsorted(peak_idx, p)) + 1
                ti += 1

        if len(cycles) == 0:
            return empty

        # ── Step 4: Midline crossing → boundaries ───────────────────────
        def first_crossing(sig, start, end, level):
            end = min(end, len(sig))
            if start >= end:
                return None
            seg   = sig[start:end]
            above = seg >= level
            for i in range(1, len(above)):
                if above[i] != above[i - 1]:
                    return start + i
            return None

        all_peaks_pos  = []
        all_peaks_neg  = []
        cycle_amps     = []
        cycle_pps      = []
        boundaries     = []
        corrected_segs = []
        cycle_times    = []
        prev_right     = 0

        for (p, t_idx) in cycles:
            y_max = float(y_clean[p])
            y_min = float(y_clean[t_idx])
            y_mid = (y_max + y_min) * 0.5

            left_anchor  = min(p, t_idx)
            right_anchor = max(p, t_idx)

            left_bd = first_crossing(y_clean, prev_right, left_anchor, y_mid)
            if left_bd is None:
                left_bd = prev_right

            right_bd = first_crossing(y_clean, right_anchor, n, y_mid)
            if right_bd is None:
                right_bd = min(right_anchor + int(T_est), n)

            if right_bd <= left_bd + 2:
                prev_right = right_bd
                continue

            # Per-cycle DC removal
            seg           = y_clean[left_bd:right_bd].copy()
            seg_corrected = seg - float(np.mean(seg))

            local_p = int(np.clip(p     - left_bd, 0, len(seg_corrected) - 1))
            local_t = int(np.clip(t_idx - left_bd, 0, len(seg_corrected) - 1))

            amp = float(np.max(np.abs(seg_corrected)))
            pp  = float(np.max(seg_corrected) - np.min(seg_corrected))

            cycle_amps.append(amp)
            cycle_pps.append(pp)
            boundaries.append((left_bd, right_bd))
            corrected_segs.append(seg_corrected)   # 1-D array only
            cycle_times.append((left_bd, right_bd))
            all_peaks_pos.append(left_bd + local_p)
            all_peaks_neg.append(left_bd + local_t)

            prev_right = right_bd

        if len(cycle_amps) == 0:
            return empty

        amps = np.array(cycle_amps)
        pps  = np.array(cycle_pps)

        mean_period = float(np.mean(
            [(r - l) / fs for l, r in boundaries]
        ))

        result = empty.copy()
        result.update({
            'amplitude_global':  float(np.max(amps)),
            'amplitude_mean':    float(np.mean(amps)),
            'amplitude_median':  float(np.median(amps)),
            'amplitude_min':     float(np.min(amps)),
            'amplitude_std':     float(np.std(amps)),
            'peak_to_peak_global':  float(np.max(pps)),
            'peak_to_peak_mean':    float(np.mean(pps)),
            'peak_to_peak_median':  float(np.median(pps)),
            'peak_to_peak_min':     float(np.min(pps)),
            'peak_to_peak_max':     float(np.max(pps)),
            'peak_to_peak_std':     float(np.std(pps)),
            'peaks_pos':         np.array(all_peaks_pos, dtype=int),
            'peaks_neg':         np.array(all_peaks_neg, dtype=int),
            'dominant_freq_est': f_dom,
            'mean_period':       mean_period,
            'num_cycles':        len(cycle_amps),
            'cycle_amplitudes':  cycle_amps,
            'cycle_pp':          cycle_pps,
            'cycle_boundaries':  boundaries,
            'cycle_corrected':   corrected_segs,  # list of 1-D np.ndarray
            'cycle_times':       cycle_times,
        })
        return result


# ──────────────────────────────────────────────
# Background analysis thread
# ──────────────────────────────────────────────
class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, analyzer, measured_time, measured_fft, method, params):
        super().__init__()
        self.analyzer      = analyzer
        self.measured_time = measured_time
        self.measured_fft  = measured_fft
        self.method        = method
        self.params        = params

    def run(self):
        try:
            self.progress.emit(5)
            time_arr = self.measured_time['x']
            fs       = self.analyzer.estimate_fs(time_arr)
            n_meas   = len(self.measured_time['y'])

            scale = self.analyzer.noise_scale_factor(self.measured_time['y'])
            self.progress.emit(15)

            noise_peaks = self.analyzer.detect_noise_peaks(
                threshold_factor=self.params.get('threshold_factor', 3.0))
            self.progress.emit(25)

            method        = self.params.get('method', 'Wiener')
            notch_applied = []
            stft_data     = None

            if method == 'Wiener':
                y_clean, scale = self.analyzer.apply_wiener(
                    self.measured_time,
                    scale_factor=scale,
                    gain_floor=self.params.get('gain_floor', 0.05))

            elif method == 'Notch':
                y_clean, notch_applied = self.analyzer.apply_notch(
                    self.measured_time,
                    threshold_factor=self.params.get('threshold_factor', 3.0),
                    q_factor=self.params.get('q_factor', 30.0))

            elif method == 'STFT':
                y_clean, sf, st_t, Zmeas, Zclean, scale = \
                    self.analyzer.apply_stft_denoise(
                        self.measured_time,
                        scale_factor=scale,
                        nperseg=self.params.get('nperseg', 256),
                        threshold_factor=self.params.get('threshold_factor', 3.0))
                stft_data = {'f': sf, 't': st_t,
                             'Zmeas': Zmeas, 'Zclean': Zclean}
            else:
                y_clean = self.measured_time['y'].copy()

            self.progress.emit(60)

            # ── Build noise array (length-matched) ──────────────────────
            if self.analyzer.noise_time is not None:
                y_n = self.analyzer.noise_time['y'].astype(float).copy()
                if len(y_n) >= n_meas:
                    y_n = y_n[:n_meas]
                else:
                    y_n = np.pad(y_n, (0, n_meas - len(y_n)))
                y_noise = y_n * float(scale)
            else:
                y_noise = np.zeros(n_meas)

            # ── Noise FFT arrays ────────────────────────────────────────
            if self.analyzer.noise_fft is not None:
                noise_freq = self.analyzer.noise_fft['x']
                noise_amp  = self.analyzer.noise_fft['y']
            else:
                noise_freq = None
                noise_amp  = None

            # ── Measured FFT (from provided file or computed) ───────────
            if self.measured_fft is not None:
                fft_freq = self.measured_fft['x']
                fft_amp  = self.measured_fft['y']
            else:
                fft_freq = np.fft.rfftfreq(n_meas, d=1.0 / fs)
                fft_amp  = np.abs(np.fft.rfft(
                    self.measured_time['y'])) * 2.0 / n_meas

            # ── Clean FFT ───────────────────────────────────────────────
            n_clean      = len(y_clean)
            fft_c_freq   = np.fft.rfftfreq(n_clean, d=1.0 / fs)
            fft_c_amp    = np.abs(np.fft.rfft(y_clean)) * 2.0 / n_clean

            self.progress.emit(80)

            amp_stats = self.analyzer.compute_amplitude_stats(
                y_clean, fs,
                peak_prominence_factor=self.params.get(
                    'peak_prominence_factor', 0.1))

            self.progress.emit(95)

            result = {
                # time arrays
                'x_time':      self.measured_time['x'],
                'y_measured':  self.measured_time['y'].astype(float),
                'y_noise':     y_noise,          # scaled, length-matched
                'y_clean':     y_clean,
                # scalars
                'scale_factor': float(scale),
                'fs':           fs,
                'method':       method,
                # noise info
                'noise_peaks':  noise_peaks,
                'noise_freq':   noise_freq,
                'noise_amp':    noise_amp,
                # fft
                'fft_freq':     fft_freq,
                'fft_amp':      fft_amp,
                'fft_c_freq':   fft_c_freq,
                'fft_c_amp':    fft_c_amp,
                # notch
                'notch_applied': notch_applied,
                # stft
                'stft_data':    stft_data,
                # stats
                'amp_stats':    amp_stats,
            }

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ──────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration Analyzer")
        self.setMinimumSize(1400, 900)
        self.analyzer        = VibrationAnalyzer()
        self.noise_time_data = None
        self.noise_fft_data  = None
        self.meas_files      = []
        self.current_result  = None
        self.canvases        = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        splitter.addWidget(self._build_left_panel())
        right_splitter = QSplitter(Qt.Vertical)
        self.tab_graphs = QTabWidget()
        right_splitter.addWidget(self.tab_graphs)
        right_splitter.addWidget(self._build_bottom_panel())
        right_splitter.setSizes([650, 280])
        splitter.addWidget(right_splitter)
        splitter.setSizes([300, 1100])
        self._init_graph_tabs()

    def _build_left_panel(self):
        widget = QWidget()
        widget.setMaximumWidth(320)
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        grp_noise = QGroupBox("Noise Files")
        vn = QVBoxLayout(grp_noise)
        self.btn_load_noise_time = QPushButton("Load Noise Time File")
        self.btn_load_noise_fft  = QPushButton("Load Noise FFT File")
        self.lbl_noise_time = QLabel("Not selected")
        self.lbl_noise_fft  = QLabel("Not selected")
        self.lbl_noise_time.setWordWrap(True)
        self.lbl_noise_fft.setWordWrap(True)
        self.btn_load_noise_time.clicked.connect(self._load_noise_time)
        self.btn_load_noise_fft.clicked.connect(self._load_noise_fft)
        vn.addWidget(self.btn_load_noise_time)
        vn.addWidget(self.lbl_noise_time)
        vn.addWidget(self.btn_load_noise_fft)
        vn.addWidget(self.lbl_noise_fft)
        layout.addWidget(grp_noise)

        grp_meas = QGroupBox("Measurement Files (multi-select)")
        vm = QVBoxLayout(grp_meas)
        btn_row = QHBoxLayout()
        self.btn_load_meas  = QPushButton("Add Files")
        self.btn_clear_meas = QPushButton("Clear List")
        self.btn_load_meas.clicked.connect(self._load_meas_files)
        self.btn_clear_meas.clicked.connect(self._clear_meas_files)
        btn_row.addWidget(self.btn_load_meas)
        btn_row.addWidget(self.btn_clear_meas)
        vm.addLayout(btn_row)
        self.list_meas = QListWidget()
        self.list_meas.setSelectionMode(QAbstractItemView.SingleSelection)
        vm.addWidget(self.list_meas)
        layout.addWidget(grp_meas)

        grp_unit = QGroupBox("Display Unit")
        vu = QFormLayout(grp_unit)
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(['m', 'mm', 'um', 'nm', 'pm'])
        self.combo_unit.setCurrentText('nm')
        vu.addRow("Displacement unit:", self.combo_unit)
        layout.addWidget(grp_unit)

        grp_filter = QGroupBox("Filter Settings")
        vf = QFormLayout(grp_filter)

        self.combo_method = QComboBox()
        self.combo_method.addItems(['Wiener', 'Notch', 'STFT'])
        vf.addRow("Filter method:", self.combo_method)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 20.0)
        self.spin_threshold.setValue(3.0)
        self.spin_threshold.setSingleStep(0.5)
        vf.addRow("Peak threshold (x mean):", self.spin_threshold)

        self.spin_q = QDoubleSpinBox()
        self.spin_q.setRange(1.0, 200.0)
        self.spin_q.setValue(30.0)
        self.spin_q.setSingleStep(5.0)
        vf.addRow("Notch Q factor:", self.spin_q)

        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(32, 4096)
        self.spin_nperseg.setValue(256)
        self.spin_nperseg.setSingleStep(32)
        vf.addRow("STFT nperseg:", self.spin_nperseg)

        self.spin_prominence = QDoubleSpinBox()
        self.spin_prominence.setRange(0.01, 1.0)
        self.spin_prominence.setValue(0.1)
        self.spin_prominence.setSingleStep(0.01)
        self.spin_prominence.setDecimals(2)
        vf.addRow("Peak prominence (x std):", self.spin_prominence)

        layout.addWidget(grp_filter)

        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.setStyleSheet(
            "QPushButton{background:#2e86de;color:white;"
            "font-weight:bold;border-radius:5px;}"
            "QPushButton:hover{background:#1e6fbf;}"
            "QPushButton:disabled{background:#aaa;}")
        self.btn_analyze.clicked.connect(self._run_analysis)
        layout.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        return widget

    def _build_bottom_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title_row = QHBoxLayout()
        lbl = QLabel("Analysis Results")
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        title_row.addWidget(lbl)
        btn_help = QPushButton("?")
        btn_help.setFixedSize(24, 24)
        btn_help.setStyleSheet(
            "QPushButton{background:#2e86de;color:white;"
            "border-radius:12px;font-weight:bold;font-size:13px;}"
            "QPushButton:hover{background:#1e6fbf;}")
        btn_help.clicked.connect(self._show_help_dialog)
        title_row.addWidget(btn_help)
        title_row.addStretch()
        layout.addLayout(title_row)

        self.result_tab = QTabWidget()
        self.result_tab.setStyleSheet("QTabBar::tab{min-width:130px;}")
        self.table_noise     = self._make_result_table()
        self.table_amplitude = self._make_result_table()
        self.table_pp        = self._make_result_table()
        self.table_freq      = self._make_result_table()

        for title, table in [
            ("Signal & Noise", self.table_noise),
            ("Amplitude",      self.table_amplitude),
            ("Peak-to-Peak",   self.table_pp),
            ("Frequency",      self.table_freq),
        ]:
            w  = QWidget()
            tl = QVBoxLayout(w)
            tl.setContentsMargins(0, 0, 0, 0)
            tl.addWidget(table)
            self.result_tab.addTab(w, title)

        layout.addWidget(self.result_tab)
        return widget

    def _make_result_table(self):
        t = QTableWidget()
        t.setColumnCount(4)
        t.setHorizontalHeaderLabels(["Item", "Value", "Unit", "Note"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        return t

    def _fill_table(self, table, rows):
        table.setRowCount(len(rows))
        for i, (item, val, unit, note) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(item)))
            table.setItem(i, 1, QTableWidgetItem(str(val)))
            table.setItem(i, 2, QTableWidgetItem(str(unit)))
            table.setItem(i, 3, QTableWidgetItem(str(note)))

    def _show_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Analysis Results - Description")
        dlg.setMinimumSize(720, 660)
        layout = QVBoxLayout(dlg)
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        combo_lang = QComboBox()
        combo_lang.addItems(["English", "Korean"])
        lang_row.addWidget(combo_lang)
        lang_row.addStretch()
        layout.addLayout(lang_row)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Arial", 9))
        text.setStyleSheet("background:#f8f9fa;")
        text.setHtml(HELP_TEXT_EN)
        layout.addWidget(text)
        combo_lang.currentIndexChanged.connect(
            lambda idx: text.setHtml(
                HELP_TEXT_EN if idx == 0 else HELP_TEXT_KO))
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dlg.close)
        layout.addWidget(btn_box)
        dlg.exec_()

    def _init_graph_tabs(self):
        for title, creator in [
            ("Time Signal",      self._create_time_canvas),
            ("FFT Spectrum",     self._create_fft_canvas),
            ("Filtered Result",  self._create_filtered_canvas),
            ("STFT Spectrogram", self._create_stft_canvas),
        ]:
            canvas, fig = creator()
            self.canvases[title] = (canvas, fig)
            tab_w   = QWidget()
            tl      = QVBoxLayout(tab_w)
            toolbar = NavigationToolbar(canvas, tab_w)
            tl.addWidget(toolbar)
            tl.addWidget(canvas)
            self.tab_graphs.addTab(tab_w, title)

    def _create_time_canvas(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), tight_layout=True)
        fig.patch.set_facecolor('#f8f9fa')
        self._ax_time = axes
        return FigureCanvas(fig), fig

    def _create_fft_canvas(self):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
        fig.patch.set_facecolor('#f8f9fa')
        self._ax_fft = axes
        return FigureCanvas(fig), fig

    def _create_filtered_canvas(self):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
        fig.patch.set_facecolor('#f8f9fa')
        self._ax_filtered = axes
        return FigureCanvas(fig), fig

    def _create_stft_canvas(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        fig.patch.set_facecolor('#f8f9fa')
        self._ax_stft = axes
        return FigureCanvas(fig), fig

    def _load_noise_time(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Noise Time File", "",
            "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            data = parse_vibro_file(path)
            if data['domain'] != 'time':
                QMessageBox.warning(self, "Warning", "Not a Time domain file.")
                return
            self.noise_time_data = data
            self.lbl_noise_time.setText(data['filename'])
            self.lbl_noise_time.setStyleSheet("color:green;")
            self._update_noise()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_noise_fft(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Noise FFT File", "",
            "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            data = parse_vibro_file(path)
            if data['domain'] != 'fft':
                QMessageBox.warning(self, "Warning", "Not an FFT domain file.")
                return
            self.noise_fft_data = data
            self.lbl_noise_fft.setText(data['filename'])
            self.lbl_noise_fft.setStyleSheet("color:green;")
            self._update_noise()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_noise(self):
        self.analyzer.set_noise(self.noise_time_data, self.noise_fft_data)

    def _load_meas_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Measurement Files", "",
            "Text Files (*.txt);;All Files (*)")
        if not paths:
            return
        parsed = []
        for p in paths:
            try:
                parsed.append(parse_vibro_file(p))
            except Exception as e:
                QMessageBox.warning(self, "Parse Error",
                                    f"{os.path.basename(p)}: {e}")
        time_files = [d for d in parsed if d['domain'] == 'time']
        fft_files  = [d for d in parsed if d['domain'] == 'fft']
        for time_d, fft_d, label in self._match_pairs(time_files, fft_files):
            self.meas_files.append((time_d, fft_d, label))
            self.list_meas.addItem(QListWidgetItem(label))

    def _match_pairs(self, time_files, fft_files):
        result, used_fft = [], set()
        for tf in time_files:
            best_match, best_score = None, 0
            tf_base = os.path.splitext(tf['filename'])[0].lower()
            for i, ff in enumerate(fft_files):
                if i in used_fft:
                    continue
                ff_base = os.path.splitext(ff['filename'])[0].lower()
                score   = sum(c in ff_base for c in tf_base)
                if score > best_score:
                    best_score, best_match = score, (i, ff)
            if best_match and best_score > 3:
                used_fft.add(best_match[0])
                result.append((tf, best_match[1], tf['filename']))
            else:
                result.append((tf, None, tf['filename'] + " (no FFT)"))
        return result

    def _clear_meas_files(self):
        self.meas_files.clear()
        self.list_meas.clear()

    def _run_analysis(self):
        row = self.list_meas.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning",
                                "Please select a measurement file.")
            return
        if self.noise_time_data is None:
            QMessageBox.warning(self, "Warning",
                                "Please load a noise time file.")
            return
        time_d, fft_d, _ = self.meas_files[row]
        method = self.combo_method.currentText()
        params = {
            'method':                 method,
            'threshold_factor':       self.spin_threshold.value(),
            'q_factor':               self.spin_q.value(),
            'nperseg':                int(self.spin_nperseg.value()),
            'peak_prominence_factor': self.spin_prominence.value(),
            'gain_floor':             0.05,
        }
        if method in ('Notch', 'STFT') and self.noise_fft_data is None:
            QMessageBox.warning(self, "Warning",
                "Notch/STFT filter requires a noise FFT file.")
            return
        self.btn_analyze.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.thread = AnalysisThread(
            self.analyzer, time_d, fft_d, method, params)
        self.thread.finished.connect(self._on_analysis_done)
        self.thread.error.connect(self._on_analysis_error)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.start()

    def _on_analysis_done(self, result):
        self.current_result = result
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._update_all_plots(result)
        self._update_result_table(result)

    def _on_analysis_error(self, msg):
        self.btn_analyze.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _unit_scale(self):
        return UNIT_SCALE[self.combo_unit.currentText()]

    def _unit_label(self):
        return UNIT_LABEL[self.combo_unit.currentText()]

    def _update_all_plots(self, r):
        self._plot_time(r)
        self._plot_fft(r)
        self._plot_filtered(r)
        if r['method'] == 'STFT' and r.get('stft_data') is not None:
            self._plot_stft(r)

    # ── Plot: Time Signal ─────────────────────
    def _plot_time(self, r):
        canvas, fig = self.canvases["Time Signal"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_time:
            ax.cla()

        n = min(len(r['x_time']), len(r['y_measured']),
                len(r['y_noise']), len(r['y_clean']))
        t       = r['x_time'][:n]
        y_meas  = r['y_measured'][:n] * scale
        y_noise = r['y_noise'][:n]    * scale   # already scaled in run()
        y_clean = r['y_clean'][:n]    * scale

        self._ax_time[0].plot(t, y_meas,  color='#2e86de', lw=0.8,
                              label='Measured Signal')
        self._ax_time[0].set_ylabel(f'Displacement [{ulbl}]')
        self._ax_time[0].set_title('Raw Measured Signal')
        self._ax_time[0].legend(loc='upper right')
        self._ax_time[0].grid(True, alpha=0.3)

        self._ax_time[1].plot(t, y_noise, color='#e74c3c', lw=0.8,
                              label='Scaled Noise')
        self._ax_time[1].set_ylabel(f'Displacement [{ulbl}]')
        self._ax_time[1].set_title(
            f'Noise Signal  (scale factor = {fmt(r["scale_factor"], 4)})')
        self._ax_time[1].legend(loc='upper right')
        self._ax_time[1].grid(True, alpha=0.3)

        self._ax_time[2].plot(t, y_clean, color='#27ae60', lw=0.8,
                              label='Filtered Signal')
        self._ax_time[2].set_xlabel('Time [s]')
        self._ax_time[2].set_ylabel(f'Displacement [{ulbl}]')
        self._ax_time[2].set_title(
            f'Noise-Removed Signal  ({r["method"]} Filter)')
        self._ax_time[2].legend(loc='upper right')
        self._ax_time[2].grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

    # ── Plot: FFT Spectrum ────────────────────
    def _plot_fft(self, r):
        canvas, fig = self.canvases["FFT Spectrum"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_fft:
            ax.cla()

        # Noise FFT
        if r.get('noise_freq') is not None and r.get('noise_amp') is not None:
            self._ax_fft[0].semilogy(
                r['noise_freq'], r['noise_amp'] * scale + 1e-30,
                color='#e74c3c', lw=0.8, label='Noise FFT')
            for pk in r['noise_peaks']:
                self._ax_fft[0].axvline(
                    pk['freq'], color='#e74c3c',
                    linestyle='--', alpha=0.6, lw=1)
                self._ax_fft[0].annotate(
                    f"{fmt(pk['freq'], 1)} Hz",
                    xy=(pk['freq'], pk['amplitude'] * scale),
                    fontsize=7, color='#c0392b',
                    xytext=(5, 5), textcoords='offset points')
        self._ax_fft[0].set_xlabel('Frequency [Hz]')
        self._ax_fft[0].set_ylabel(f'Amplitude [{ulbl}]')
        self._ax_fft[0].set_title('Noise FFT Spectrum (peaks marked)')
        self._ax_fft[0].legend()
        self._ax_fft[0].grid(True, alpha=0.3)

        # Measured FFT
        if r.get('fft_freq') is not None and r.get('fft_amp') is not None:
            self._ax_fft[1].semilogy(
                r['fft_freq'], r['fft_amp'] * scale + 1e-30,
                color='#2e86de', lw=0.8, label='Measured FFT')
        self._ax_fft[1].set_xlabel('Frequency [Hz]')
        self._ax_fft[1].set_ylabel(f'Amplitude [{ulbl}]')
        self._ax_fft[1].set_title('Measured Signal FFT Spectrum')
        self._ax_fft[1].legend()
        self._ax_fft[1].grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

    # ── Plot: Filtered Result ─────────────────
    def _plot_filtered(self, r):
        canvas, fig = self.canvases["Filtered Result"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_filtered:
            ax.cla()

        n = min(len(r['x_time']), len(r['y_measured']), len(r['y_clean']))
        t       = r['x_time'][:n]
        y_meas  = r['y_measured'][:n] * scale
        y_clean = r['y_clean'][:n]    * scale

        self._ax_filtered[0].plot(t, y_meas,  color='#2e86de', lw=0.8,
                                  alpha=0.5, label='Raw Signal', zorder=1)
        self._ax_filtered[0].plot(t, y_clean, color='#27ae60', lw=0.9,
                                  alpha=0.9,
                                  label=f'{r["method"]} Filtered', zorder=2)

        s = r.get('amp_stats')
        if s and s.get('num_cycles', 0) > 0:
            # Draw per-cycle DC-corrected segments
            for idx, (lb, rb) in enumerate(s['cycle_boundaries']):
                seg_corr = s['cycle_corrected'][idx]
                lb = min(lb, n - 1)
                rb = min(rb, n)
                seg_t = t[lb:lb + len(seg_corr)]
                if len(seg_t) == len(seg_corr):
                    self._ax_filtered[0].plot(
                        seg_t, seg_corr * scale,
                        color='#f39c12', lw=0.9, alpha=0.75, zorder=3)

            # Peaks / troughs
            pk_pos = s['peaks_pos']
            pk_pos = pk_pos[pk_pos < n]
            pk_neg = s['peaks_neg']
            pk_neg = pk_neg[pk_neg < n]

            if len(pk_pos) > 0:
                self._ax_filtered[0].scatter(
                    t[pk_pos], y_clean[pk_pos],
                    color='#e74c3c', s=30, zorder=6,
                    label=f'Cycle peaks ({len(pk_pos)})', marker='^')
            if len(pk_neg) > 0:
                self._ax_filtered[0].scatter(
                    t[pk_neg], y_clean[pk_neg],
                    color='#8e44ad', s=30, zorder=6,
                    label=f'Cycle troughs ({len(pk_neg)})', marker='v')

            # Reference lines
            amp_g = s['amplitude_global']  * scale
            pp_g  = s['peak_to_peak_global'] * scale
            pp_m  = s['peak_to_peak_mean']   * scale
            yc    = (float(np.max(y_clean)) + float(np.min(y_clean))) / 2.0

            for yval, clr, ls, lbl in [
                (yc + pp_g / 2, '#f39c12', '--',
                 f'P-P global: {fmt(pp_g, 4)} {ulbl}'),
                (yc - pp_g / 2, '#f39c12', '--', ''),
                (yc + pp_m / 2, '#16a085', '-.',
                 f'P-P mean: {fmt(pp_m, 4)} {ulbl}'),
                (yc - pp_m / 2, '#16a085', '-.', ''),
                (yc + amp_g,    '#e74c3c', ':',
                 f'Amplitude global: {fmt(amp_g, 4)} {ulbl}'),
                (yc - amp_g,    '#e74c3c', ':', ''),
            ]:
                kw = dict(color=clr, linestyle=ls, lw=1.0, alpha=0.8)
                if lbl:
                    kw['label'] = lbl
                self._ax_filtered[0].axhline(yval, **kw)

        self._ax_filtered[0].set_xlabel('Time [s]')
        self._ax_filtered[0].set_ylabel(f'Displacement [{ulbl}]')
        self._ax_filtered[0].set_title('Raw vs Filtered  (Cycle peaks / P-P)')
        self._ax_filtered[0].legend(fontsize=7, loc='upper right')
        self._ax_filtered[0].grid(True, alpha=0.3)

        # FFT comparison
        fs_val = r.get('fs', 1.0 / float(np.mean(np.diff(t))))
        N      = len(y_meas)
        freqs  = np.fft.rfftfreq(N, d=1.0 / fs_val)
        amp_m  = (2 / N) * np.abs(np.fft.rfft(r['y_measured'][:n])) * scale
        amp_c  = (2 / N) * np.abs(np.fft.rfft(r['y_clean'][:n]))    * scale

        self._ax_filtered[1].semilogy(freqs, amp_m + 1e-30,
                                      color='#2e86de', lw=0.8,
                                      alpha=0.6, label='Raw FFT')
        self._ax_filtered[1].semilogy(freqs, amp_c + 1e-30,
                                      color='#27ae60', lw=0.8,
                                      alpha=0.9, label='Filtered FFT')

        if s and s.get('dominant_freq_est') is not None:
            f_dom = s['dominant_freq_est']
            self._ax_filtered[1].axvline(
                f_dom, color='#e74c3c', linestyle='--', lw=1.2, alpha=0.8,
                label=f'Est. dominant: {fmt(f_dom, 2)} Hz')

        self._ax_filtered[1].set_xlabel('Frequency [Hz]')
        self._ax_filtered[1].set_ylabel(f'Amplitude [{ulbl}]')
        self._ax_filtered[1].set_title('FFT Comparison: Raw vs Filtered')
        self._ax_filtered[1].legend(fontsize=8)
        self._ax_filtered[1].grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

    # ── Plot: STFT ────────────────────────────
    def _plot_stft(self, r):
        canvas, fig = self.canvases["STFT Spectrogram"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_stft:
            ax.cla()

        sd     = r['stft_data']
        f      = sd['f']
        t      = sd['t']
        Zmeas  = sd['Zmeas']  * scale
        Zclean = sd['Zclean'] * scale

        pos_vals = Zmeas[Zmeas > 0]
        vmin = float(np.percentile(pos_vals, 5))  if len(pos_vals) > 0 else 1e-12
        vmax = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 0 else 1e-6
        norm = matplotlib.colors.LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, 1e-29))

        im0 = self._ax_stft[0].pcolormesh(t, f, Zmeas,
                                           norm=norm, cmap='inferno',
                                           shading='gouraud')
        self._ax_stft[0].set_title('STFT - Raw')
        self._ax_stft[0].set_xlabel('Time [s]')
        self._ax_stft[0].set_ylabel('Frequency [Hz]')
        fig.colorbar(im0, ax=self._ax_stft[0], label=f'Amplitude [{ulbl}]')

        Zclean_safe = np.where(Zclean > 0, Zclean, vmin)
        im1 = self._ax_stft[1].pcolormesh(t, f, Zclean_safe,
                                           norm=norm, cmap='inferno',
                                           shading='gouraud')
        self._ax_stft[1].set_title('STFT - After Denoising')
        self._ax_stft[1].set_xlabel('Time [s]')
        self._ax_stft[1].set_ylabel('Frequency [Hz]')
        fig.colorbar(im1, ax=self._ax_stft[1], label=f'Amplitude [{ulbl}]')

        fig.tight_layout()
        canvas.draw()

    # ── Result tables ─────────────────────────
    def _update_result_table(self, r):
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        s     = r.get('amp_stats')

        n = min(len(r['y_measured']), len(r['y_clean']), len(r['y_noise']))

        # ── Tab 1 : Signal & Noise ────────────
        rms_meas  = VibrationAnalyzer.rms(r['y_measured'][:n]) * scale
        rms_clean = VibrationAnalyzer.rms(r['y_clean'][:n])    * scale
        rms_noise = VibrationAnalyzer.rms(r['y_noise'][:n])    * scale

        rows_noise = [
            ("Noise scale factor",
             fmt(r['scale_factor'], 6), "-",
             "RMS(measured) / RMS(noise)"),
            ("RMS - Raw measured",
             fmt(rms_meas, 4), ulbl,
             "Energy of raw (device + noise) signal"),
            ("RMS - Noise (scaled)",
             fmt(rms_noise, 4), ulbl,
             "Energy of scaled noise signal"),
            ("RMS - Filtered signal",
             fmt(rms_clean, 4), ulbl,
             "Energy of pure device vibration"),
            ("Noise reduction",
             fmt(rms_meas - rms_clean, 4), ulbl,
             "RMS(raw) - RMS(filtered)"),
            ("Filter method",
             r['method'], "-", ""),
        ]
        for i, pk in enumerate(r['noise_peaks']):
            rows_noise.append((
                f"Noise peak #{i+1}",
                fmt(pk['freq'], 3), "Hz",
                f"Amplitude: {fmt(pk['amplitude'] * scale, 4)} {ulbl}"))
        for f0 in r.get('notch_applied', []):
            rows_noise.append((
                "Notch removed freq.",
                fmt(f0, 3), "Hz", "IIR notch filter applied"))
        self._fill_table(self.table_noise, rows_noise)

        # ── Tab 2 : Amplitude ─────────────────
        rows_amp = []
        if s and s.get('num_cycles', 0) > 0:
            rows_amp = [
                ("Cycles detected",
                 str(s['num_cycles']), "cycles",
                 "Complete cycles found in signal"),
                ("Amplitude - global max",
                 fmt(s['amplitude_global'] * scale, 4), ulbl,
                 "Largest amplitude across all cycles"),
                ("Amplitude - mean",
                 fmt(s['amplitude_mean'] * scale, 4), ulbl,
                 "Mean of per-cycle amplitudes"),
                ("Amplitude - median",
                 fmt(s['amplitude_median'] * scale, 4), ulbl,
                 "Median of per-cycle amplitudes"),
                ("Amplitude - min",
                 fmt(s['amplitude_min'] * scale, 4), ulbl,
                 "Smallest amplitude across all cycles"),
                ("Amplitude - std",
                 fmt(s['amplitude_std'] * scale, 4), ulbl,
                 "Std dev of per-cycle amplitudes"),
            ]
        else:
            rows_amp = [("No cycles detected", "N/A", "-",
                         "Adjust prominence or check signal")]
        self._fill_table(self.table_amplitude, rows_amp)

        # ── Tab 3 : Peak-to-Peak ──────────────
        rows_pp = []
        if s and s.get('num_cycles', 0) > 0:
            rows_pp = [
                ("P-P - global max",
                 fmt(s['peak_to_peak_global'] * scale, 4), ulbl,
                 "Largest P-P across all cycles"),
                ("P-P - mean",
                 fmt(s['peak_to_peak_mean'] * scale, 4), ulbl,
                 "Mean of per-cycle P-P"),
                ("P-P - median",
                 fmt(s['peak_to_peak_median'] * scale, 4), ulbl,
                 "Median of per-cycle P-P"),
                ("P-P - min",
                 fmt(s['peak_to_peak_min'] * scale, 4), ulbl,
                 "Smallest P-P across all cycles"),
                ("P-P - std",
                 fmt(s['peak_to_peak_std'] * scale, 4), ulbl,
                 "Std dev of per-cycle P-P"),
                ("P-P range (max - min)",
                 fmt((s['peak_to_peak_max'] -
                      s['peak_to_peak_min']) * scale, 4),
                 ulbl, "Spread of P-P variation"),
            ]
        else:
            rows_pp = [("No cycles detected", "N/A", "-",
                        "Adjust prominence or check signal")]
        self._fill_table(self.table_pp, rows_pp)

        # ── Tab 4 : Frequency ─────────────────
        rows_freq = []
        if s and s.get('dominant_freq_est') is not None:
            rows_freq += [
                ("Est. dominant frequency",
                 fmt(s['dominant_freq_est'], 4), "Hz",
                 "FFT peak of filtered signal"),
                ("Est. dominant period",
                 fmt(s['mean_period'], 6), "s",
                 "Mean duration of one cycle"),
            ]
        for i, pk in enumerate(r['noise_peaks']):
            rows_freq.append((
                f"Noise frequency #{i+1}",
                fmt(pk['freq'], 3), "Hz",
                f"Noise amplitude: {fmt(pk['amplitude'] * scale, 4)} {ulbl}"))
        for f0 in r.get('notch_applied', []):
            rows_freq.append((
                "Notch filter frequency",
                fmt(f0, 3), "Hz", "Removed by notch filter"))
        self._fill_table(self.table_freq, rows_freq)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
