# HELP 한국어 버전까지 추가하고 결과값을 보기 쉽게 탭별로 정리
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
UNIT_SCALE = {
    'm':  1e0,
    'mm': 1e3,
    'um': 1e6,
    'nm': 1e9,
    'pm': 1e12,
}
UNIT_LABEL = {
    'm':  'm',
    'mm': 'mm',
    'um': 'um',
    'nm': 'nm',
    'pm': 'pm',
}


# ──────────────────────────────────────────────
# Help text (English)
# ──────────────────────────────────────────────
HELP_TEXT_EN = """
<html><body style="font-family: Arial; font-size: 10pt; line-height: 1.6;">
<h2 style="color:#2e86de;">Analysis Results — Field Descriptions</h2>
<p>Results are split into four tabs. Each tab focuses on a specific aspect of the vibration analysis.</p>
<hr>

<h3 style="color:#e74c3c;">Tab 1 | Signal &amp; Noise</h3>
<p>This tab summarises how much noise was present and how effectively it was removed from the measured signal.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Noise scale factor</b></td><td>Ratio of RMS(measured) to RMS(noise). The noise file is multiplied by this factor before subtraction so that its energy level matches the measurement environment.</td></tr>
  <tr><td><b>RMS – Raw measured</b></td><td>Root-mean-square of the original signal before any filtering. Represents the overall energy level of the combined (device + noise) signal.</td></tr>
  <tr><td><b>RMS – Noise (scaled)</b></td><td>RMS of the noise signal after applying the scale factor. Ideally this should be close to the noise contribution inside the measured signal.</td></tr>
  <tr><td><b>RMS – Filtered signal</b></td><td>RMS of the signal after noise removal. This represents the pure device vibration energy.</td></tr>
  <tr><td><b>Noise reduction</b></td><td>RMS(raw) − RMS(filtered). A positive value confirms that noise energy was successfully removed.</td></tr>
  <tr><td><b>Filter method</b></td><td>The denoising algorithm that was applied (Wiener, Notch, or STFT).</td></tr>
  <tr><td><b>Noise peak #N</b></td><td>Frequency of the N-th dominant peak detected in the noise FFT. Peaks are identified where amplitude exceeds threshold × mean(noise FFT). Amplitude is shown in the Note column.</td></tr>
  <tr><td><b>Notch removed frequency</b></td><td>(Notch filter only) Frequency at which an IIR notch filter was applied. The Q-factor controls the rejection bandwidth.</td></tr>
</table><br>

<h3 style="color:#27ae60;">Tab 2 | Amplitude</h3>
<p>Displacement amplitude of the <b>noise-removed</b> device signal only.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Amplitude – global max|y|</b></td><td>The single largest absolute displacement in the entire filtered signal: max(|y|). This is the worst-case displacement from equilibrium, useful for structural safety assessments.</td></tr>
  <tr><td><b>Amplitude – peak mean</b></td><td>Average magnitude of all individually detected peaks (both positive and negative). For steady sinusoidal vibration this equals the half-amplitude.</td></tr>
  <tr><td><b>Amplitude – peak std</b></td><td>Standard deviation of peak magnitudes. Small value = consistent vibration. Large value = amplitude modulation or transient events.</td></tr>
  <tr><td><b>Positive / Negative peaks count</b></td><td>Total peaks detected above the prominence threshold. Roughly equal counts indicate symmetric oscillation. Unequal counts may suggest asymmetric motion or DC offset.</td></tr>
</table><br>

<h3 style="color:#f39c12;">Tab 3 | Peak-to-Peak</h3>
<p>P-P displacement of the <b>noise-removed</b> signal. P-P = positive peak − nearest negative peak, representing the full swing of one vibration cycle.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>P-P – global</b></td><td>max(y) − min(y) over the entire signal. Most conservative estimate. Sensitive to a single outlier or transient spike.</td></tr>
  <tr><td><b>P-P – pair mean</b></td><td>Mean P-P computed by pairing each positive peak with its nearest negative peak. More representative of steady-state vibration amplitude.</td></tr>
  <tr><td><b>P-P – pair std</b></td><td>Standard deviation of per-cycle P-P values. Low std = consistent vibration. High std = varying or beating vibration.</td></tr>
  <tr><td><b>P-P – maximum</b></td><td>Largest P-P value among all detected cycles. Useful for identifying peak mechanical stress condition.</td></tr>
  <tr><td><b>P-P – minimum</b></td><td>Smallest P-P value among all detected cycles. Useful for identifying the quietest operating instant.</td></tr>
  <tr><td><b>P-P range (max − min)</b></td><td>Spread between largest and smallest cycle P-P. Large range indicates significant amplitude variation (beating, modulation, or intermittent excitation).</td></tr>
</table><br>

<h3 style="color:#8e44ad;">Tab 4 | Frequency</h3>
<p>Frequency information extracted from the filtered device signal and from the noise spectrum.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>Item</th><th>Description</th></tr>
  <tr><td><b>Est. dominant frequency</b></td><td>Frequency estimated from mean interval between consecutive positive peaks in the filtered signal. For a pure sinusoid this matches the excitation frequency.</td></tr>
  <tr><td><b>Est. dominant period</b></td><td>Reciprocal of dominant frequency (1/f). Average duration of one vibration cycle.</td></tr>
  <tr><td><b>Noise frequency #N</b></td><td>N-th dominant frequency found in noise FFT. External disturbances (e.g. power-line harmonics, bench resonances) present even when device is off.</td></tr>
  <tr><td><b>Notch filter frequency</b></td><td>(Notch filter only) Frequency at which the IIR notch was placed. Corresponds to a detected noise peak.</td></tr>
</table><br>

<h3 style="color:#555;">Filter Method Reference</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>Method</th><th>How it works</th><th>Best used when</th></tr>
  <tr><td><b>Wiener</b></td><td>Computes H[k] = SNR[k]/(1+SNR[k]) for each FFT bin. Low-SNR bins are suppressed; high-SNR bins are kept.</td><td>Broadband noise; noise and signal overlap in frequency.</td></tr>
  <tr><td><b>Notch</b></td><td>Places a narrow IIR band-reject filter at each detected noise peak frequency.</td><td>Discrete tonal noise (e.g. 50/60 Hz mains, specific harmonics).</td></tr>
  <tr><td><b>STFT</b></td><td>Applies a Wiener-like mask in each time-frequency cell and reconstructs via inverse STFT.</td><td>Non-stationary noise that changes over time.</td></tr>
</table>
</body></html>
"""

# ──────────────────────────────────────────────
# Help text (Korean)
# ──────────────────────────────────────────────
HELP_TEXT_KO = """
<html><body style="font-family: Arial; font-size: 10pt; line-height: 1.6;">
<h2 style="color:#2e86de;">분석 결과 항목 설명</h2>
<p>결과는 4개의 탭으로 나뉩니다. 각 탭은 진동 분석의 특정 측면에 집중합니다.</p>
<hr>

<h3 style="color:#e74c3c;">Tab 1 | Signal &amp; Noise (신호 및 노이즈)</h3>
<p>측정 신호에 얼마나 많은 노이즈가 포함되었는지, 얼마나 효과적으로 제거되었는지를 요약합니다.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Noise scale factor</b></td><td>RMS(측정 신호) / RMS(노이즈 신호) 비율입니다. 노이즈 파일에 이 계수를 곱해 측정 환경의 에너지 수준에 맞게 보정한 후 제거합니다. 전압이나 접지 상태에 따라 노이즈 크기가 달라지기 때문에 이 보정이 필요합니다.</td></tr>
  <tr><td><b>RMS – Raw measured</b></td><td>필터링 전 원본 신호의 RMS(제곱평균제곱근)입니다. 전자기기 진동과 노이즈가 합쳐진 신호의 전체 에너지 수준을 나타냅니다.</td></tr>
  <tr><td><b>RMS – Noise (scaled)</b></td><td>스케일 보정 후 노이즈 신호의 RMS입니다. 이 값이 측정 신호 내 노이즈 기여도에 가까울수록 보정이 정확합니다.</td></tr>
  <tr><td><b>RMS – Filtered signal</b></td><td>노이즈 제거 후 신호의 RMS입니다. 순수한 전자기기 진동 에너지를 나타냅니다.</td></tr>
  <tr><td><b>Noise reduction</b></td><td>RMS(원본) - RMS(필터 후) 값입니다. 양수이면 노이즈 에너지가 성공적으로 제거된 것입니다.</td></tr>
  <tr><td><b>Filter method</b></td><td>적용된 노이즈 제거 알고리즘입니다 (Wiener, Notch, STFT 중 하나).</td></tr>
  <tr><td><b>Noise peak #N</b></td><td>노이즈 FFT에서 감지된 N번째 주요 주파수 성분입니다. 평균 진폭의 임계값 배수 이상인 피크를 노이즈 주파수로 판별합니다. 해당 진폭은 Note 열에 표시됩니다.</td></tr>
  <tr><td><b>Notch removed frequency</b></td><td>(Notch 필터 전용) IIR 노치 필터가 적용된 주파수입니다. Q-factor가 클수록 제거 대역폭이 좁아집니다.</td></tr>
</table><br>

<h3 style="color:#27ae60;">Tab 2 | Amplitude (진폭)</h3>
<p><b>노이즈가 제거된</b> 순수 전자기기 신호의 변위 진폭입니다.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Amplitude – global max|y|</b></td><td>필터링된 신호 전체에서 절댓값이 가장 큰 변위값: max(|y|)입니다. 평형 위치로부터 가장 큰 단방향 변위이며, 구조적 안전성 평가에 활용됩니다.</td></tr>
  <tr><td><b>Amplitude – peak mean</b></td><td>감지된 모든 양/음의 피크 크기의 평균입니다. 순수 정현파 진동에서는 반진폭(half-amplitude)과 같습니다. 불규칙 진동에서는 평균적인 변위 크기를 나타냅니다.</td></tr>
  <tr><td><b>Amplitude – peak std</b></td><td>피크 크기의 표준편차입니다. 작으면 진폭이 시간에 따라 일정한 것이고, 크면 진폭 변조나 일시적 충격이 있는 것입니다.</td></tr>
  <tr><td><b>Positive / Negative peaks count</b></td><td>prominence 임계값 이상으로 감지된 양/음의 피크 개수입니다. 두 값이 비슷하면 대칭적 진동, 차이가 크면 비대칭 운동이나 DC 오프셋을 의심할 수 있습니다.</td></tr>
</table><br>

<h3 style="color:#f39c12;">Tab 3 | Peak-to-Peak (피크-피크)</h3>
<p><b>노이즈가 제거된</b> 신호의 P-P 변위입니다. P-P = 양의 피크 - 가장 가까운 음의 피크로, 한 진동 사이클의 전체 진폭 범위를 나타냅니다.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>P-P – global</b></td><td>전체 신호에서 max(y) - min(y) 값입니다. 가장 보수적(큰) 추정값으로, 단 하나의 이상값이나 충격에도 영향을 받습니다.</td></tr>
  <tr><td><b>P-P – pair mean</b></td><td>각 양의 피크와 가장 가까운 음의 피크를 쌍으로 묶어 계산한 P-P의 평균입니다. 정상 상태 진동 진폭을 가장 잘 대표하는 값입니다.</td></tr>
  <tr><td><b>P-P – pair std</b></td><td>사이클별 P-P 값의 표준편차입니다. 작으면 진동이 안정적, 크면 맥놀이(beating)나 진폭 변동이 있는 것입니다.</td></tr>
  <tr><td><b>P-P – maximum</b></td><td>감지된 모든 사이클 중 가장 큰 P-P 값입니다. 최대 기계적 응력 조건 파악에 사용됩니다.</td></tr>
  <tr><td><b>P-P – minimum</b></td><td>감지된 모든 사이클 중 가장 작은 P-P 값입니다. 가장 조용한 동작 순간을 파악하는 데 사용됩니다.</td></tr>
  <tr><td><b>P-P range (max - min)</b></td><td>최대 P-P와 최소 P-P의 차이입니다. 값이 크면 맥놀이, 변조, 간헐적 가진 등 진폭이 크게 변동하는 것을 의미합니다.</td></tr>
</table><br>

<h3 style="color:#8e44ad;">Tab 4 | Frequency (주파수)</h3>
<p>필터링된 장치 신호와 노이즈 스펙트럼에서 추출한 주파수 정보입니다.</p>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>항목</th><th>설명</th></tr>
  <tr><td><b>Est. dominant frequency</b></td><td>필터링된 신호에서 연속된 양의 피크 간격의 평균으로 추정한 지배 주파수입니다. 순수 정현파에서는 인가 주파수와 일치합니다.</td></tr>
  <tr><td><b>Est. dominant period</b></td><td>지배 주파수의 역수(1/f)입니다. 한 진동 사이클의 평균 지속 시간입니다.</td></tr>
  <tr><td><b>Noise frequency #N</b></td><td>노이즈 FFT에서 발견된 N번째 주요 주파수입니다. 전자기기가 꺼져 있어도 존재하는 외부 노이즈(예: 전원선 고조파, 테스트 벤치 공진)입니다.</td></tr>
  <tr><td><b>Notch filter frequency</b></td><td>(Notch 필터 전용) IIR 노치가 배치된 주파수입니다. 감지된 노이즈 피크 주파수와 대응됩니다.</td></tr>
</table><br>

<h3 style="color:#555;">필터 방법 요약</h3>
<table border="1" cellpadding="5" cellspacing="0" width="100%" style="border-collapse:collapse;">
  <tr style="background-color:#dce8f7;"><th>방법</th><th>동작 원리</th><th>적합한 상황</th></tr>
  <tr><td><b>Wiener</b></td><td>각 FFT 빈마다 H[k] = SNR[k]/(1+SNR[k]) 이득을 계산합니다. SNR이 낮은 빈은 억제되고 높은 빈은 유지됩니다.</td><td>광대역 노이즈, 노이즈와 신호가 주파수 영역에서 겹치는 경우.</td></tr>
  <tr><td><b>Notch</b></td><td>감지된 각 노이즈 피크 주파수에 좁은 IIR 대역 저지 필터를 배치합니다.</td><td>특정 주파수의 이산 톤 노이즈 (예: 50/60Hz 전원선, 특정 고조파).</td></tr>
  <tr><td><b>STFT</b></td><td>신호를 짧은 구간으로 나누어 각 시간-주파수 셀에 Wiener 마스크를 적용한 후 역STFT로 복원합니다.</td><td>시간에 따라 변하는 비정상(non-stationary) 노이즈.</td></tr>
</table>
</body></html>
"""


# ──────────────────────────────────────────────
# Number formatter : no scientific notation
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
            mag = -int(math.floor(math.log10(abs_v)))
            total = mag + decimals
            return f"{v:.{total}f}"
    except Exception:
        return "N/A"


# ──────────────────────────────────────────────
# File parser
# ──────────────────────────────────────────────
def parse_vibro_file(filepath):
    domain = 'time'
    x_unit = 's'
    y_unit = 'm'
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
        return rms_measured / rms_noise

    def detect_noise_peaks(self, threshold_factor=3.0):
        if self.noise_fft is None:
            return []
        freqs = self.noise_fft['x']
        amps  = self.noise_fft['y']
        mean_amp  = np.mean(amps)
        threshold = threshold_factor * mean_amp
        peaks, _ = scipy_signal.find_peaks(
            amps,
            height=threshold,
            distance=max(1, len(freqs) // 200)
        )
        return [{'freq': float(freqs[p]), 'amplitude': float(amps[p])} for p in peaks]

    @staticmethod
    def estimate_fs(time_x):
        diffs = np.diff(time_x)
        dt = float(np.mean(diffs[diffs > 0]))
        return 1.0 / dt

    # ── Wiener Filter ────────────────────────
    def apply_wiener(self, measured_time, scale_factor=None):
        if self.noise_time is None:
            raise ValueError("No noise data loaded.")
        y_meas  = measured_time['y'].copy().astype(float)
        y_noise = self.noise_time['y'].copy().astype(float)
        n = min(len(y_meas), len(y_noise))
        y_meas  = y_meas[:n]
        y_noise = y_noise[:n]
        if scale_factor is None:
            scale_factor = self.noise_scale_factor(y_meas)
        y_noise_scaled = y_noise * scale_factor
        Y_meas  = np.fft.rfft(y_meas)
        Y_noise = np.fft.rfft(y_noise_scaled)
        P_meas  = np.abs(Y_meas)  ** 2
        P_noise = np.abs(Y_noise) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            SNR = np.where(P_noise > 0,
                           (P_meas - P_noise) / (P_noise + 1e-30), 0.0)
            SNR = np.maximum(SNR, 0.0)
            H   = SNR / (1.0 + SNR)
        Y_clean = H * Y_meas
        y_clean = np.fft.irfft(Y_clean, n=n)
        return y_clean, scale_factor

    # ── Notch Filter ─────────────────────────
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
    def apply_stft_denoise(self, measured_time, scale_factor=None,
                           nperseg=256, threshold_factor=3.0):
        if self.noise_time is None:
            raise ValueError("No noise data loaded.")
        y_meas  = measured_time['y'].copy().astype(float)
        y_noise = self.noise_time['y'].copy().astype(float)
        fs      = self.estimate_fs(measured_time['x'])
        n = min(len(y_meas), len(y_noise))
        y_meas  = y_meas[:n]
        y_noise = y_noise[:n]
        # nperseg must not exceed signal length
        nperseg = min(nperseg, n)
        if scale_factor is None:
            scale_factor = self.noise_scale_factor(y_meas)
        y_noise_scaled = y_noise * scale_factor
        f, t, Zxx_meas  = stft(y_meas,         fs=fs, nperseg=nperseg)
        _, _, Zxx_noise = stft(y_noise_scaled,  fs=fs, nperseg=nperseg)
        noise_power = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)
        meas_power  = np.abs(Zxx_meas) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = np.where(
                noise_power > 0,
                np.maximum((meas_power - noise_power) / (meas_power + 1e-30), 0.0),
                1.0
            )
        Zxx_clean  = Zxx_meas * mask
        _, y_clean = scipy_signal.istft(Zxx_clean, fs=fs, nperseg=nperseg)
        y_clean = y_clean[:n]
        return y_clean, f, t, np.abs(Zxx_meas), np.abs(Zxx_clean), scale_factor

    # ── Amplitude & Peak-to-Peak stats ───────
    def compute_amplitude_stats(self, y_clean, fs=None,
                                peak_prominence_factor=0.1):
        y = np.asarray(y_clean, dtype=float)
        if len(y) == 0:
            return None

        prominence_threshold = peak_prominence_factor * float(np.std(y))
        amplitude_global     = float(np.max(np.abs(y)))
        peak_to_peak_global  = float(np.max(y) - np.min(y))

        min_dist = max(1, len(y) // 500)
        peaks_pos, _ = scipy_signal.find_peaks(
             y, prominence=prominence_threshold, distance=min_dist)
        peaks_neg, _ = scipy_signal.find_peaks(
            -y, prominence=prominence_threshold, distance=min_dist)

        amp_pos = y[peaks_pos] if len(peaks_pos) > 0 else np.array([])
        amp_neg = y[peaks_neg] if len(peaks_neg) > 0 else np.array([])

        all_amps       = np.concatenate([np.abs(amp_pos), np.abs(amp_neg)])
        amplitude_mean = float(np.mean(all_amps)) if len(all_amps) > 0 \
                         else amplitude_global
        amplitude_std  = float(np.std(all_amps))  if len(all_amps) > 0 else 0.0

        pp_values = []
        if len(peaks_pos) > 0 and len(peaks_neg) > 0:
            for idx_p, val_p in zip(peaks_pos, amp_pos):
                closest = int(np.argmin(np.abs(peaks_neg - idx_p)))
                val_n   = amp_neg[closest]
                pp_values.append(float(val_p - val_n))

        pp_array = np.array(pp_values) if pp_values \
                   else np.array([peak_to_peak_global])

        dominant_freq_est = None
        if fs is not None and len(peaks_pos) >= 2:
            intervals   = np.diff(peaks_pos) / fs
            mean_period = float(np.mean(intervals))
            if mean_period > 0:
                dominant_freq_est = 1.0 / mean_period

        return {
            'amplitude_global':    amplitude_global,
            'peak_to_peak_global': peak_to_peak_global,
            'peaks_pos':           peaks_pos,
            'peaks_neg':           peaks_neg,
            'amplitude_mean':      amplitude_mean,
            'amplitude_std':       amplitude_std,
            'peak_to_peak_mean':   float(np.mean(pp_array)),
            'peak_to_peak_std':    float(np.std(pp_array)),
            'peak_to_peak_max':    float(np.max(pp_array)),
            'peak_to_peak_min':    float(np.min(pp_array)),
            'dominant_freq_est':   dominant_freq_est,
        }


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
            result = {}
            scale  = self.analyzer.noise_scale_factor(self.measured_time['y'])
            result['scale_factor'] = scale
            result['noise_peaks']  = self.analyzer.detect_noise_peaks(
                self.params.get('threshold_factor', 3.0))
            self.progress.emit(20)

            if self.method == 'Wiener':
                y_clean, _ = self.analyzer.apply_wiener(
                    self.measured_time, scale_factor=scale)
                result['y_clean'] = y_clean
                result['method']  = 'Wiener'

            elif self.method == 'Notch':
                y_clean, applied = self.analyzer.apply_notch(
                    self.measured_time,
                    threshold_factor=self.params.get('threshold_factor', 3.0),
                    q_factor=self.params.get('q_factor', 30.0))
                result['y_clean']       = y_clean
                result['notch_applied'] = applied
                result['method']        = 'Notch'

            elif self.method == 'STFT':
                y_clean, f, t, Zmeas, Zclean, _ = \
                    self.analyzer.apply_stft_denoise(
                        self.measured_time,
                        scale_factor=scale,
                        nperseg=self.params.get('nperseg', 256),
                        threshold_factor=self.params.get('threshold_factor', 3.0))
                result['y_clean']     = y_clean
                result['stft_f']      = f
                result['stft_t']      = t
                result['stft_before'] = Zmeas
                result['stft_after']  = Zclean
                result['method']      = 'STFT'

            self.progress.emit(60)

            fs_est = self.analyzer.estimate_fs(self.measured_time['x'])
            result['amp_stats'] = self.analyzer.compute_amplitude_stats(
                result['y_clean'],
                fs=fs_est,
                peak_prominence_factor=self.params.get('peak_prominence_factor', 0.1))

            self.progress.emit(85)

            result['x_time']     = self.measured_time['x']
            result['y_measured'] = self.measured_time['y']
            result['y_noise']    = self.analyzer.noise_time['y']

            mf = self.measured_fft
            result['fft_freq']   = mf['x'] if mf is not None else None
            result['fft_amp']    = mf['y'] if mf is not None else None

            nf = self.analyzer.noise_fft
            result['noise_freq'] = nf['x'] if nf is not None else None
            result['noise_amp']  = nf['y'] if nf is not None else None

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
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
        self._build_ui()

    # ── UI layout ────────────────────────────
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

    # ── Left panel ───────────────────────────
    def _build_left_panel(self):
        widget = QWidget()
        widget.setMaximumWidth(320)
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # Noise files
        grp_noise = QGroupBox("Noise Files")
        vn = QVBoxLayout(grp_noise)
        self.btn_load_noise_time = QPushButton("Load Noise Time File")
        self.btn_load_noise_fft  = QPushButton("Load Noise FFT File")
        self.lbl_noise_time      = QLabel("Not selected")
        self.lbl_noise_fft       = QLabel("Not selected")
        self.lbl_noise_time.setWordWrap(True)
        self.lbl_noise_fft.setWordWrap(True)
        self.btn_load_noise_time.clicked.connect(self._load_noise_time)
        self.btn_load_noise_fft.clicked.connect(self._load_noise_fft)
        vn.addWidget(self.btn_load_noise_time)
        vn.addWidget(self.lbl_noise_time)
        vn.addWidget(self.btn_load_noise_fft)
        vn.addWidget(self.lbl_noise_fft)
        layout.addWidget(grp_noise)

        # Measurement files
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

        # Display unit
        grp_unit = QGroupBox("Display Unit")
        vu = QFormLayout(grp_unit)
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(['m', 'mm', 'um', 'nm', 'pm'])
        self.combo_unit.setCurrentText('nm')
        vu.addRow("Displacement unit:", self.combo_unit)
        layout.addWidget(grp_unit)

        # Filter settings
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

        # Run button
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_analyze.setMinimumHeight(40)
        self.btn_analyze.setStyleSheet(
            "QPushButton{background:#2e86de;color:white;font-weight:bold;"
            "border-radius:5px;}"
            "QPushButton:hover{background:#1e6fbf;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_analyze.clicked.connect(self._run_analysis)
        layout.addWidget(self.btn_analyze)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return widget

    # ── Bottom panel ─────────────────────────
    def _build_bottom_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title row
        title_row = QHBoxLayout()
        lbl = QLabel("Analysis Results")
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        title_row.addWidget(lbl)

        btn_help = QPushButton("?")
        btn_help.setFixedSize(24, 24)
        btn_help.setStyleSheet(
            "QPushButton{background:#2e86de;color:white;"
            "border-radius:12px;font-weight:bold;font-size:13px;}"
            "QPushButton:hover{background:#1e6fbf;}"
        )
        btn_help.setToolTip("Show result descriptions")
        btn_help.clicked.connect(self._show_help_dialog)
        title_row.addWidget(btn_help)
        title_row.addStretch()
        layout.addLayout(title_row)

        # Result tabs
        self.result_tab = QTabWidget()
        self.result_tab.setStyleSheet("QTabBar::tab{min-width:130px;}")

        self.table_noise     = self._make_result_table()
        self.table_amplitude = self._make_result_table()
        self.table_pp        = self._make_result_table()
        self.table_freq      = self._make_result_table()

        for tab_title, table in [
            ("Signal & Noise", self.table_noise),
            ("Amplitude",      self.table_amplitude),
            ("Peak-to-Peak",   self.table_pp),
            ("Frequency",      self.table_freq),
        ]:
            w  = QWidget()
            tl = QVBoxLayout(w)
            tl.setContentsMargins(0, 0, 0, 0)
            tl.addWidget(table)
            self.result_tab.addTab(w, tab_title)

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

    # ── Help dialog ──────────────────────────
    def _show_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Analysis Results - Description")
        dlg.setMinimumSize(720, 660)
        layout = QVBoxLayout(dlg)

        # Language selector
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
        text.setStyleSheet("background-color:#f8f9fa;")
        text.setHtml(HELP_TEXT_EN)
        layout.addWidget(text)

        def on_lang_changed(idx):
            text.setHtml(HELP_TEXT_EN if idx == 0 else HELP_TEXT_KO)

        combo_lang.currentIndexChanged.connect(on_lang_changed)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dlg.close)
        layout.addWidget(btn_box)
        dlg.exec_()

    # ── Graph tabs ───────────────────────────
    def _init_graph_tabs(self):
        for title, creator in [
            ("Time Signal",      self._create_time_canvas),
            ("FFT Spectrum",     self._create_fft_canvas),
            ("Filtered Result",  self._create_filtered_canvas),
            ("STFT Spectrogram", self._create_stft_canvas),
        ]:
            canvas, fig = creator()
            if not hasattr(self, 'canvases'):
                self.canvases = {}
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

    # ── File loading ─────────────────────────
    def _load_noise_time(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Noise Time File", "", "Text Files (*.txt);;All Files (*)")
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
            self, "Select Noise FFT File", "", "Text Files (*.txt);;All Files (*)")
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
            self, "Select Measurement Files", "", "Text Files (*.txt);;All Files (*)")
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

    # ── Run analysis ─────────────────────────
    def _run_analysis(self):
        row = self.list_meas.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Warning", "Please select a measurement file.")
            return
        if self.noise_time_data is None:
            QMessageBox.warning(self, "Warning", "Please load a noise time file.")
            return
        time_d, fft_d, _ = self.meas_files[row]
        method = self.combo_method.currentText()
        params = {
            'threshold_factor':       self.spin_threshold.value(),
            'q_factor':               self.spin_q.value(),
            'nperseg':                self.spin_nperseg.value(),
            'peak_prominence_factor': self.spin_prominence.value(),
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

    # ── Unit helpers ─────────────────────────
    def _unit_scale(self):
        return UNIT_SCALE[self.combo_unit.currentText()]

    def _unit_label(self):
        return UNIT_LABEL[self.combo_unit.currentText()]

    # ── Plot updates ─────────────────────────
    def _update_all_plots(self, r):
        self._plot_time(r)
        self._plot_fft(r)
        self._plot_filtered(r)
        if r['method'] == 'STFT' and 'stft_f' in r:
            self._plot_stft(r)

    def _plot_time(self, r):
        canvas, fig = self.canvases["Time Signal"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_time:
            ax.cla()

        n = min(len(r['x_time']), len(r['y_measured']), len(r['y_noise']),
                len(r['y_clean']))
        t       = r['x_time'][:n]
        y_meas  = r['y_measured'][:n] * scale
        y_noise = r['y_noise'][:n]    * r['scale_factor'] * scale
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

    def _plot_fft(self, r):
        canvas, fig = self.canvases["FFT Spectrum"]
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        for ax in self._ax_fft:
            ax.cla()

        if r['noise_freq'] is not None:
            self._ax_fft[0].semilogy(
                r['noise_freq'], r['noise_amp'] * scale,
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

        if r['fft_freq'] is not None:
            self._ax_fft[1].semilogy(
                r['fft_freq'], r['fft_amp'] * scale,
                color='#2e86de', lw=0.8, label='Measured FFT')
        self._ax_fft[1].set_xlabel('Frequency [Hz]')
        self._ax_fft[1].set_ylabel(f'Amplitude [{ulbl}]')
        self._ax_fft[1].set_title('Measured Signal FFT Spectrum')
        self._ax_fft[1].legend()
        self._ax_fft[1].grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()

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

        if 'amp_stats' in r and r['amp_stats'] is not None:
            stats   = r['amp_stats']
            pk_pos  = stats['peaks_pos']
            pk_neg  = stats['peaks_neg']

            # guard : only plot indices within n
            pk_pos = pk_pos[pk_pos < n]
            pk_neg = pk_neg[pk_neg < n]

            if len(pk_pos) > 0:
                self._ax_filtered[0].scatter(
                    t[pk_pos], y_clean[pk_pos],
                    color='#e74c3c', s=25, zorder=5,
                    label=f'Positive peaks ({len(pk_pos)})', marker='^')
            if len(pk_neg) > 0:
                self._ax_filtered[0].scatter(
                    t[pk_neg], y_clean[pk_neg],
                    color='#8e44ad', s=25, zorder=5,
                    label=f'Negative peaks ({len(pk_neg)})', marker='v')

            y_center = (float(np.max(y_clean)) + float(np.min(y_clean))) / 2

            pp_half  = stats['peak_to_peak_global'] * scale / 2
            pp_val   = fmt(stats['peak_to_peak_global'] * scale, 4)
            self._ax_filtered[0].axhline(
                y_center + pp_half, color='#f39c12',
                linestyle='--', lw=1.0, alpha=0.8,
                label=f'P-P (global): {pp_val} {ulbl}')
            self._ax_filtered[0].axhline(
                y_center - pp_half, color='#f39c12',
                linestyle='--', lw=1.0, alpha=0.8)

            amp_val = fmt(stats['amplitude_global'] * scale, 4)
            self._ax_filtered[0].axhline(
                 stats['amplitude_global'] * scale, color='#e74c3c',
                linestyle=':', lw=1.0, alpha=0.6,
                label=f'Amplitude (global): {amp_val} {ulbl}')
            self._ax_filtered[0].axhline(
                -stats['amplitude_global'] * scale, color='#e74c3c',
                linestyle=':', lw=1.0, alpha=0.6)

            pp_mean_half = stats['peak_to_peak_mean'] * scale / 2
            pp_mean_val  = fmt(stats['peak_to_peak_mean'] * scale, 4)
            self._ax_filtered[0].axhline(
                y_center + pp_mean_half, color='#16a085',
                linestyle='-.', lw=1.0, alpha=0.7,
                label=f'P-P (mean): {pp_mean_val} {ulbl}')
            self._ax_filtered[0].axhline(
                y_center - pp_mean_half, color='#16a085',
                linestyle='-.', lw=1.0, alpha=0.7)

        self._ax_filtered[0].set_xlabel('Time [s]')
        self._ax_filtered[0].set_ylabel(f'Displacement [{ulbl}]')
        self._ax_filtered[0].set_title(
            'Raw vs Filtered Signal  (Amplitude / P-P)')
        self._ax_filtered[0].legend(fontsize=7, loc='upper right')
        self._ax_filtered[0].grid(True, alpha=0.3)

        # FFT comparison
        fs    = 1.0 / float(np.mean(np.diff(t)))
        N     = len(y_meas)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        amp_meas  = (2/N) * np.abs(np.fft.rfft(r['y_measured'][:n])) * scale
        amp_clean = (2/N) * np.abs(np.fft.rfft(r['y_clean'][:n]))    * scale

        self._ax_filtered[1].semilogy(freqs, amp_meas  + 1e-30,
                                      color='#2e86de', lw=0.8,
                                      alpha=0.6, label='Raw FFT')
        self._ax_filtered[1].semilogy(freqs, amp_clean + 1e-30,
                                      color='#27ae60', lw=0.8,
                                      alpha=0.9, label='Filtered FFT')

        if r['amp_stats'] is not None and \
                r['amp_stats']['dominant_freq_est'] is not None:
            f_dom = r['amp_stats']['dominant_freq_est']
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

    def _plot_stft(self, r):
        canvas, fig = self.canvases["STFT Spectrogram"]
        scale = self._unit_scale()
        for ax in self._ax_stft:
            ax.cla()

        f      = r['stft_f']
        t      = r['stft_t']
        Zmeas  = r['stft_before'] * scale
        Zclean = r['stft_after']  * scale

        pos_vals = Zmeas[Zmeas > 0]
        vmin = float(np.percentile(pos_vals, 5))  if len(pos_vals) > 0 else 1e-12
        vmax = float(np.percentile(pos_vals, 99)) if len(pos_vals) > 0 else 1e-6

        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        ulbl = self._unit_label()

        im0 = self._ax_stft[0].pcolormesh(
            t, f, Zmeas, norm=norm, cmap='inferno', shading='gouraud')
        self._ax_stft[0].set_title('STFT - Raw')
        self._ax_stft[0].set_xlabel('Time [s]')
        self._ax_stft[0].set_ylabel('Frequency [Hz]')
        fig.colorbar(im0, ax=self._ax_stft[0], label=f'Amplitude [{ulbl}]')

        Zclean_safe = np.where(Zclean > 0, Zclean, vmin)
        im1 = self._ax_stft[1].pcolormesh(
            t, f, Zclean_safe, norm=norm, cmap='inferno', shading='gouraud')
        self._ax_stft[1].set_title('STFT - After Denoising')
        self._ax_stft[1].set_xlabel('Time [s]')
        self._ax_stft[1].set_ylabel('Frequency [Hz]')
        fig.colorbar(im1, ax=self._ax_stft[1], label=f'Amplitude [{ulbl}]')

        fig.tight_layout()
        canvas.draw()

    # ── Result table ─────────────────────────
    def _update_result_table(self, r):
        scale = self._unit_scale()
        ulbl  = self._unit_label()
        n     = min(len(r['y_measured']), len(r['y_clean']))

        # ── Tab 1 : Signal & Noise ────────────
        rms_meas         = VibrationAnalyzer.rms(r['y_measured'][:n]) * scale
        rms_clean        = VibrationAnalyzer.rms(r['y_clean'][:n])    * scale
        rms_noise_scaled = VibrationAnalyzer.rms(
            r['y_noise'][:n] * r['scale_factor']) * scale

        rows_noise = [
            ("Noise scale factor",
             fmt(r['scale_factor'], 6), "-",
             "RMS(measured) / RMS(noise)"),
            ("RMS - Raw measured",
             fmt(rms_meas, 4), ulbl,
             "Energy of raw (device + noise) signal"),
            ("RMS - Noise (scaled)",
             fmt(rms_noise_scaled, 4), ulbl,
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
                f"Amplitude: {fmt(pk['amplitude']*scale, 4)} {ulbl}"))
        if r['method'] == 'Notch' and 'notch_applied' in r:
            for f0 in r['notch_applied']:
                rows_noise.append((
                    "Notch removed freq.",
                    fmt(f0, 3), "Hz", "IIR notch filter applied"))
        self._fill_table(self.table_noise, rows_noise)

        # ── Tab 2 : Amplitude ─────────────────
        rows_amp = []
        s = r.get('amp_stats')
        if s:
            rows_amp = [
                ("Amplitude - global max|y|",
                 fmt(s['amplitude_global'] * scale, 4), ulbl,
                 "max(|y|)  largest displacement from zero"),
                ("Amplitude - peak mean",
                 fmt(s['amplitude_mean'] * scale, 4), ulbl,
                 "Mean of all detected peak magnitudes"),
                ("Amplitude - peak std",
                 fmt(s['amplitude_std'] * scale, 4), ulbl,
                 "Std dev of peak magnitudes"),
                ("Positive peaks count",
                 str(len(s['peaks_pos'])), "count",
                 "Detected above prominence threshold"),
                ("Negative peaks count",
                 str(len(s['peaks_neg'])), "count",
                 "Detected above prominence threshold"),
            ]
        self._fill_table(self.table_amplitude, rows_amp)

        # ── Tab 3 : Peak-to-Peak ──────────────
        rows_pp = []
        if s:
            rows_pp = [
                ("P-P - global",
                 fmt(s['peak_to_peak_global'] * scale, 4), ulbl,
                 "max(y) - min(y)  entire signal"),
                ("P-P - pair mean",
                 fmt(s['peak_to_peak_mean'] * scale, 4), ulbl,
                 "Mean of nearest pos/neg peak pairs"),
                ("P-P - pair std",
                 fmt(s['peak_to_peak_std'] * scale, 4), ulbl,
                 "Std dev of per-cycle P-P"),
                ("P-P - maximum",
                 fmt(s['peak_to_peak_max'] * scale, 4), ulbl,
                 "Largest single-cycle P-P"),
                ("P-P - minimum",
                 fmt(s['peak_to_peak_min'] * scale, 4), ulbl,
                 "Smallest single-cycle P-P"),
                ("P-P range (max - min)",
                 fmt((s['peak_to_peak_max'] - s['peak_to_peak_min']) * scale, 4),
                 ulbl, "Spread of P-P variation"),
            ]
        self._fill_table(self.table_pp, rows_pp)

        # ── Tab 4 : Frequency ─────────────────
        rows_freq = []
        if s and s['dominant_freq_est'] is not None:
            rows_freq += [
                ("Est. dominant frequency",
                 fmt(s['dominant_freq_est'], 4), "Hz",
                 "From mean interval of positive peaks"),
                ("Est. dominant period",
                 fmt(1.0 / s['dominant_freq_est'], 6), "s",
                 "1 / dominant frequency"),
            ]
        for i, pk in enumerate(r['noise_peaks']):
            rows_freq.append((
                f"Noise frequency #{i+1}",
                fmt(pk['freq'], 3), "Hz",
                f"Noise amplitude: {fmt(pk['amplitude']*scale, 4)} {ulbl}"))
        if r['method'] == 'Notch' and 'notch_applied' in r:
            for f0 in r['notch_applied']:
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
