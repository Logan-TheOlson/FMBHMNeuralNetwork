"""IMRPhenomD + LISA-response MBHB waveforms (bbhx) as A/E time series.

bbhx returns the TDI response in the frequency domain. We evaluate it on the
rfft grid of a data segment and inverse-FFT, after applying the time shift of
the segment start (bbhx puts the coalescence at absolute time t_ref).

Times are absolute seconds on the LISA orbit clock; `t_start` is the absolute
time of the first sample of the segment.
"""
import numpy as np
from bbhx.waveformbuild import BBHWaveformFD

MPC = 3.0856775814913673e22   # metres
YEAR = 31558149.763545603     # s, as in the Spritz catalog

_waves = {}


def gpu_available():
    """True if bbhx was built with CUDA support and a CUDA device is visible."""
    try:
        import cupy
        import bbhx.waveformbuild as wb
        return hasattr(wb, "direct_sum_wrap_gpu") and cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _generator(use_gpu=False):
    if use_gpu not in _waves:
        _waves[use_gpu] = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True),
                                        response_kwargs=dict(TDItag="AET", tdi2=True),
                                        use_gpu=use_gpu)
    return _waves[use_gpu]


def mbhb_aet(m1, m2, chi1, chi2, dist_mpc, inc, lam, beta, psi, phi_ref,
             t_c, n, dt, t_start, f_lo=1e-4, f_hi=0.1, f_ref=0.0, use_gpu=False):
    """Return (A, E, T) time series of n samples at spacing dt.

    m1, m2 are detector-frame masses in solar masses (m1 >= m2); lam/beta are
    ecliptic longitude/latitude; t_c is the absolute coalescence time (s).

    With use_gpu=True (needs bbhx-cuda12x + CuPy) the waveform, response and
    inverse FFT all run on the GPU and CuPy arrays are returned.
    """
    xp = np
    if use_gpu:
        import cupy as xp
    freqs = xp.asarray(np.fft.rfftfreq(n, dt))
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    wave = _generator(use_gpu)(
        m1, m2, chi1, chi2, dist_mpc * MPC, phi_ref, f_ref, inc, lam, beta, psi,
        t_c, freqs=freqs[sel], direct=True, fill=False, squeeze=True,
        t_obs_start=0.0, t_obs_end=(t_c + 3 * 86400.0) / YEAR, shift_t_limits=True)
    wave = xp.asarray(wave)                       # (3, nfreq_selected)
    spec = xp.zeros((3, len(freqs)), dtype=complex)
    spec[:, sel] = wave
    spec *= xp.exp(2j * np.pi * freqs * t_start)  # move segment start to t_start
    ts = xp.fft.irfft(spec, n=n, axis=1) / dt
    return ts[0], ts[1], ts[2]
