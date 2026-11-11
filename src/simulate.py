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

_wave = None


def _generator():
    global _wave
    if _wave is None:
        _wave = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True),
                              response_kwargs=dict(TDItag="AET", tdi2=True))
    return _wave


def mbhb_aet(m1, m2, chi1, chi2, dist_mpc, inc, lam, beta, psi, phi_ref,
             t_c, n, dt, t_start, f_lo=1e-4, f_hi=0.1, f_ref=0.0):
    """Return (A, E, T) time series of n samples at spacing dt.

    m1, m2 are detector-frame masses in solar masses (m1 >= m2); lam/beta are
    ecliptic longitude/latitude; t_c is the absolute coalescence time (s).
    """
    freqs = np.fft.rfftfreq(n, dt)
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    wave = _generator()(
        m1, m2, chi1, chi2, dist_mpc * MPC, phi_ref, f_ref, inc, lam, beta, psi,
        t_c, freqs=freqs[sel], direct=True, fill=False, squeeze=True,
        t_obs_start=0.0, t_obs_end=(t_c + 3 * 86400.0) / YEAR, shift_t_limits=True)
    wave = np.asarray(wave)                       # (3, nfreq_selected)
    spec = np.zeros((3, len(freqs)), dtype=complex)
    spec[:, sel] = wave
    spec *= np.exp(2j * np.pi * freqs * t_start)  # move segment start to t_start
    ts = np.fft.irfft(spec, n=n, axis=1) / dt
    return ts[0], ts[1], ts[2]
