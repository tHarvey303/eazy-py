import numpy as np
import numba
from numpy.typing import NDArray
from typing import Tuple

# A small value to prevent division by zero and for float comparisons
EPSILON = np.finfo(np.float64).eps

@numba.njit(cache=True, fastmath=True)
def _nnls_numba(A: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
    """
    Numba implementation of the Lawson-Hanson NNLS algorithm.

    Solves argmin_x || Ax - b ||_2 for x >= 0.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)
    w = np.dot(A.T, (b - np.dot(A, x)))

    # Active and passive sets
    P = np.zeros(n, dtype=np.bool_)
    Z = np.ones(n, dtype=np.bool_)

    max_iter = 3 * n
    for _ in range(max_iter):
        if np.all(w[Z] <= EPSILON):
            break

        j = np.argmax(w[Z])
        j_idx = np.where(Z)[0][j]
        
        P[j_idx] = True
        Z[j_idx] = False

        while True:
            Ap = A[:, P]
            s, _, _, _ = np.linalg.lstsq(Ap, b)

            s_full = np.zeros(n, dtype=np.float64)
            s_full[P] = s

            if np.all(s > -EPSILON):
                x = s_full
                w = np.dot(A.T, (b - np.dot(A, x)))
                break
            
            alpha = np.inf
            active_indices = np.where(P)[0]
            for i in range(len(s)):
                if s[i] < 0:
                    p_idx = active_indices[i]
                    val = x[p_idx] / (x[p_idx] - s[i])
                    if val < alpha:
                        alpha = val
            
            x += alpha * (s_full - x)
            
            # Move elements from P to Z where x is zero
            P[(P) & (np.abs(x) < EPSILON)] = False
            Z = ~P

    rnorm = np.linalg.norm(np.dot(A, x) - b)
    return x, rnorm

@numba.njit(cache=True, fastmath=True)
def template_lsq_numba(
    fnu_i: NDArray,
    efnu_i: NDArray,
    Ain: NDArray,
    TEFz: NDArray,
    zp: NDArray,
    renorm_t: bool,
    hess_threshold: float,
) -> Tuple[float, NDArray, NDArray]:
    """
    Numba implementation of the core template least-squares fitting.
    """
    NTEMP, NFILT = Ain.shape
    MIN_VALID_FILTERS = 1

    ok_band = (efnu_i / zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
    
    if np.sum(ok_band) < MIN_VALID_FILTERS:
        coeffs_i = np.zeros(NTEMP, dtype=fnu_i.dtype)
        fmodel = np.dot(coeffs_i, Ain)
        return np.inf, coeffs_i, fmodel
    
    var = efnu_i**2 + (TEFz * np.maximum(fnu_i, 0.0))**2
    rms = np.sqrt(var)
    
    A_ok = Ain[:, ok_band]
    rms_ok = rms[ok_band]
    
    Anorm = np.ones(NTEMP, dtype=fnu_i.dtype)
    if renorm_t:
        for i in range(NTEMP):
            # L2 norm: np.linalg.norm(x) == np.sqrt(np.sum(x**2))
            norm_val = np.sqrt(np.sum((A_ok[i, :] / rms_ok)**2))
            if norm_val > 0:
                Anorm[i] = norm_val

    A = np.zeros_like(Ain)
    for i in range(NTEMP):
        A[i, :] = Ain[i, :] / Anorm[i]

    ok_temp = Anorm > 0

    if np.sum(ok_temp) == 0:
        coeffs_i = np.zeros(NTEMP, dtype=fnu_i.dtype)
        fmodel = np.dot(coeffs_i, Ain)
        return np.inf, coeffs_i, fmodel

    Ax = np.zeros((np.sum(ok_band), NTEMP), dtype=fnu_i.dtype)
    for i in range(NTEMP):
        Ax[:, i] = A[i, ok_band] / rms_ok
        
    if hess_threshold < 1.0:
        Hess = np.dot(Ax.T, Ax)
        temp_indices = np.where(ok_temp)[0]
        
        for i_idx, i in enumerate(temp_indices):
            if not ok_temp[i]:
                continue
            
            for j in temp_indices[i_idx + 1:]:
                if Hess[i, j] > hess_threshold:
                    ok_temp[j] = False
    
    active_temps = np.sum(ok_temp)
    if active_temps == 0:
        coeffs_i = np.zeros(NTEMP, dtype=fnu_i.dtype)
        fmodel = np.dot(coeffs_i, Ain)
        return np.inf, coeffs_i, fmodel
        
    Ax_fit = np.ascontiguousarray(Ax[:, ok_temp])
    b = fnu_i[ok_band] / rms_ok

    # NNLS solver requires float64 for stability
    coeffs_x, _ = _nnls_numba(Ax_fit.astype(np.float64), b.astype(np.float64))

    coeffs_i = np.zeros(NTEMP, dtype=fnu_i.dtype)
    coeffs_i[ok_temp] = coeffs_x.astype(fnu_i.dtype)
    
    for i in range(NTEMP):
        coeffs_i[i] /= Anorm[i]

    fmodel = np.dot(coeffs_i, Ain)
    chi2_i = np.sum(((fnu_i[ok_band] - fmodel[ok_band])**2) / var[ok_band])

    return chi2_i, coeffs_i, fmodel


@numba.njit(parallel=True, cache=True)
def fit_by_redshift_numba(
    A: NDArray, 
    fnu_corr: NDArray, 
    efnu_corr: NDArray,
    TEFz: NDArray,
    zp: NDArray,
    renorm_t: bool,
    hess_threshold: float,
) -> Tuple[NDArray, NDArray]:
    """
    Fits all objects at a single redshift in parallel using Numba.
    """
    NOBJ, NFILT = fnu_corr.shape
    NTEMP = A.shape[0]
    
    chi2 = np.zeros(NOBJ, dtype=fnu_corr.dtype)
    coeffs = np.zeros((NOBJ, NTEMP), dtype=fnu_corr.dtype)

    for iobj in numba.prange(NOBJ):
        fnu_i = fnu_corr[iobj, :]
        efnu_i = efnu_corr[iobj, :]
        
        chi2_obj, coeffs_obj, _ = template_lsq_numba(
            fnu_i, efnu_i, A, TEFz, zp, renorm_t, hess_threshold
        )
        chi2[iobj] = chi2_obj
        coeffs[iobj, :] = coeffs_obj

    return chi2, coeffs

def fit_by_redshift_dispatcher(
    fitter_name: str,
    A: NDArray, 
    fnu_corr: NDArray, 
    efnu_corr: NDArray,
    TEFz: NDArray,
    zp: NDArray,
    renorm_t: bool,
    hess_threshold: float,
) -> Tuple[NDArray, NDArray]:
    """
    Selects the fitting function based on the provided name.
    This allows for easy extension to other numba-based fitters.
    """
    if fitter_name.lower() == 'numba_nnls':
        return fit_by_redshift_numba(
            A, fnu_corr, efnu_corr, TEFz, zp, renorm_t, hess_threshold
        )
    else:
        raise ValueError(f"Fitter '{fitter_name}' is not implemented in the dispatcher.")
    

@numba.njit(cache=True, fastmath=True)
def template_lsq_numba_covar(
    fnu_i: NDArray,
    efnu_i: NDArray,
    Ain: NDArray,
    TEFz: NDArray,
    zp: NDArray,
    renorm_t: bool,
    hess_threshold: float,
) -> Tuple[float, NDArray, NDArray, NDArray]:
    """
    Numba implementation of template_lsq that also returns the covariance matrix.
    """
    chi2_i, coeffs_i, fmodel = template_lsq_numba(
        fnu_i, efnu_i, Ain, TEFz, zp, renorm_t, hess_threshold
    )

    NTEMP = Ain.shape[0]
    covar = np.full((NTEMP, NTEMP), np.nan, dtype=np.float64)
    active_mask = coeffs_i > 0
    n_active = np.sum(active_mask)

    if n_active > 0:
        var = efnu_i**2 + (TEFz * np.maximum(fnu_i, 0.0))**2
        rms = np.sqrt(var)
        ok_band = (efnu_i / zp > 0) & np.isfinite(fnu_i) & np.isfinite(efnu_i)
        
        Ax = np.zeros((np.sum(ok_band), n_active), dtype=fnu_i.dtype)
        active_indices = np.where(active_mask)[0]
        
        for i_idx, i in enumerate(active_indices):
            Ax[:, i_idx] = Ain[i, ok_band] / rms[ok_band]

        hessian = np.dot(Ax.T, Ax)

        try:
            # Add small value to diagonal for stability before inverting
            hessian += np.eye(n_active) * 1e-8
            covar_active = np.linalg.inv(hessian.astype(np.float64))

            for r_idx, r in enumerate(active_indices):
                for c_idx, c in enumerate(active_indices):
                    covar[r, c_idx] = covar_active[r_idx, c_idx]
        except Exception:
            # Failed to invert, covar remains NaN
            pass
            
    return chi2_i, coeffs_i, fmodel, covar


@numba.njit(cache=True, fastmath=True)
def _single_object_rest_fluxes_refit(
    fnu_i: NDArray, efnu_i: NDArray, z: float, A: NDArray, zp: NDArray,
    rf_tempfilt_iz: NDArray, rf_lc: NDArray, pivot: NDArray,
    pad_width: float, max_err: float,
    renorm_t: bool, hess_threshold: float,
    percentile_sigma_multipliers: NDArray
) -> NDArray:
    """
    Numba kernel to perform re-fitting for all rest-frame filters for a single object.
    """
    NREST = len(rf_lc)
    f_rest_i = np.zeros((NREST, len(percentile_sigma_multipliers)), dtype=fnu_i.dtype)
    
    for i in range(NREST):
        # Calculate TEFz for this rest-frame band to re-weight photometry
        lc_i = rf_lc[i]
        x = np.log(lc_i / (pivot / (1 + z)))
        grow_denom = np.log(1 / (1 + pad_width))**2
        grow = np.exp(-x**2 / (2 * grow_denom))
        
        max_grow = 0.0
        for val in grow:
            if val > max_grow:
                max_grow = val

        if max_grow <= 0:
             TEFz = np.full(len(pivot), max_err, dtype=fnu_i.dtype)
        else:
             TEFz = (2 / (1 + grow / max_grow) - 1) * max_err

        # Get coefficients and covariance matrix from the re-weighted fit
        _, coeffs, _, covar = template_lsq_numba_covar(
            fnu_i, efnu_i, A, TEFz, zp, renorm_t, hess_threshold
        )
        
        # **FIX:** Cast template vector to float64 for all subsequent math
        rf_templates_i = rf_tempfilt_iz[:, i].astype(np.float64)
        
        # Now this dot product will have matching dtypes (float64, float64)
        mean_flux = np.dot(coeffs, rf_templates_i)
        
        active_mask = coeffs > 0
        sigma_flux = 0.0
        if np.sum(active_mask) > 0 and np.all(np.isfinite(covar)):
            rf_templates_active = rf_templates_i[active_mask]
            
            # Select sub-matrix from covar for active templates
            n_active = len(rf_templates_active)
            covar_active = np.zeros((n_active, n_active), dtype=covar.dtype)
            active_indices = np.where(active_mask)[0]
            for r_idx, r in enumerate(active_indices):
                for c_idx, c in enumerate(active_indices):
                    covar_active[r_idx, c_idx] = covar[r, c_idx]

            # Variance = F^T * Cov(c) * F
            var_flux = np.dot(rf_templates_active, np.dot(covar_active, rf_templates_active))
            sigma_flux = np.sqrt(var_flux) if var_flux > 0 else 0.0
            
        # Estimate percentiles assuming a Gaussian distribution
        for p_idx, p_mult in enumerate(percentile_sigma_multipliers):
            f_rest_i[i, p_idx] = mean_flux + sigma_flux * p_mult

    return f_rest_i

@numba.njit(parallel=True, cache=True)
def _rest_frame_fluxes_numba_simple(
    coeffs_draws: NDArray,
    izbest: NDArray,
    rf_tempfilt: NDArray,
    percentiles: NDArray
) -> NDArray:
    """
    Numba-accelerated rest-frame flux calculation for simple=True case.
    """
    NOBJ, NDRAWS, NTEMP = coeffs_draws.shape
    NREST = rf_tempfilt.shape[2]
    f_rest = np.zeros((NOBJ, NREST, len(percentiles)), dtype=coeffs_draws.dtype)
    
    for iobj in numba.prange(NOBJ):
        iz = izbest[iobj]
        rf_iz = rf_tempfilt[iz, :, :]
        
        for i in range(NREST):
            # Calculate flux for every draw
            dval = np.dot(coeffs_draws[iobj, :, :], rf_iz[:, i])
            f_rest[iobj, i, :] = np.percentile(dval, percentiles)
            
    return f_rest

@numba.njit(cache=True, fastmath=True)
def _single_object_fit_at_zbest_numba(
    fnu_i: NDArray,
    efnu_i: NDArray,
    A: NDArray,
    TEFz: NDArray,
    zp: NDArray,
    get_err: bool,
    renorm_t: bool,
    hess_threshold: float,
) -> Tuple[float, NDArray, NDArray, NDArray]:
    """
    Numba kernel to fit a single object at its best redshift and get errors.
    """
    chi2, coeffs, fmodel, covar = template_lsq_numba_covar(
        fnu_i, efnu_i, A, TEFz, zp, renorm_t, hess_threshold
    )

    efmodel = np.zeros_like(fmodel, dtype=fnu_i.dtype)
    if not get_err:
        return chi2, coeffs, fmodel, efmodel

    active_mask = coeffs > 0
    if np.sum(active_mask) > 0 and np.all(np.isfinite(covar)):
        A_active = A[active_mask, :]
        
        # Select sub-matrix from covar for active templates
        n_active = A_active.shape[0]
        covar_active = np.zeros((n_active, n_active), dtype=covar.dtype)
        active_indices = np.where(active_mask)[0]
        for r_idx, r in enumerate(active_indices):
            for c_idx, c in enumerate(active_indices):
                covar_active[r_idx, c_idx] = covar[r, c_idx]
        
        # Propagate covariance to model flux: var(f_k) = A_k^T Cov(c) A_k
        # This is the diagonal of A * Cov(c) * A.T
        model_covar = np.dot(A_active.T, np.dot(covar_active, A_active))
        var_fmodel = np.diag(model_covar)
        efmodel = np.sqrt(var_fmodel)

    return chi2, coeffs, fmodel, efmodel