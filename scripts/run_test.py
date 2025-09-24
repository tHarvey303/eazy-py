from eazy.photoz import PhotoZ
from astropy.table import Table
from eazy.param import EazyParam
# Generate a mock 10,000 galaxy catalog and fit with EAZY - basic JWSDT photometry and defasult templates

import re
from typing import Any, Dict, List, Optional, Tuple
import time
import copy
import matplotlib.pyplot as plt
import numpy as np

# A type alias for the parsed line data for clarity
ParsedLine = Dict[str, Any]

# Priority list for known ambiguous instruments (more common first)
INSTRUMENT_PRIORITY: List[str] = ['nircam', 'niriss']

# Regex to parse a line from the reference file, capturing the filter code
# and the rest of the line for tokenizing.
LINE_RE = re.compile(r'^\s*(\d+)\s+\d+\s+(.*)$')

def _parse_svo_name(name: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parses a filter name into (telescope, instrument, filter_name).

    Examples:
        'F150W' -> (None, None, 'f150w')
        'NIRCam.F150W' -> (None, 'nircam', 'f150w')
        'JWST/NIRCam.F150W' -> ('jwst', 'nircam', 'f150w')
    """
    name_lower = name.lower()
    
    parts = name_lower.split('/')
    telescope = parts[0] if len(parts) == 2 else None
    rest = parts[1] if len(parts) == 2 else parts[0]
        
    parts = rest.split('.')
    instrument = parts[0] if len(parts) == 2 else None
    filter_name = parts[1] if len(parts) == 2 else parts[0]
        
    return telescope, instrument, filter_name

def _get_best_match(
    matches: List[ParsedLine], 
    specified_instrument: Optional[str]
) -> Optional[int]:
    """Applies ambiguity rules to find the best match from a list."""
    if not matches:
        return None
        
    if len(matches) == 1:
        return matches[0]['code']

    # Rule 1: Instrument preference (only if user did not specify one)
    if not specified_instrument:
        found_instruments = {
            inst for m in matches for inst in INSTRUMENT_PRIORITY if inst in m['tokens']
        }
        
        # Find the highest-priority instrument present in the matches
        highest_priority_inst = next(
            (inst for inst in INSTRUMENT_PRIORITY if inst in found_instruments), None
        )
        
        # If found, filter the matches to only include that instrument
        if highest_priority_inst:
            matches = [m for m in matches if highest_priority_inst in m['tokens']]

    # Rule 2: Prefer newer versions (assumed to be those with the highest filter code)
    best_match = max(matches, key=lambda m: m['code'])
    return best_match['code']

def get_filter_codes(
    filter_names: List[str], 
    reference_file_path: str
) -> List[Optional[int]]:
    """
    Finds filter codes from a reference file for a list of filter names.

    For each input filter name, this function searches a reference file to
    find the corresponding filter code. It handles various name formats and
    resolves ambiguities by preferring newer versions and specified instruments.
    
    Args:
        filter_names: A list of filter names to look up.
        reference_file_path: The path to the filter reference file.

    Returns:
        A list of integer filter codes, with None for any names not found.
    """
    parsed_lines: List[ParsedLine] = []
    try:
        with open(reference_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = LINE_RE.match(line)
                if match:
                    description = match.group(2).lower()
                    parsed_lines.append({
                        'code': int(match.group(1)),
                        'tokens': set(re.split(r'[^a-z0-9]+', description))
                    })
    except FileNotFoundError:
        print(f"Error: Reference file not found at '{reference_file_path}'")
        return [None] * len(filter_names)

    results: List[Optional[int]] = []
    for name in filter_names:
        telescope, instrument, filter_part = _parse_svo_name(name)
        
        potential_matches: List[ParsedLine] = []
        for line_data in parsed_lines:
            tokens = line_data['tokens']
            # Check that all specified parts are present in the line's tokens
            if filter_part not in tokens:
                continue
            if instrument and instrument not in tokens:
                continue
            if telescope and telescope not in tokens:
                continue
            potential_matches.append(line_data)
        
        best_code = _get_best_match(potential_matches, instrument)
        results.append(best_code)
        
    return results

filter_ref = '/home/tharvey/work/eazy-py/eazy/data/eazy-photoz/filters/FILTER.RES.latest.info'

if __name__ == "__main__":
    catalog = "/home/tharvey/work/JADES-DR3-GS_MASTER_Sel-F277W+F356W+F444W_v13.fits"
    filters = ['F606W', 'F775W', 'F814W', 'F850LP', 'F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
    codes = get_filter_codes(filters, filter_ref)

    phot = Table.read(catalog, hdu='OBJECTS')
    # filter to first 10,000 galaxies
    phot = phot[:100_000]
    flux_col = [phot[f'FLUX_APER_{band}_aper_corr_Jy'] for band in filters]
    flux_err_col = [phot[f'FLUXERR_APER_{band}_loc_depth_10pc_Jy'] for band in filters]
    ids = phot['UNIQUE_ID']

    phot = Table()
    phot['id'] = ids
    for code, flux, flux_err in zip(codes, flux_col, flux_err_col):
        phot[f'F{code}'] = flux
        phot[f'E{code}'] = flux_err

    # Make input catalog

    params = {}

    params['CATALOG_FILE'] = phot

    ez = PhotoZ(
        param_file=None,
        translate_file=None,
        zeropoint_file=None,
        params=params,
        load_prior=False,
        load_products=False,
        n_proc=4,
        )
    
    # 1. Run the timing benchmarks
    print("Running Numba version of fit_catalog...")
    start = time.time()
    ez.fit_catalog(n_proc=4, get_best_fit=True, prior=False, beta_prior=False, use_numba=True)
    end = time.time()
    numba_cat_time = end - start

    zbest_numba = copy.deepcopy(ez.zml)

    print("Running Numba version of standard_output...")
    start = time.time()
    ez.standard_output(use_numba=True, vnorm_type=1, n_proc=4)
    end = time.time()
    numba_output_time = end - start
    
    print("Running original version of fit_catalog...")
    start = time.time()
    ez.fit_catalog(n_proc=4, get_best_fit=True, prior=False, beta_prior=False, use_numba=False)
    end = time.time()
    orig_cat_time = end - start

    zbest_orig = copy.copy(ez.zml)

    # Do a scatter plot

    plt.scatter(zbest_orig, zbest_numba, alpha=0.5)
    plt.xlabel('Original z')
    plt.ylabel('Numba z')
    plt.title('Comparison of Original and Numba z values')
    plt.grid()
    plt.show()

    print("Running original version of standard_output...")
    start = time.time()
    ez.standard_output(use_numba=False, vnorm_type=1, n_proc=4)
    end = time.time()
    orig_output_time = end - start

    print("\n--- Timing Results ---")
    print(f"fit_catalog: Original={orig_cat_time:.2f}s, Numba={numba_cat_time:.2f}s")
    print(f"standard_output: Original={orig_output_time:.2f}s, Numba={numba_output_time:.2f}s")
    print("----------------------\n")


    # 2. Create the histogram (bar chart)
    labels = ['fit_catalog', 'standard_output']
    original_times = [orig_cat_time, orig_output_time]
    numba_times = [numba_cat_time, numba_output_time]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, original_times, width, label='Original', color='skyblue')
    rects2 = ax.bar(x + width/2, numba_times, width, label='Numba', color='coral')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Performance Comparison: Original Python vs. Numba')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar, displaying its height
    ax.bar_label(rects1, padding=3, fmt='%.1fs')
    ax.bar_label(rects2, padding=3, fmt='%.1fs')

    # Add a grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    plt.show()




    # Time with ant without numba for 10,000 galaxies (for both fit_cataog and output) and make a comparison plot of time improvement

