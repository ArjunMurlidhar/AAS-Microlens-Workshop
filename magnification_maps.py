import numpy as np
import VBMicrolensing as vbm
import astropy.io.fits as fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def _compute_mag_row(args):
    """
    Compute one row (fixed u, varying t) of the magnification map.
    Returns a tuple (i, row) so the caller can place it correctly.
    """
    VBM = vbm.VBMicrolensing()
    i, s, q, rho, tau, ulist, use_float32 = args
    
    u = float(ulist[i])

    # Compute with float64 for stability, then cast if requested
    row64 = np.empty(tau.shape[0], dtype=np.float64)
    for j in range(tau.shape[0]):
        row64[j] = VBM.BinaryMag2(s, q, float(tau[j]), u, rho)

    if use_float32:
        return i, row64.astype(np.float32, copy=False)
    return i, row64


def create_mag_map(s, q, rho, res=1e-3, side=8, num_cores=1, float_precision='float64', backend='auto'):
    '''
    Create a magnification map for a given binary lens model.

    Parameters
    ----------
    s : float
        Lens separation in units of the Einstein radius.
    q : float
        The ratio of the mass of the secondary lens to the primary lens.
    rho : float
        Source radius in units of the Einstein radius.
    res : float, optional
        Resolution of the grid in units of the Einstein radius.
    side : float, optional
        Side length of the grid in units of the Einstein radius.
    num_cores : int, optional
        Number of CPU cores to use for parallel computation (default 1 = serial).
    float_precision : {'float32', 'float64'}, optional
        Precision of the returned magnification map. Calculations are performed
        in float64 and cast to float32 if requested.
    backend : {'auto', 'processes', 'threads'}, optional
        Parallel backend. Use 'threads' in notebooks if processes cause errors.

    Returns
    -------
    mag_map : numpy.ndarray
        A 2D array representing the magnification map, with the requested dtype.
    '''

    # Grid setup
    t_steps = int(side / res)
    u_steps = int(side / res)

    use_float32 = str(float_precision).lower() == 'float32'
    out_dtype = np.float32 if use_float32 else np.float64

    # Shape (u_steps, t_steps): each row corresponds to a u value, columns to t
    mag_map = np.empty((u_steps, t_steps), dtype=out_dtype)

    # Axes (computed in float64 regardless of output dtype)
    tau = np.linspace(-side/2, side/2, t_steps, dtype=np.float64)
    ulist = np.linspace(-side/2, side/2, u_steps, dtype=np.float64)

    # Serial path
    if num_cores is None or int(num_cores) <= 1:
        for i in range(u_steps):
            _, row = _compute_mag_row((i, s, q, rho, tau, ulist, use_float32))
            mag_map[i, :] = row
        return mag_map

    # Parallel path
    max_workers = min(int(num_cores), u_steps) if num_cores is not None else u_steps

    def _run_parallel_pool(executor_cls):
        with executor_cls(max_workers=max_workers) as ex:
            futures = [ex.submit(_compute_mag_row, (i, s, q, rho, tau, ulist, use_float32)) for i in range(u_steps)]
            for fut in as_completed(futures):
                i, row = fut.result()
                mag_map[i, :] = row

    if backend == 'threads':
        _run_parallel_pool(ThreadPoolExecutor)
    elif backend == 'processes':
        _run_parallel_pool(ProcessPoolExecutor)
    else:  # auto
        try:
            _run_parallel_pool(ProcessPoolExecutor)
        except Exception:
            _run_parallel_pool(ThreadPoolExecutor)

    return mag_map


def save_mag_map(mag_map, filename,
                                 compression='GZIP_2',
                                 tile_shape=None,
                                 quantize_level=None,
                                 metadata=None):
    '''
    Save magnification map as a compressed FITS file using tiled image compression.

    Parameters
    ----------
    mag_map : np.ndarray
        2D array with rows=u and columns=t.
    filename : str
        Output filename, e.g. 'mag_map.fits.fz' or 'mag_map.fits'.
    compression : {'GZIP_2', 'RICE_1', 'PLIO_1', 'HCOMPRESS_1'}
        Compression algorithm. Use 'GZIP_2' for lossless floats.
        Use 'RICE_1' with quantize_level for smaller, lossy float files.
    tile_shape : tuple[int, int] | None
        Tiling for compression. Defaults to up to 512x512.
    quantize_level : float | None
        Quantization step for floating point when using 'RICE_1'.
        Smaller values => better fidelity, larger => smaller files.
        Ignored for lossless (e.g., 'GZIP_2').
    metadata : dict | None
        Optional FITS header keywords to record (keys will be uppercased and
        truncated to 8 chars as needed).
    '''
    if tile_shape is None:
        tile_shape = (min(mag_map.shape[0], 512), min(mag_map.shape[1], 512))

    # Create compressed image HDU
    hdu = fits.CompImageHDU(
        data=mag_map,
        compression_type=compression,
        tile_shape=tile_shape,
        quantize_level=quantize_level
    )

    # Add optional metadata
    if metadata:
        for k, v in metadata.items():
            key = str(k).upper()[:8]
            try:
                hdu.header[key] = v
            except Exception:
                # Skip values that cannot be serialized into FITS header
                pass

    # Write file (PrimaryHDU + compressed image extension)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(filename, overwrite=True)


# Example usage:
# meta = {'S': s, 'Q': q, 'RHO': rho, 'RES': res, 'SIDE': side, 'DTYPE': str(mag_map.dtype)}
# save_mag_map_compressed_fits(
#     mag_map,
#     filename='mag_map.fits.fz',
#     compression='GZIP_2',        # lossless for floats
#     tile_shape=None,             # defaults to up to 512x512 tiles
#     quantize_level=None,         # not used for GZIP_2
#     metadata=meta
# )
#
# For smaller (lossy) files with very good fidelity on float data:
# save_mag_map_compressed_fits(
#     mag_map.astype(np.float32, copy=False),
#     filename='mag_map_rice.fits.fz',
#     compression='RICE_1',
#     tile_shape=None,
#     quantize_level=16.0,         # adjust based on acceptable quantization
#     metadata=meta
# )

if __name__ == "__main__":
    magmap = create_mag_map(1.35, 0.0058, 0.0098, res=1e-3, side=8, num_cores=4, float_precision='float32', backend='threads')
    save_mag_map(magmap, 'magmap_test.fits.fz', compression='GZIP_2', tile_shape=None, quantize_level=None, metadata={'S': 1.35, 'Q': 0.0058, 'RHO': 0.0098, 'RES': 1e-3, 'SIDE': 8, 'DTYPE': str(magmap.dtype)})

