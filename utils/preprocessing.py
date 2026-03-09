## Pipeline for spectra preparation
import pandas as pd
import numpy as np
from chemotools.baseline import LinearCorrection
from chemotools.derivative import SavitzkyGolay
from sklearn.pipeline import make_pipeline

def process_amide_region(df):
    """
    a Dataframe where:
        - Col 0 : Sample ID
        - Col 1+ : spectra with wavenumbers as column headers
    """
    # Separate ID with the spectra and extract wavenumbers
    ids = df.iloc[:, 0].reset_index(drop=True)
    spectra = df.iloc[:, 1:]
    wavenumbers = spectra.columns.astype(float).to_numpy()

    # Define the range mask first
    mask = (wavenumbers >= 1480) & (wavenumbers <= 1710)
    wavenumbers_cut = wavenumbers[mask]
    spectra_cut = spectra.iloc[:, mask]
    
    # Define the processing pipeline
    pipe = make_pipeline(
        LinearCorrection(),
        SavitzkyGolay(window_size=11, polynomial_order=3, derivate_order=0, mode='nearest')
    )
    
    # Process spectra
    processed_spectra = pipe.fit_transform(spectra_cut)
    
    # Reconstruct DataFrame with correct column names
    df_clean = pd.DataFrame(processed_spectra, columns=wavenumbers_cut)
    
    # Reattach the IDs
    df_final = pd.concat([ids, df_clean], axis=1)
    
    return df_final


def integrate_band(df, start, end):
    """
    Integrates a specific spectral range using the Trapezoidal rule.
    
    Parameters:
    - df: DataFrame (Col 0 is ID, Cols 1+ are Wavenumbers)
    - start, end: The range to integrate (default is Amide II range)
    """

    # Separate ID with the spectra and extract wavenumbers
    ids = df.iloc[:, 0].reset_index(drop=True)
    spectra = df.iloc[:, 1:]
    wavenumbers = spectra.columns.astype(float).to_numpy()
    mask = (wavenumbers >= start) & (wavenumbers <= end)

    # get the x (wavenumbers) and y (absorbance)
    x = wavenumbers[mask]
    y = spectra.loc[:, mask].values

    # perform integration
    areas = np.abs(np.trapezoid(y, x, axis=1))

    results = pd.DataFrame({
        'SampleID': ids,
        'Areas': areas
    })

    return results