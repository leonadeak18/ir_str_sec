import pandas as pd
from chemotools.outliers import DModX, QResiduals, HotellingT2
from sklearn.decomposition import PCA
from chemotools.derivative import SavitzkyGolay

def combined_outlier_test(df, n_components=0.95):
    """
    Detect outliers using DModX, HotellingT2, and Q-Residuals per sample group.
    """
    
    # 1. CRITICAL: Reset index to ensure alignment between IDs and Spectra
    # This prevents index mismatches if the input df was shuffled or filtered
    df_clean = df.reset_index(drop=True)
    
    ids = df_clean.iloc[:, 0]      # Column 0 is ID
    spectra = df_clean.iloc[:, 1:] # Column 1+ is Spectra

    # 2. Preprocess with SG
    sg = SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=1, mode='nearest')
    spectra_processed_np = sg.fit_transform(spectra) # Returns NumPy Array
    
    spectra_processed = pd.DataFrame(spectra_processed_np, index=df_clean.index)
    
    # 3. Create containers for results
    outlier_series_dmodx = pd.Series(index=df_clean.index, data=False)
    outlier_series_t2 = pd.Series(index=df_clean.index, data=False)
    outlier_series_q = pd.Series(index=df_clean.index, data=False)

    # 4. Loop through each SampleID group
    unique_ids = ids.unique()
    
    for uid in unique_ids:
        # Get indices for this group
        current_indices = ids[ids == uid].index
        
        # Extract the processed spectra for this group (now works because it's a DataFrame)
        group_spectra = spectra_processed.loc[current_indices]
        
        # --- Safety Check for Small Groups ---
        # PCA cannot find more components than samples. 
        # If you have 6 replicates, n_components=0.95 is fine (it picks float variance),
        # but we must ensure we don't crash on very small groups.
        n_samples = group_spectra.shape[0]
        n_features = group_spectra.shape[1]
        
        # PCA fit
        pca = PCA(n_components=n_components)
        pca.fit(group_spectra)
        
        # Initialize Detectors with the LOCAL model
        dmodx = DModX(model=pca, confidence=0.95)
        hotelling = HotellingT2(model=pca, confidence=0.95)
        qresiduals = QResiduals(model=pca, confidence=0.95)
        
        dmodx.fit(group_spectra)
        hotelling.fit(group_spectra)
        qresiduals.fit(group_spectra)

        # Predict and Store
        outlier_series_dmodx.loc[current_indices] = dmodx.predict(group_spectra)
        outlier_series_t2.loc[current_indices] = hotelling.predict(group_spectra)
        outlier_series_q.loc[current_indices] = qresiduals.predict(group_spectra)

    # 5. Format Output
    output = pd.DataFrame({
        'Sample_ID': ids,
        'DModX': outlier_series_dmodx,
        'HotellingT2': outlier_series_t2,
        'QResiduals': outlier_series_q
    })

    return output