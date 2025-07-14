# demo.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model import WDSGformer

def load_sample_data(data_path: Path):
    """Loads sample data from the specified directory."""
    try:
        turbine_info = pd.read_csv(data_path / "turbine_info.csv")
        scada_data = pd.read_csv(data_path / "scada_data.csv")
        # weather_data is not directly used in this simplified demo, but we check for its existence
        weather_data_path = data_path / "weather_data.csv"
        if not weather_data_path.exists():
            raise FileNotFoundError(f"{weather_data_path} not found.")
        print("Sample data loaded successfully.")
        return turbine_info, scada_data
    except FileNotFoundError as e:
        print(f"Error: Could not find sample data files. {e}")
        print("Please make sure 'turbine_info.csv', 'scada_data.csv', and 'weather_data.csv' are in the 'sample_data' directory.")
        return None, None

def prepare_input_tensors(turbine_info: pd.DataFrame, scada_data: pd.DataFrame, 
                          input_len: int, d_feature: int, num_weather: int, weather_feat_dim: int):
    """Constructs a batch of input tensors from the sample data."""
    num_turbines = len(turbine_info)
    
    # 1. Turbine features tensor
    turbine_x = np.zeros((1, input_len, num_turbines, 5), dtype=np.float32) # B, T, N_t, F_t
    for i in range(num_turbines):
        power_col, wind_col = f'power_{i}', f'wind_speed_{i}'
        if power_col in scada_data.columns and wind_col in scada_data.columns:
            # Ensure we don't go out of bounds if scada_data is shorter than input_len
            n_rows = min(len(scada_data), input_len)
            turbine_x[0, :n_rows, i, 0] = scada_data[power_col].values[:n_rows]
            turbine_x[0, :n_rows, i, 1] = scada_data[wind_col].values[:n_rows]
    
    # Add static geographical features, broadcasted across the time dimension
    geo_feats = turbine_info[['norm_lon', 'norm_lat', 'norm_elevation']].values
    turbine_x[0, :, :, 2:5] = geo_feats
    
    # 2. Weather features tensor (using random data for demo simplicity)
    weather_x = np.random.randn(1, input_len, num_weather, weather_feat_dim).astype(np.float32)

    # 3. Dynamic features for the graph bias (e.g., wind speed and power)
    dynamic_features = turbine_x[:, :, :, :d_feature]

    # 4. Node coordinates for spatial encoding
    turbine_coords = turbine_info[['norm_lat', 'norm_lon']].values
    weather_coords = np.random.rand(num_weather, 2)
    node_coords = np.vstack([turbine_coords, weather_coords])

    # Convert all to tensors
    return (torch.from_numpy(arr) for arr in [turbine_x, weather_x, dynamic_features, node_coords])

def main():
    """Main function to run the demo."""
    print("--- WD-SGformer Demo with Sample Data ---")
    
    # --- Parameters ---
    data_path = Path("./sample_data")
    input_len, pred_steps = 96, 48
    turbine_feat_dim, weather_feat_dim = 5, 21
    d_feature = 2 # e.g., wind speed and power
    num_weather = 20 # assumption for demo
    
    # --- Load Data ---
    turbine_info, scada_data = load_sample_data(data_path)
    if turbine_info is None: 
        return
    num_turbines = len(turbine_info)
        
    # --- Prepare Tensors ---
    turbine_x, weather_x, dynamic_features, node_coords = prepare_input_tensors(
        turbine_info, scada_data, input_len, d_feature, num_weather, weather_feat_dim
    )
    print(f"Model input tensors constructed. Shapes:")
    print(f"  - Turbine X: {turbine_x.shape}")
    print(f"  - Weather X: {weather_x.shape}")
    print(f"  - Dynamic Features: {dynamic_features.shape}")
    
    # --- Initialize Model ---
    model = WDSGformer(
        num_turbines=num_turbines, num_weather=num_weather,
        turbine_feat_dim=turbine_feat_dim, weather_feat_dim=weather_feat_dim,
        d_feature=d_feature, input_len=input_len, pred_steps=pred_steps,
        node_coords=node_coords.float()
    )
    print(f"\nModel initialized. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # --- Forward Pass ---
    try:
        model.eval()
        with torch.no_grad():
            output = model(turbine_x, weather_x, dynamic_features)
        print(f"\nForward pass successful!")
        print(f"Output shape: {output.shape} (Expected: [1, {pred_steps}])")
        assert output.shape == (1, pred_steps), "Output shape mismatch!"
        print("Assertion passed: Output shape is correct.")
    except Exception as e:
        import traceback
        print(f"\nError during forward pass: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()