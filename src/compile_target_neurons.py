import pandas as pd
import numpy as np

def compile_target_neurons(
    df, 
    target_count=83, 
    total_layers=32, 
    axis_start_fraction=0.45, 
    top_axis_threshold=50, 
    axis_min_count=4  # TWEAKED: Increased to 4
):
    df = df.copy()
    
    # Extract neuron_id and layer_num
    df['neuron_id'] = df['feature'].apply(lambda x: x.split('-')[1])
    df['layer_num'] = df['feature'].apply(lambda x: int(x.split('-')[0].split('_')[1]))
    df['rank'] = pd.to_numeric(df['rank'])

    # --- Criteria 1: The Robust Core ---
    core_df = df[df['selection_frequency'] == 1.0].copy()
    
    # --- Criteria 2: Heuristic Axis Expansion ---
    # DECOUPLED FIX: Detect structural axes independently of the Core.
    # We look at the top N candidates and count neuron ID occurrences.
    top_candidates = df.head(top_axis_threshold)
    axis_counts = top_candidates['neuron_id'].value_counts()
    
    # Select neurons that meet the minimum count threshold
    strong_axes = axis_counts[axis_counts >= axis_min_count].index.tolist()
    
    # Calculate heuristic layer boundaries (0.45 * 32 = 14)
    start_layer = int(total_layers * axis_start_fraction) 
    end_layer = total_layers - 1 # Layer 31
    
    interpolated_features = []
    for n_id in strong_axes:
        for l in range(start_layer, end_layer + 1):
            feature_name = f"layer_{l}-{n_id}"
            interpolated_features.append({
                'feature': feature_name,
                'neuron_id': n_id,
                'layer_num': l,
                'rank': float('inf'), # Fake rank so real SVM features take precedence
                'selection_frequency': 0.0
            })
                
    interpolated_df = pd.DataFrame(interpolated_features)
    
    # Combine Core IDs and Strong Axis IDs to pull their real CSV data
    ids_to_keep = set(core_df['neuron_id'].unique()).union(set(strong_axes))
    axis_expansion_df = df[df['neuron_id'].isin(ids_to_keep)].copy()
    
    # Merge real features with the heuristic features
    combined_axis_df = pd.concat([axis_expansion_df, interpolated_df])
    
    # Sort by rank and drop duplicates. Real CSV ranks beat heuristic 'inf' ranks.
    current_selection = combined_axis_df.sort_values('rank').drop_duplicates(subset=['feature'], keep='first')
    
    # --- Criteria 3: Rank Fill ---
    slots_remaining = target_count - len(current_selection)
    
    if slots_remaining > 0:
        remaining_candidates = df[~df['feature'].isin(current_selection['feature'])]
        fill_df = remaining_candidates.sort_values('rank').head(slots_remaining)
        final_list = pd.concat([current_selection, fill_df])
    else:
        final_list = current_selection.sort_values('rank').head(target_count)
        
    return final_list['feature'].tolist()
