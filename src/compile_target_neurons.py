import pandas as pd

def compile_target_neurons(df, target_count=80):
    # Assuming format "layer_X-neuron_Y"
    df['neuron_id'] = df['feature'].apply(lambda x: x.split('-')[-1])

    # --- Criteria 1: The Robust Core ---
    # Select features that appeared in 100% of folds (freq == 1.0)
    core_df = df[df['selection_frequency'] == 1.0].copy()
    
    # --- Criteria 2: Axis Expansion ---
    # Identify unique neuron IDs from the core
    core_neuron_ids = core_df['neuron_id'].unique()
    # Find ALL instances of these IDs in the full dataset (even low ranks)
    # This captures the "vertical" structure (e.g., the 2058 axis)
    axis_expansion_df = df[df['neuron_id'].isin(core_neuron_ids)].copy()
    
    # --- Criteria 3: Rank Fill ---
    # Combine Core + Axis (drop duplicates)
    current_selection = axis_expansion_df.drop_duplicates(subset=['feature'])
    
    slots_remaining = target_count - len(current_selection)
    
    if slots_remaining > 0:
        remaining_candidates = df[~df['feature'].isin(current_selection['feature'])]
        fill_df = remaining_candidates.sort_values('rank').head(slots_remaining)
        final_list = pd.concat([current_selection, fill_df])
    else:
        final_list = current_selection.sort_values('rank').head(target_count)
        
    return final_list['feature'].tolist()
