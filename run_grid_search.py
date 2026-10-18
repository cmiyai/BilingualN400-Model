import numpy as np
import pandas as pd
from run_simulations import run_simulation, lexicon, cloze_levels, lex_c1, lex_c2, NUM_ITERS, unrelated_stims_1

def run_bilingual_grid_search(lex_L1_expected, lex_L2_translation, lex_L1_unrelated, lex_L2_unrelated, total_c=0.1, step=0.01):
    results = []
    
    # Constant sum search: maintain a fixed pool of language activation resources
    c1_values = np.arange(0.0, total_c + step, step)
    grid = [(np.round(c1, 3), np.round(total_c - c1, 3)) for c1 in c1_values]

    cloze_val = cloze_levels["high_cloze"]

    for c1, c2 in grid:
        print(f"Testing c1={c1}, c2={c2}...")
        
        # 1. TOP-DOWN: Pre-activate the context (Predicting the L1 concept)
        sim_preact = run_simulation(
            sim_input_topdown=lex_L1_expected,
            clamp_iterations=NUM_ITERS,
            BU_TD_mode="top_down",
            cloze=cloze_val,
            sim_filename=None, 
            c1=c1, c2=c2
        )
        
        # 2. BOTTOM-UP conditions
        conditions = {
            "L1_Expected": lex_L1_expected,
            "L2_Translation": lex_L2_translation,
            "L1_Unrelated": lex_L1_unrelated,
            "L2_Unrelated": lex_L2_unrelated
        }
        
        row_data = {'c1': c1, 'c2': c2}
        
        for cond_name, bottom_up_stimuli in conditions.items():
            sim_target = run_simulation(
                sim_input_bottomup=bottom_up_stimuli,
                cloze=cloze_val,
                clamp_iterations=NUM_ITERS,
                BU_TD_mode="bottom_up",
                prevSim=sim_preact,
                sim_filename=None, 
                c1=c1, c2=c2
            )
            
            data = sim_target['simulation_data']
            
            # Extract timecourses
            mean_lex_err_tc = np.mean(data['total_lex_err'][0], axis=0)
            mean_sem_err_tc = np.mean(data['total_sem_err'][0], axis=0)
            
            # Extract N400 in the target time window (e.g., iterations 2 through 11)
            time_window = slice(-NUM_ITERS + 1, -NUM_ITERS + 11)
            
            lex_N400 = np.mean(mean_lex_err_tc[time_window])
            sem_N400 = np.mean(mean_sem_err_tc[time_window])
            
            row_data[f'{cond_name}_Lex_PE'] = lex_N400
            row_data[f'{cond_name}_Sem_PE'] = sem_N400
            row_data[f'{cond_name}_Total_N400'] = lex_N400 + sem_N400
            
        results.append(row_data)
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # We will use lex_c1 and lex_c2 from run_simulations
    num_stims = min(len(lex_c1), len(lex_c2))
    L1_expected = lex_c1[:num_stims]
    L2_translation = lex_c2[:num_stims]
    
    # We use unrelated_stims_1 from run_simulations, splitting it into two parts for L1/L2
    # Ensure we don't go out of bounds if unrelated_stims_1 is small
    max_unrelated = len(unrelated_stims_1) // 2
    actual_stims = min(num_stims, max_unrelated)
    
    L1_unrelated = unrelated_stims_1[:actual_stims]
    L2_unrelated = unrelated_stims_1[actual_stims:actual_stims*2]
    
    print("Starting Bilingual Grid Search...")
    df = run_bilingual_grid_search(L1_expected[:actual_stims], 
                                   L2_translation[:actual_stims], 
                                   L1_unrelated, 
                                   L2_unrelated, 
                                   total_c=0.1, step=0.01)
    
    print("\nResults:")
    print(df)
    
    csv_filename = "bilingual_n400_grid_search.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nSaved results to {csv_filename}")
