import os
import random
import numpy as np
import pandas as pd

from plot import *
from get_summary import *
from orth_neighborhood_utils import *
from PredictiveCoding_Model import *
from stimulus_counterbalancing_script import *
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(1)
np.random.seed(1)


def run_simulation(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation(**kwargs)
    data,fname,sim_input_bottomup,sim_input_topdown = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input_bottomup, full_simulation.sim_input_topdown
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input_bottomup": sim_input_bottomup,
             "sim_input_topdown": sim_input_topdown}
    # return full_simulation
def run_simulation_comp(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation_comp(**kwargs)
    data,fname,sim_input_bottomup,sim_input_topdown = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input_bottomup, full_simulation.sim_input_topdown
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input_bottomup": sim_input_bottomup,
             "sim_input_topdown": sim_input_topdown}
    # return full_simulation



lexicon = Lexicon()
NUM_ITERS = 20
THRESHOLD = 4.0

wordlist = np.array(lexicon.words)

##################### DEFINE STIMULI FOR ALL CONDITIONS ##################### 

## create stimuli that are counterbalanced
# quads = find_counterbalanced_quads(lexicon,min_sem_overlap = 1)
# stim_dict = create_stimulus_lists(quads, lexicon)
# verify_stimulus_lists(stim_dict, lexicon)
# np.save('./helper_txt_files/standard_stims_word_idx.npy', stim_dict['standard_idx'])
# np.save('./helper_txt_files/sem_related_word_idx_nonneighbor.npy', stim_dict['sem_related_idx'])
# np.save('./helper_txt_files/fully_unrelated_word_idx_1.npy', stim_dict['unrelated_idx'])

standard_stims_word_idx = np.load('./helper_txt_files/standard_stims_word_idx.npy')
bilingual_stims_30_idx = np.load('./helper_txt_files/bilingual_50_word_idx.npy') # note 50
fully_unrelated_word_idx_1 = np.load('./helper_txt_files/fully_unrelated_word_idx_1.npy')
# sem_related_word_idx_nonneighbor = np.load('./helper_txt_files/sem_related_word_idx_nonneighbor.npy')

#standard_stims = wordlist[standard_stims_word_idx]
# print(standard_stims.shape)
bilingual_stims_30 = wordlist[bilingual_stims_30_idx]
print(bilingual_stims_30.shape)
print(bilingual_stims_30)
# pdb.set_trace()
unrelated_stims_1 = wordlist[fully_unrelated_word_idx_1]
# semrelated_stims = wordlist[sem_related_word_idx_nonneighbor]

# final_standard_idx = [list(lexicon.words).index(w) for w in standard_stims]
# final_unrelated_idx_1 = [list(lexicon.words).index(w) for w in unrelated_stims_1]
# final_semrelated_idx = [list(lexicon.words).index(w) for w in semrelated_stims]

# shared_feats = lexicon.semfeatmatrix[:,final_standard_idx] * lexicon.semfeatmatrix[:,final_semrelated_idx] 
# assert (lexicon.semfeatmatrix[:,final_unrelated_idx_1] * lexicon.semfeatmatrix[:,final_semrelated_idx]).sum() == 0

##################### LEXICAL EFFECT SIMULATIONS #####################


num_stims = 1400 
lex_c1 = lexicon.words[0:50]
print("lex1")
print(lex_c1)
lex_c2 = lexicon.words[1568:1618]
print("lex2")
print(lex_c2)

lex_bilingual = np.concatenate((lex_c1,lex_c2))
c2_index_start = 1568 # 30 inside the bilingual lexicon

c1_const = .09
c2_const = .01
bilingual_final_standard_idx = [list(lexicon.words).index(w) for w in bilingual_stims_30]
print(bilingual_final_standard_idx)

#standard_simulation = run_simulation(sim_input_bottomup = lex_bilingual, clamp_iterations = NUM_ITERS, sim_filename = f'bilingual_simulation_{c1_const:.2f}_{c2_const:.2f}', c1 = c1_const)
pdb.set_trace()
# for __ in loop over dc where c1+c2 sum to consts

# we want to double english lexicon for the context
ctx_doubled_english = np.repeat(lex_c1,2)
print(ctx_doubled_english)



# ##################### CONTEXTUAL EFFECT SIMULATIONS #####################
# Run the semantic priming simulation
# No semantic Priming for now
'''sem_priming = {'semunrelated': run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semunrelated5'),
                'semrelated': run_simulation(sim_input_bottomup = semrelated_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'sem_priming_semrelated5')}

# Run the repetition priming simulation
rep_priming = {'unrepeated': run_simulation(sim_input_bottomup = unrelated_stims_1[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_unrepeated5'),
               'repeated' : run_simulation(sim_input_bottomup = standard_stims[:num_stims], clamp_iterations =NUM_ITERS, blanks_before_clamp = 5, prevSim = standard_simulation, sim_filename = 'rep_priming_repeated5')}
'''

# Run the lexical predictability simulation
cloze_levels = {
    "low_cloze": 1/lexicon.size, 
                #"med_low_cloze": 0.25,
                #"med_high_cloze": 0.5,
                "high_cloze": 0.99}

cloze_simulations_preactivate_en = {}
cloze_simulations_bottomup_en = {}
cloze_simulations_preactivate_l2 = {}
cloze_simulations_bottomup_l2 = {}

############################################################################################################################################
# Entire Simulation: English
for key, val in cloze_levels.items():
    # pre-activate each of the 512 standard inputs from the TOP DOWN
    # (20 iterations)
    cloze_simulations_preactivate_en.update({key: run_simulation(sim_input_topdown = lex_c1, # maybe drop this: double english lexicon, indicies(concept1) < size(bilingual_leixonc/2) maps to english ... rest for l2
                                                              clamp_iterations =NUM_ITERS,
                                                              BU_TD_mode = "top_down", 
                                                              cloze = val,
                                                              sim_filename = f'cloze_simulations_preact_en{key}_2',
                                                              c1 = .045,
                                                              c2 = 0.005)})
    # present each of the 512 standard inputs from the BOTTOM UP
    # 20 iterations
    cloze_simulations_bottomup_en.update({key: run_simulation(sim_input_bottomup = lex_c1, # full bilingual lexicon
                                                           #sim_input_topdown = lex_c1, # double english concept
                                                           cloze = val,
                                                           clamp_iterations =NUM_ITERS,
                                                           BU_TD_mode = "bottom_up", 
                                                           prevSim = cloze_simulations_preactivate_en[key], 
                                                           sim_filename = f'cloze_simulations_bottomup_en{key}_2',
                                                           c1 = .045,
                                                            c2 = 0.005)})
    
# Entire Simulation: NonEnglish
for key, val in cloze_levels.items():
    # pre-activate each of the 512 standard inputs from the TOP DOWN
    # (20 iterations)
    cloze_simulations_preactivate_l2.update({key: run_simulation(sim_input_topdown = lex_c1, # maybe drop this: double english lexicon, indicies(concept1) < size(bilingual_leixonc/2) maps to english ... rest for l2
                                                              clamp_iterations =NUM_ITERS,
                                                              BU_TD_mode = "top_down", 
                                                              cloze = val,
                                                              sim_filename = f'cloze_simulations_preact_l2{key}_2',
                                                              c1 = .045,
                                                            c2 = 0.005)})
    # present each of the 512 standard inputs from the BOTTOM UP
    # 20 iterations
    cloze_simulations_bottomup_l2.update({key: run_simulation(sim_input_bottomup = lex_c2, # full bilingual lexicon
                                                           #sim_input_topdown = lex_c1, # double english concept
                                                           cloze = val,
                                                           clamp_iterations =NUM_ITERS,
                                                           BU_TD_mode = "bottom_up", 
                                                           prevSim = cloze_simulations_preactivate_l2[key], 
                                                           sim_filename = f'cloze_simulations_bottomup_l2{key}_2',
                                                           c1 = .045,
                                                            c2 = 0.005)})
    
pdb.set_trace()
#plotter = CustomPlot("Len50Lex")
#plotter1 = CustomPlot("Len50Lexp2")

#plotter.update_stretch(cloze_simulations_bottomup_en[f'high_cloze'], color= 'b', linestyle = '-' , sim_label = f'high_cloze_en')

get_maxlex = lambda x : x['simulation_data']['max_lex_state_activation'][0]
get_secmaxlex = lambda x : x['simulation_data']['secmax_lex_state_activation'][0]
get_maxlex_id = lambda x : x['simulation_data']['max_lex_state_identity'][0]
get_secmaxlex_id = lambda x : x['simulation_data']['secmax_lex_state_identity'][0]


maxlex_hc = get_maxlex(cloze_simulations_bottomup_en[f'high_cloze'])[0]
maxlex_id_hc = get_maxlex_id(cloze_simulations_bottomup_en[f'high_cloze'])[0]
secmaxlex_hc = get_secmaxlex(cloze_simulations_bottomup_en[f'high_cloze'])[0]
secmaxlex_id_hc = get_secmaxlex_id(cloze_simulations_bottomup_en[f'high_cloze'])[0]
print(maxlex_hc.shape)
print(lex_c1)
print(lex_c2)
pdb.set_trace()
plt.plot(maxlex_hc.T)
plt.plot(maxlex_id_hc.T)

plt.show()
pdb.set_trace()
print("hi")