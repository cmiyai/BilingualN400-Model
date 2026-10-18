import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from orth_neighborhood_utils import *
import bz2
import _pickle as cPickle
from get_summary import get_summary
from os.path import exists
from collections import OrderedDict
import pdb
import logging
from pathlib import Path

num = 2
LOG_PATH = Path(f"./logs/sim_debug{num}.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("sim_debug")
logger.setLevel(logging.INFO)
if not logger.handlers:  # avoid duplicate handlers on reload
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)

class Simulation:
    def __init__(self, **kwargs):
    
        default_args = {
            "sim_input_bottomup":[],# list of bottom-up words
            "sim_input_topdown":[],# list of top-down words
            "clamp_iterations" : 1, # how many iterations to run the input
            "blanks_before_clamp":0 , # number of blank trials before clamping input
            "BU_TD_mode": "bottom_up", # inputs can be "bottom_up" or "top_down"
            "cloze" : None, # this only applies for top-down inputs
            "prevSim" : {}, # if this is a simulation running immediately after another simulation
            "EPSILON1" : 0.005, # hyperparameter for elementwise division
            "EPSILON2" : 0.0001,  # hyperparameter for elementwise multiplication
            "preact_resource" : 1,
            "sim_filename": False,
            "iterations_this_run" : 0,
            "simulation_data": {},
            "c1" : .095,
            "c2" : .005
        }
        # if this simulation has already been run, load it from memory.
        if exists(f"./data/{kwargs.get('sim_filename')}" + '.pbz2'):
            data = bz2.BZ2File(f"./data/{kwargs.get('sim_filename')}" + '.pbz2', 'rb')
            data = cPickle.load(data)
            self.__dict__.update(data)
        else:
            for (prop, default) in default_args.items():
                setattr(self, prop, kwargs.get(prop, default))
            # if applicable, set the initial state of the current simulation to the final state of the previous simulation.
            if self.prevSim.get('sim_filename',None) != None:
                self.load_prevSimulation()
            # otherwise, set the initial state of the current simulation to the default values
            else:
                self.define_weights()
                self.define_statespace()            
            assert self.BU_TD_mode in ["bottom_up", "top_down", "both"] # only the orthographic or contextual layers may have inputs defined for them
            if self.BU_TD_mode == "bottom_up" or self.BU_TD_mode == "both":
                # get orthographic vectors for each word input
                self.bottomup_input = wordlist_to_orth(self.sim_input_bottomup)
            if self.BU_TD_mode == "top_down" or self.BU_TD_mode == "both":
                # get conceptual vectors for each word input
                self.topdown_input = wordlist_to_ctx(self.sim_input_topdown, self.lexicon.words, cloze = self.cloze, preact_resource = self.preact_resource)
            self.run_simulation()

    def load_prevSimulation(self):
        '''
        load a copy of the previous simulation, update the statespace, weights, lexicon and current_iteration of the model;
        then delete the prevSimulation copy. Keeping it around consumes too much memory.
        '''
        data = bz2.BZ2File(f'./data/{self.prevSim["sim_filename"]}' + '.pbz2', 'rb')
        data = cPickle.load(data)
        # specify which aspects of the previous simulation's initial state you want to load
        data_to_load = ['weights','statespace', 'current_iteration','lexicon','simulation_data', 'I1', 'I2', 'num_trials']
        # load each value (e.g. statespace) from the previous simulation to the current simulation
        for key in data_to_load:
            setattr(self, key, copy.deepcopy(data[key]))
        # clear the memory from the previous simulation 
        data.clear()

    def define_statespace(self):
        # define the model representations at the orthographic, lexical, semantic and conceptual levels
        self.num_trials = max(len(self.sim_input_bottomup), len(self.sim_input_topdown))

        self.statespace = {'orth': np.ones((4,104,self.num_trials))/26, # first four dims are st/r/tdB/PE respectively
                            'lex': np.ones((4,self.lexicon.size,self.num_trials))/self.lexicon.size,
                            'sem': np.ones((4,self.lexicon.semfeatmatrix.shape[0], self.num_trials))/self.lexicon.semfeatmatrix.shape[0],
                            'ctx': np.ones((4,self.lexicon.ctxsize,self.num_trials))/self.lexicon.ctxsize}
        self.statespace['ctx'][1:,:,:] = np.nan # contextual level doesn't have reconstructions, top-down biases or prediction errors 
        self.statespace['lex'][2,:,:] = np.zeros((self.lexicon.size,self.num_trials))
        self.statespace['sem'][2,:,:] = np.zeros((self.lexicon.semfeatmatrix.shape[0], self.num_trials))
        self.current_iteration = 0

    def define_weights(self):
        #x = np.load('./helper_txt_files/gpt_semfeatmatrix_threshold2_fast.npy')
        #print(x)
        self.weights = {}
        self.lexicon = Lexicon()
        W1 = wordlist_to_orth(self.lexicon.words).T
        # define orthographic-to-lexical mapping (and vice versa)
        self.weights['divide_wt_O_to_L'] = np.dot(np.block([W1, np.eye(self.lexicon.size)]), np.ones((self.lexicon.size + 104,1)))
        self.weights['O_to_L'] = np.divide(W1, self.weights['divide_wt_O_to_L'])
        self.weights['L_to_O'] = self.weights['O_to_L'].T
        print(f"W1 orth to lex: {self.weights['O_to_L'].shape}")


        # define lexical-to-semantic mapping (and vice versa)
        self.weights['L_to_S'] = self.lexicon.semfeatmatrix
        self.weights['divide_wt_L_to_S'] = np.dot(np.block([self.weights['L_to_S'],  np.eye(self.lexicon.semfeatmatrix.shape[0])]), np.ones((self.lexicon.size + self.lexicon.semfeatmatrix.shape[0],1)))  
        self.weights['L_to_S'] = np.divide(self.weights['L_to_S'], self.weights['divide_wt_L_to_S']) #breakpoint2
        print(f"W2 lex to sem: {self.weights['L_to_S'].shape}")
        self.weights['S_to_L'] = self.weights['L_to_S'].T
        # define semantic-to-conceptual mapping (and vice versa)

        self.weights['S_to_C'] = self.lexicon.semfeatmatrix_C.T
        self.weights['divide_wt_S_to_C'] = np.dot(self.weights['S_to_C'],np.ones((self.lexicon.semfeatmatrix_C.shape[0],1)))
        self.weights['S_to_C'] = np.divide(self.weights['S_to_C'], self.weights['divide_wt_S_to_C'])
        print(f"W3 sem to ctx: {self.weights['S_to_C'].shape}")
        self.weights['C_to_S'] = self.weights['S_to_C'].T
        # define identity matrix normalization
        self.I1 = np.divide(np.eye(self.lexicon.size),self.weights['divide_wt_O_to_L'])
        self.I2 = np.divide(np.eye(self.lexicon.semfeatmatrix.shape[0]),self.weights['divide_wt_L_to_S'])

        # add frequency bias to non-zero feedback weights
        # self.weights['L_to_O'] = np.multiply(self.weights['L_to_O'] + self.lexicon.frequency, self.lexicon.orthmatrix > 0)
        print(self.weights['O_to_L'].shape)
        print(f"shud be (3136, 1) and is: {self.lexicon.frequency.T.shape}")
        print(self.lexicon.orthmatrix.T.shape)

        self.weights['O_to_L'] = np.multiply(self.weights['O_to_L'] + self.lexicon.frequency.T, self.lexicon.orthmatrix.T > 0) 
        # self.weights['S_to_L']  = np.multiply(self.weights['S_to_L'] + self.lexicon.frequency.T, self.lexicon.semfeatmatrix.T > 0)
        # self.weights['C_to_S'] = np.multiply(self.weights['C_to_S'] + self.lexicon.frequency, self.lexicon.semfeatmatrix > 0)

    def run_one_iteration(self):
        def eps_div(x,y):
            return x / np.maximum(self.EPSILON1, y)
        def eps_mul(x,y):
            return np.maximum(self.EPSILON2, x) * y 
        def get_wt(x,y):
            return ''.join(['OLSC'[x],'_to_','OLSC'[y]])

        levels = ['orth','lex','sem','ctx']
        kinds = ['state', 'tdR', 'tdB', 'PE']

        # compute states (ST), prediction errors (PE) and top-down biases (tdB) in that order for each level of representation (orth -> lex -> sem) 

        # for the orth level in bottom-up mode, set the input either as blanks or self.input depending on number of blank iterations
        if self.iterations_this_run < self.blanks_before_clamp:
            self.statespace['orth'][0] = np.zeros((self.statespace['orth'][0].shape))
        else:
            if self.BU_TD_mode == "top_down":
                self.statespace['orth'][0] = eps_mul(self.statespace['orth'][0], self.statespace['orth'][2])
            else:
                self.statespace['orth'][0] = self.bottomup_input # ST_orth <- input
        self.statespace['orth'][3] = eps_div(self.statespace['orth'][0] , self.statespace['orth'][1]) # PE_orth <- max(eps2, ST_orth) ./ max(tdR_orth, eps1)
        self.statespace['orth'][2] =  eps_div(self.statespace['orth'][1], self.statespace['orth'][0]) # tdB_orth <- tdR_orth ./ max(ST_orth,eps1); won't be needed if mode is bottom up
        
        # Bilingual constants implementation
        # constant values
        # language constant vector
        # pdb.set_trace()
        lang_const = np.zeros_like(self.statespace['lex'][2]) #
        #pdb.set_trace()
        #lang_const = self.c1
        lang_const[:self.lexicon.size//2, :] = self.c1
        lang_const[self.lexicon.size//2 : , :] = self.c2
        #pdb.set_trace()

        # NEW LANGUAGE UPDATE
        # pdb.set_trace()
        pre_lang = self.weights['O_to_L'] @ self.statespace['orth'][3] + self.I1 @ self.statespace['lex'][2]
        lang_update = pre_lang + lang_const
        print("PreLanguage Lexical Weight Update:")
        print(np.array2string(pre_lang, formatter={'float_kind': lambda x: f"{x:.4f}"}))
        logger.info(np.array2string(pre_lang, formatter={'float_kind': lambda x: f"{x:.4f}"})) 
        print("LanguageConstant Lexical Weight Update:")
        print(np.array2string(lang_update, formatter={'float_kind': lambda x: f"{x:.4f}"}))
        logger.info(np.array2string(lang_update, formatter={'float_kind': lambda x: f"{x:.4f}"})) 


        lex_update_term = self.weights['O_to_L'] @ self.statespace['orth'][3] + self.I1 @ self.statespace['lex'][2] + lang_const # update <- W1*PE_orth + I1*tdB_lex + lang_const
        
        self.statespace['lex'][0] = eps_mul(self.statespace['lex'][0], lex_update_term) # ST_lex <- max(eps2, ST_lex) .* update
        self.statespace['lex'][3] = eps_div(self.statespace['lex'][0], self.statespace['lex'][1]) # PE_lex <- ST_lex ./ max(tdR_lex, eps1)
        self.statespace['lex'][2] = eps_div(self.statespace['lex'][1], self.statespace['lex'][0]) # tdB_lex <- tdR_lex ./ max(ST_lex,eps1)
        
        sem_update_term = self.weights['L_to_S'] @ self.statespace['lex'][3] + self.I2 @ self.statespace['sem'][2] # update <- W2*PE_lex + I2*tdB_sem
        self.statespace['sem'][0] = eps_mul(self.statespace['sem'][0], sem_update_term) # ST_sem <- max(eps2, ST_sem) .* update
        self.statespace['sem'][3] = eps_div(self.statespace['sem'][0], self.statespace['sem'][1]) # PE_sem <- ST_sem ./ max(tdR_sem, eps1)
        self.statespace['sem'][2] = eps_div(self.statespace['sem'][1], self.statespace['sem'][0]) # tdB_sem <- tdR_sem ./ max(ST_sem,eps1)
        
        # ST_ctx <- max(eps2, ST_ctx) .* update if in bottom-up mode; otherwise set it to the pseudoprobability distribution
        ctx_update_term = self.weights['S_to_C'] @ self.statespace['sem'][3]
        if self.BU_TD_mode == "bottom_up":
            self.statespace['ctx'][0]  = eps_mul(self.statespace['ctx'][0], ctx_update_term)
        else:
            #self.statespace['ctx'][0]  = self.input 
            self.statespace['ctx'][0]  = eps_mul(self.statespace['ctx'][0], ctx_update_term)
            topdown_input_inds = np.array([list(self.lexicon.words).index(w) for w in self.sim_input_topdown])
            for input_idx, trial_idx in zip(topdown_input_inds, np.arange(self.num_trials)):
                # clamp the contextual state unit to either the updated value or the clamped
                self.statespace['ctx'][0][input_idx, trial_idx] = np.max(self.topdown_input) 
            # self.statespace['ctx'][0][topdown_input_inds,np.arange(self.num_trials)]  = np.max(self.topdown_input) #(1568?, 512)
        # compute all reconstructions
        for lvl in range(3):
            self.statespace[levels[lvl]][1] = self.weights[get_wt(lvl+1, lvl)] @ self.statespace[levels[lvl+1]][0] # tdR_lowerlevel = V_higher_to_lower * ST_higherlevel      
        
        self.current_iteration += 1
        self.iterations_this_run += 1
        print(f'Just finished iteration {self.current_iteration}')
        print(f"lex_update_term mean={np.mean(lex_update_term):.4f}, max={np.max(lex_update_term):.4f}")
        print(f"lex state mean={np.mean(self.statespace['lex'][0]):.4f}, max={np.max(self.statespace['lex'][0]):.4f}")
        print(f"lex PE mean={np.mean(self.statespace['lex'][3]):.4f}, max={np.max(self.statespace['lex'][3]):.4f}")
        print(f"sem state mean={np.mean(self.statespace['sem'][0]):.4f}, max={np.max(self.statespace['sem'][0]):.4f}")
        print(f"sem PE mean={np.mean(self.statespace['sem'][3]):.4f}, max={np.max(self.statespace['sem'][3]):.4f}")
        print(f"total lex err={np.sum(self.statespace['lex'][3]):.4f}")
        print(f"total sem err={np.sum(self.statespace['sem'][3]):.4f}")
        print(f"total lexsem err={np.sum(self.statespace['lex'][3]) + np.sum(self.statespace['sem'][3]):.4f}")
    def extract_info(self):
        # take the statespace at each iteration and extract an arbitrary value from it
        if self.simulation_data == {}:
            self.simulation_data = copy.deepcopy(get_summary(self))
        else:
            for key,val in get_summary(self).items():
                self.simulation_data[key] = np.dstack((self.simulation_data[key], val))
    def run_simulation(self):
        if self.prevSim == {}:
            self.extract_info() # if there was a prevSim, this info has already been extracted. This line basically avoids extracting the same info twice
        for _ in range(self.blanks_before_clamp + self.clamp_iterations):
            self.run_one_iteration()
            self.extract_info()
        if self.sim_filename != None:
            with bz2.BZ2File(f'./data/{self.sim_filename}.pbz2', 'w') as f:
                save_dict = {key: val for key,val in self.__dict__.items()} 
                cPickle.dump(save_dict, f)

class Lexicon:
    def __init__(self, **kwargs):
        # import the lexical characteristics

        semantic_feature_information =  pd.read_csv('./helper_txt_files/threshold2_df_fast_bilingual13.csv')
        # define the lexicon based on words that have semantic feature information
        word_lang = semantic_feature_information[["Word", "language"]].drop_duplicates()
        self.words = word_lang["Word"].to_numpy()
        self.word_language = word_lang["language"].to_numpy()
        # define the orthographic representations as slot-coded representations (four concatenated one-hot vectors to represent four letter words)
        self.orthmatrix = wordlist_to_orth(self.words)

        # define the lexical representations as one-hot vectors
        self.size = self.words.shape[0]
        self.lexicalmatrix = np.eye(self.size)
        # context size
        self.ctxsize = self.words.shape[0]//2
        # define the semantic feature labels
        self.semantic_feature_labels = np.array(list(OrderedDict.fromkeys(semantic_feature_information['features'])))
        self.language =  np.array(list(OrderedDict.fromkeys(semantic_feature_information['language'])))
        ################################################################
        # this block serves to retrieve lexical frequency information
        # load the subtlex data
        subtlexus_frequency = pd.read_csv('./helper_txt_files/SUBTLEXus2007.csv') 
        # convert everything to lowercase
        subtlexus_frequency['Word'] = subtlexus_frequency['Word'].astype(str).apply(lambda x: x.lower())
        # retrieve the frequency values corresponding to english
        half = self.size // 2 # halve the lexicon
        english_words = list(self.words[:half])
        filtered_df = subtlexus_frequency[subtlexus_frequency['Word'].isin(list(english_words))]
        filtered_df['Word'] = pd.Categorical(filtered_df['Word'], categories=list(english_words), ordered=True)
        filtered_df = filtered_df.sort_values('Word')
        logFreq = filtered_df['Lg10WF']
        min_freq_score, max_freq_score = 0,0.1
        scale_shift_factors = (
            np.linalg.inv(
                np.array([[np.min(logFreq), 1],[np.max(logFreq), 1]])
            ) 
            @ np.array([min_freq_score,max_freq_score])
        )
        self.frequency = scale_shift_factors[0] * logFreq.to_numpy() + scale_shift_factors[1]

        self.frequency = np.expand_dims(self.frequency,axis = 0).astype('float32')

        self.frequency = np.tile(self.frequency, (1,2)) # duplicated rows for double the lexicon
        ################################################################
        # load the semantic feature matrix 
        self.semfeatmatrix = np.load('./helper_txt_files/gpt_semfeatmatrix_threshold2_fast.npy') # loading monolingual matrix
        self.semfeatmatrix_C = self.semfeatmatrix.copy()
        self.semfeatmatrix = np.tile(self.semfeatmatrix, (1,2))
        self.num_semfeats = np.sum(self.semfeatmatrix, axis = 0)
        # The median is 9 features; concrete words are defined to have 10+ features:
        self.concreteness = self.num_semfeats > np.median(self.num_semfeats)
        self.concreteness = self.concreteness.astype('int32') # makes concreteness 1 vs 0

        # find all words that overlap in 3 places with each word (e.g. TROD has PROD and TROT as neighbors)
        # use this to define the orthographic neighborhood size.
        ONsize = ((self.orthmatrix.T @ self.orthmatrix) == 3).sum(axis = 0)
        self.ONsize = np.expand_dims(ONsize, axis = 0)

        
        
    
    
        