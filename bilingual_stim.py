import numpy as np
from PredictiveCoding_Model import Lexicon
# USE smaller bilingaul stim npy file

lexicon = Lexicon()

bilingual_idx = np.concatenate((
    np.arange(0, 50),
    np.arange(1568, 1618)
))

np.save('./helper_txt_files/bilingual_50_word_idx.npy', bilingual_idx)