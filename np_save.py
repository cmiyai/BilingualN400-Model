import numpy as np

bilingual_idx = np.concatenate((
    np.arange(0, 50),
    np.arange(1568, 1618),
))
np.save("./helper_txt_files/bilingual_13_word_idx.npy", bilingual_idx)