import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

get_le = lambda x : x['simulation_data']['total_lex_err'][0].mean(0)
get_se = lambda x : x['simulation_data']['total_sem_err'][0].mean(0)
get_lb = lambda x : -x['simulation_data']['total_lex_bias'][0].mean(0)
get_sb = lambda x : -x['simulation_data']['total_sem_bias'][0].mean(0)
get_lse = lambda x: get_le(x) + get_se(x)
get_lsb = lambda x: get_lb(x) + get_sb(x)
get_maxlex = lambda x: x['simulation_data']['max_lex_state_activation'][0].mean(0)


func_dic = {
    'lex_error': get_le,
    'sem_error': get_se,
    'lexsem_error': get_lse,
    'lex_bias': get_lb,
    'sem_bias': get_sb,
    'lexsem_bias': get_lsb,
}


def extract_values(simulation, error_bias):
    if 'error' in error_bias:
        return get_le(simulation), get_se(simulation), get_lse(simulation)
    if 'bias' in error_bias:
        return get_lb(simulation), get_sb(simulation), get_lsb(simulation)

def extract_diff(sim_a, sim_b, error_bias):
    a1, b1, c1 = extract_values(sim_a, error_bias)
    a2, b2, c2 = extract_values(sim_b, error_bias)
    return a1 - a2, b1 - b2, c1 - c2


class CustomPlot:
    def __init__(self, simulation_title):
        self.simulation_title = simulation_title
        self.fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 2)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_ylabel('Lexical')
        self.ax2 = self.fig.add_subplot(gs[1, 0], sharex=self.ax1, sharey=self.ax1)
        self.ax2.set_ylabel('Semantic')
        self.ax3 = self.fig.add_subplot(gs[:, 1], sharex=self.ax1, sharey=self.ax1)
        self.ax3.set_ylabel('Lexico-semantic')
        plt.tight_layout()
        self.fig.suptitle(f'{simulation_title}', fontsize=16)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

    def update(self, simulation, color='blue', linestyle='-', sim_label = ''):
        for error_bias in ['error']:
            a, b, c = extract_values(simulation, error_bias)

            self.ax1.plot(a, color=color, linestyle=linestyle)
            self.ax2.plot(b, color=color, linestyle=linestyle)
            self.ax3.plot(c, color=color, linestyle=linestyle, label=sim_label)

            handles, labels = self.ax3.get_legend_handles_labels()
            unique = dict(zip(labels, handles))  # keeps last occurrence
            self.ax3.legend(unique.values(), unique.keys())

            self.fig.canvas.draw_idle()
    def update_stretch(self, simulation, trial=None, lexicon=None, color='blue', linestyle='-', sim_label = ''):
        # one condition: predicted {x} w/ {cloze_value} bottomup = {z}
        for error_bias in ['error']:
            a, b, c = extract_values(simulation, error_bias)
            padding_size = 41 - a.shape[0]
            a = np.append(np.zeros(padding_size), a)
            b = np.append(np.zeros(padding_size), b)
            c = np.append(np.zeros(padding_size), c)


            self.ax1.plot(np.arange(-20,21), a, color=color, linestyle=linestyle)
            self.ax1.set_ylim(-500,1500)
            self.ax2.plot(np.arange(-20,21), b, color=color, linestyle=linestyle)
            self.ax2.set_ylim(-500,1500)
            self.ax3.plot(np.arange(-20,21), c, color=color, linestyle=linestyle, label=sim_label)
            self.ax3.set_ylim(-500,1500)


            handles, labels = self.ax3.get_legend_handles_labels()
            unique = dict(zip(labels, handles))  # keeps last occurrence
            self.ax3.legend(unique.values(), unique.keys())

            self.fig.canvas.draw_idle()

    def update_lex_activation(self, simulation, word_idx, trial, color='blue', linestyle='-', sim_label=''):
        """

        """
        lex = simulation['simulation_data']['lex_state']  # (words, trials, 40 iterations)
        # 
        #trials = np.arange(lex.shape[1])
        a = self._pad_to_stretch(lex[word_idx, trial, :])
        xs = np.arange(-20, 21)

        if not hasattr(self, 'ax_lexact'):
            self.fig_lexact, self.ax_lexact = plt.subplots(figsize=(8, 6))
            self.ax_lexact.set_xlabel('Iterations')
            self.ax_lexact.set_ylabel('Lexical activation')
            self.fig_lexact.suptitle(f'{self.simulation_title} lexical activation', fontsize=16)

        self.ax_lexact.plot(xs, a, color=color, linestyle=linestyle, label=sim_label)
        handles, labels = self.ax_lexact.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.ax_lexact.legend(unique.values(), unique.keys())
        self.fig_lexact.canvas.draw_idle()

    def update_l1_l2_activation(self, simulation, en_idx, l2_idx, color='blue', sim_label=''):
        lex = simulation['simulation_data']['lex_state']  # (words, trials, iterations)
        trials = np.arange(lex.shape[1])
        l1 = self._pad_to_stretch(lex[np.asarray(en_idx), trials, :].mean(0))
        l2 = self._pad_to_stretch(lex[np.asarray(l2_idx), trials, :].mean(0))
        xs = np.arange(-20, 21)

        if not hasattr(self, 'ax_l1l2'):
            self.fig_l1l2, self.ax_l1l2 = plt.subplots(figsize=(8, 6))
            self.ax_l1l2.set_xlabel('Iterations')
            self.ax_l1l2.set_ylabel('Lexical activation')
            self.fig_l1l2.suptitle(f'{self.simulation_title} L1 vs L2', fontsize=16)

        self.ax_l1l2.plot(xs, l1, color=color, linestyle='-', label=f'{sim_label} L1')
        self.ax_l1l2.plot(xs, l2, color=color, linestyle='--', label=f'{sim_label} L2')
        handles, labels = self.ax_l1l2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.ax_l1l2.legend(unique.values(), unique.keys())
        self.fig_l1l2.canvas.draw_idle()

    def update_lang_lex_activation(self, simulation, trial, lexicon, color='blue',sim_label=''):
        # one condition: predicted {x} w/ {cloze_value} bottomup = {z}, plotting_activation {w}
        #
        orth_input = simulation['sim_input_bottomup'][trial]
        topdown = simulation['sim_input_topdown']
        pred_concept = topdown[trial] if len(topdown) else lexicon.words[trial]
        cloze_val, c1_val, c2_val = simulation['cloze'], simulation['c1'], simulation['c2']
        #pdb.set_trace()
        # trial gives orth input
        # find language
        # update l1 and l2
        print(c1_val)
        print(c2_val)

        #  TODO: mapping of warods
        plot_act_l1_idx = trial
        plot_act_l1_word = lexicon.words[plot_act_l1_idx] # map words here
        plot_act_l2_idx = trial + 1568
        plot_act_l2_word = lexicon.words[plot_act_l2_idx] # map words here
        
        lex = simulation['simulation_data']['lex_state']  # (words, trials, iterations)
        trials = np.arange(lex.shape[1])
        l1 = self._pad_to_stretch(lex[plot_act_l1_idx, trial, :])
        l2 = self._pad_to_stretch(lex[plot_act_l2_idx, trial, :])
        xs = np.arange(-20, 21)

        if not hasattr(self, 'ax_l1l2'):
            self.fig_l1l2, self.ax_l1l2 = plt.subplots(figsize=(8, 6))
            self.ax_l1l2.set_xlabel('Iterations')
            self.ax_l1l2.set_ylabel('Lexical activation')
            self.fig_l1l2.suptitle(f'{self.simulation_title} L1 vs L2', fontsize=16)
        
        sim_label_l1 = f"Predicted {pred_concept} w/ {cloze_val:.4f} bottomup = {orth_input}, plotting_activation: {plot_act_l1_word}"
        sim_label_l2 = f"Predicted {pred_concept} w/ {cloze_val:.4f} bottomup = {orth_input}, plotting_activation: {plot_act_l2_word}"


        self.ax_l1l2.plot(xs, l1, color=color, linestyle='-', label=f'{sim_label_l1}')
        self.ax_l1l2.plot(xs, l2, color=color, linestyle='--', label=f'{sim_label_l2}')
        handles, labels = self.ax_l1l2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.ax_l1l2.legend(unique.values(), unique.keys())
        self.fig_l1l2.canvas.draw_idle()
    
    def update_stretch_diff(self, sim_a, sim_b, color='blue', linestyle='-', sim_label='', ylim=(-100, 150)): 
        """Plot (sim_a - sim_b) for each PE component."""
        
        for error_bias in ['error']:
            a, b, c = extract_diff(sim_a, sim_b, error_bias)
            a, b, c = [self._pad_to_stretch(x) for x in (a, b, c)]
            xs = np.arange(-20, 21)
            
            self.ax1.plot(xs, a, color=color, linestyle=linestyle)
            self.ax1.set_ylim(*ylim)
            
            self.ax2.plot(xs, b, color=color, linestyle=linestyle)
            self.ax2.set_ylim(*ylim)
            
            self.ax3.plot(xs, c, color=color, linestyle=linestyle, label=sim_label)
            self.ax3.set_ylim(*ylim)
            
        handles, labels = self.ax3.get_legend_handles_labels()
        self.ax3.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
        self.fig.canvas.draw_idle()
    
    def _pad_to_stretch(self, arr, target_len=41, x_range=(-20, 21)):
        padding_size = target_len - arr.shape[0]
        return np.append(np.zeros(padding_size), arr)


# def make_plot(sim = simulation_post, sim_description = 'expcond'):
    # expunexp, higmedlowclz = sim_description.split('_')[0], sim_description.split('_')[1]
'''title = sim_description #f'{expunexp}_{higmedlowclz}'
plotter = CustomPlot(title)


plotter.update_stretch(sim[f'expcond_highclz'], color= 'k', linestyle = '-' , sim_label = f'expcond_highclz')
plotter.update_stretch(sim[f'{title}_medclz'], color= 'k', linestyle = '--' , sim_label = f'{title}_medclz')
plotter.update_stretch(sim[f'{title}_lowclz'], color= 'k', linestyle = ':' , sim_label = f'{title}_lowclz')'''
