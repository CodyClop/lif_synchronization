import numpy as np
import random

class LIFNeuron:
    """
        * tau_m : Leak strength
        * g_L : membrane permeability
        * tau_ref : Absolute refractory period ms (Relative refractory period is not implemented)
        * v_rest : resting membrane potential
        * v_reset : reset membrane potential after firing
        * v_thresh : threshold 
        * I_int : internal electrical activity leading to intrinsic oscillations
        * I_spike : spike intensity (negative for inhibitory neurons)
    """
    def __init__(self, tau_m=10, g_L=1, tau_ref=2, v_rest=-75, v_reset=-75, v_thresh=-55, I_int=30, I_spike=100, noise=0):
        self.tau_m = tau_m + np.random.standard_normal() * noise
        self.tau_ref = tau_ref + np.random.standard_normal() * noise
        self.v_rest = v_rest + np.random.standard_normal() * noise
        self.v_reset = v_reset + np.random.standard_normal() * noise
        self.v_thresh = v_thresh + np.random.standard_normal() * noise
        self.I_int = I_int + np.random.standard_normal() * noise
        self.v = v_rest + np.random.uniform() * 20
        self.g_L = g_L + np.random.standard_normal() * noise
        self.I_spike = I_spike + np.random.standard_normal() * 3 * noise
        self.t_ref = 0 # timer for refraction
    
    """
        updates neuron's membrane potential
        returns True if neuron fires (at the end of absolute refractory period)
        * dt : time elapsed since last update
        * I_ext : incoming electrical activity
    """
    def update(self, dt, I_ext=0):
        if self.t_ref > 0:
            self.t_ref -= dt
            return False

        # membrane potential update
        self.v += (-(self.v - self.v_rest) + (self.I_int + I_ext) / self.g_L) * (dt / self.tau_m)
        
        # beginning of refractory period
        if self.v > self.v_thresh:
            self.t_ref = self.tau_ref
            self.v = self.v_reset
            return True
        return False
    
class Network:
    """
        * n : number of neurons
        * d : density of the network, between 0 and 1
        * inhib_rate : percentage of inhibitory neurons, between 0 and 1
        * I_excitation : electrical power of excitatory spikes
        * I_inhibition : electrical power of inhibitory spikes
    """
    def __init__(self, n, d, inhib_rate, I_excitation, I_inhibition, noise=0):
        self.nodes = {}
        self.inlinks = {} # incoming arcs
        self.outlinks = {} # outgoing arcs
        self.max_degree_nodes = set() # to add random arcs
        self.add_nodes(int(n * inhib_rate), I_inhibition, noise)
        self.add_nodes(int(n * (1 - inhib_rate)), I_excitation, noise)
        self.add_links(int(n * n * d))
        self.remove_unlinked_nodes()
    

    # Adds n nodes to the graph
    def add_nodes(self, n, I_spike, noise):
        last_key = list(self.nodes.keys())[-1] if len(self.nodes.keys()) > 0 else -1
        for i in range(n):
            self.nodes[last_key + i + 1] = LIFNeuron(I_spike=I_spike, noise=noise)
            self.outlinks[last_key + i + 1] = []
            self.inlinks[last_key + i + 1] = []
        
        if n > 0:
            self.max_degree_nodes = set() # No node is at max degree anymore
    

    # Adds m arcs to the graph randomly
    def add_links(self, m):
        node_ids = set(list(self.nodes.keys()))
        n = len(node_ids)
        
        for i in range(m):
            possible_sources = node_ids.difference(self.max_degree_nodes)
            source = list(possible_sources)[random.randint(0, len(possible_sources) - 1)]

            possible_dests = node_ids.difference(set(self.outlinks[source]))#.difference(set(self.inlinks[source]))
            possible_dests.discard(source) # A node can't be linked to itself
            dest = list(possible_dests)[random.randint(0, len(possible_dests) - 1)]

            self.outlinks[source].append(dest)
            self.inlinks[dest].append(source)
            if len(self.outlinks[source]) == n - 1:
                self.max_degree_nodes.add(source)

    # removes all nodes who has no incoming or outgoing links
    def remove_unlinked_nodes(self):
        for node_id in list(self.nodes.keys()):
            if len(self.outlinks[node_id]) == 0 or len(self.inlinks[node_id]) == 0:
                self.remove_node(node_id)

    def remove_node(self, node_id):
        for outnode in self.outlinks[node_id]:
            self.inlinks[outnode].remove(node_id)
        for innode in self.inlinks[node_id]:
            self.outlinks[innode].remove(node_id)

        del self.inlinks[node_id]
        del self.outlinks[node_id]
        del self.nodes[node_id]


class Experiment:
    """
        * n : number of neurons
        * d : density of the network, between 0 and 1
        * inhib_rate : percentage of inhibitory neurons, between 0 and 1
        * I_excitation : electrical power of excitatory spikes
        * I_inhibition : electrical power of inhibitory spikes
    """
    def __init__(self, n, d, inhib_rate, I_excitation, I_inhibition, noise=0):
        self.network = Network(n, d, inhib_rate, I_excitation, I_inhibition, noise)

    """
        * T : total duration of the experiment
        * dt : update interval
    """
    def run(self, T, dt):
        n = len(self.network.nodes.keys())
        spikes = np.zeros((int(T / dt), n))
        mem_potentials = np.zeros((int(T / dt), n))
        spike_timestamps = [[] for _ in range (n)]

        for t in range(1, int(T / dt)):
            # Update neuron external inputs (spikes from t-1)
            neuron_inputs = np.zeros(n)
            for neuron_pos, fired in enumerate(spikes[t - 1]):
                if fired:
                    neuron_id = list(self.network.nodes.keys())[neuron_pos]
                    for linked_neuron_id in self.network.outlinks[neuron_id]:
                        linked_neuron_pos = list(self.network.nodes.keys()).index(linked_neuron_id)
                        
                        neuron_inputs[linked_neuron_pos] += self.network.nodes[neuron_id].I_spike

            # Update neuron membrane potentials at time t
            for neuron_pos, neuron_id in enumerate(self.network.nodes.keys()):
                neuron = self.network.nodes[neuron_id]
                fired = neuron.update(dt, neuron_inputs[neuron_pos])
                if fired:
                    spikes[t, neuron_pos] = 1
                    spike_timestamps[neuron_pos].append(t)
                mem_potentials[t, neuron_pos] = neuron.v
        
        return spikes, mem_potentials, spike_timestamps
    