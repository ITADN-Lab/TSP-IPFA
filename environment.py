import copy
import numpy as np
import torch
import pandapower as pp
from config import ENV_CONFIG
from models import ImprovedDNNModel


class ImprovedPhaseManager:
    def __init__(self):
        self.phase = 1
        self.transition_progress = 0.0

    def update(self, convergence):
        if self.phase == 1 and convergence > 0.7:
            self.transition_progress = min(1.0, self.transition_progress + 0.02)
            if self.transition_progress >= 1.0:
                self.phase = 2
                self.transition_progress = 0.0
        elif self.phase == 2 and convergence < 0.6:
            self.transition_progress = min(1.0, self.transition_progress + 0.05)
            if self.transition_progress >= 1.0:
                self.phase = 1
                self.transition_progress = 0.0
        return self.phase, self.transition_progress


class PowerFlowEnvironment:
    def __init__(self, num_generators, num_loads, load_converge_model_path, dataset, fixed_sample_idx=0):
        self.num_generators = num_generators
        self.num_loads = num_loads
        self.state_dim = (num_generators * 2) + (num_loads * 2)
        self.dataset = dataset

        self.data_mean = dataset.mean
        self.data_std = dataset.std
        self.phase_manager = ImprovedPhaseManager()
        self.episode_steps = 0
        self.min_sustain_steps = 100
        self.sustain_counter = 0
        self.constraint_satisfied = False

        self.converge_model = ImprovedDNNModel(input_dim=self.state_dim, num_buses=30)
        self.converge_model.load_state_dict(torch.load(load_converge_model_path, weights_only=True))
        self.converge_model.eval()

        self._initialize_network()
        self.net_original = copy.deepcopy(self.net)

        self.constraints = ENV_CONFIG

    def _initialize_network(self):
        self.net = pp.networks.case30()
        buss_gen = self.net.gen['bus'].values
        for bus in buss_gen:
            idx = pp.create_sgen(self.net, bus=bus, p_mw=2, q_mvar=0.5, name=bus)

        self.net.sgen['max_p_mw'] = self.net.gen.max_p_mw.values
        self.net.sgen['min_p_mw'] = self.net.gen.min_p_mw.values
        self.net.sgen['max_q_mvar'] = self.net.gen.max_q_mvar.values
        self.net.sgen['min_q_mvar'] = self.net.gen.min_q_mvar.values
        self.net.gen = self.net.gen.drop(self.net.gen.index)

    def run_power_flow(self):
        try:
            pp.runpp(self.net_original, numba=True)

            voltage_violations = ((self.net_original.res_bus.vm_pu < 0.95) | (self.net_original.res_bus.vm_pu > 1.05))
            voltage_violation_count = voltage_violations.sum()
            max_voltage_deviation = max(
                abs(self.net_original.res_bus.vm_pu - 1.05).max(),
                abs(self.net_original.res_bus.vm_pu - 0.95).max()
            )

            line_loading = self.net_original.res_line.loading_percent
            thermal_violations = line_loading > 100
            thermal_violation_count = thermal_violations.sum()
            max_thermal_deviation = max(0, line_loading.max() - 100) if thermal_violation_count > 0 else 0

            return {
                'converged': self.net_original.converged,
                'voltage_violation_count': voltage_violation_count,
                'thermal_violation_count': thermal_violation_count,
                'max_voltage_deviation': max_voltage_deviation,
                'max_thermal_deviation': max_thermal_deviation
            }
        except:
            return {
                'converged': False,
                'voltage_violation_count': 0,
                'thermal_violation_count': 0,
                'max_voltage_deviation': 0,
                'max_thermal_deviation': 0
            }

    def get_state(self):
        sgen_p_mw = self.net.sgen['p_mw']
        sgen_q_mvar = self.net.sgen['q_mvar']
        load_p_mw = self.net.load['p_mw']
        load_q_mvar = self.net.load['q_mvar']

        sgen_combined = np.empty((len(sgen_p_mw) * 2,), dtype=sgen_p_mw.dtype)
        sgen_combined[::2] = sgen_p_mw
        sgen_combined[1::2] = sgen_q_mvar

        load_combined = np.empty((len(load_p_mw) * 2,), dtype=load_p_mw.dtype)
        load_combined[::2] = load_p_mw
        load_combined[1::2] = load_q_mvar

        state = np.concatenate([sgen_combined, load_combined])
        return state.tolist()

    def apply_action(self, action):
        d_sgen_p_mw = action[0::2]
        d_sgen_q_mvar = action[1::2]

        self.net.sgen['p_mw'] += d_sgen_p_mw
        self.net.sgen['q_mvar'] += d_sgen_q_mvar

        action_reward = 0
        for i in range(len(self.net.sgen)):
            if self.net.sgen.loc[i, 'p_mw'] > self.net.sgen.loc[i, 'max_p_mw']:
                action_reward -= self.net.sgen.loc[i, 'p_mw'] - self.net.sgen.loc[i, 'max_p_mw']
                self.net.sgen.loc[i, 'p_mw'] = self.net.sgen.loc[i, 'max_p_mw']
            elif self.net.sgen.loc[i, 'p_mw'] < self.net.sgen.loc[i, 'min_p_mw']:
                action_reward -= self.net.sgen.loc[i, 'min_p_mw'] - self.net.sgen.loc[i, 'p_mw']
                self.net.sgen.loc[i, 'p_mw'] = self.net.sgen.loc[i, 'min_p_mw']

            if self.net.sgen.loc[i, 'q_mvar'] > self.net.sgen.loc[i, 'max_q_mvar']:
                action_reward -= self.net.sgen.loc[i, 'q_mvar'] - self.net.sgen.loc[i, 'max_q_mvar']
                self.net.sgen.loc[i, 'q_mvar'] = self.net.sgen.loc[i, 'max_q_mvar']
            elif self.net.sgen.loc[i, 'q_mvar'] < self.net.sgen.loc[i, 'min_q_mvar']:
                action_reward -= self.net.sgen.loc[i, 'min_q_mvar'] - self.net.sgen.loc[i, 'q_mvar']
                self.net.sgen.loc[i, 'q_mvar'] = self.net.sgen.loc[i, 'min_q_mvar']

        return action_reward

    def reset(self):
        state = self.dataset.ret_numpy(np.random.choice(len(self.dataset)))

        sgen_p_q = state[:self.num_generators * 2]
        load_p_q = state[self.num_generators * 2:]

        sgen_p_mw = sgen_p_q[0::2]
        sgen_q_mvar = sgen_p_q[1::2]
        load_p_mw = load_p_q[0::2]
        load_q_mvar = load_p_q[1::2]

        self._initialize_network()
        self.net.sgen['p_mw'] = sgen_p_mw
        self.net.sgen['q_mvar'] = sgen_q_mvar
        self.net.load['p_mw'] = load_p_mw
        self.net.load['q_mvar'] = load_q_mvar

        self.episode_steps = 0
        self.sustain_counter = 0
        self.constraint_satisfied = False
        self.net_original = copy.deepcopy(self.net)

        return self.get_state()

    def step(self, action):
        action_reward = self.apply_action(action * 0.1)
        done = False
        self.episode_steps += 1

        with torch.no_grad():
            convergence_logits, _, _ = self.converge_model(
                torch.FloatTensor(self.get_state()).unsqueeze(0)
            )
            convergence = torch.sigmoid(convergence_logits)

        current_phase, progress = self.phase_manager.update(convergence)

        original_state = self.get_state() * (self.data_std + 1e-8) + self.data_mean
        sgen_p_q = original_state[:self.num_generators * 2]
        load_p_q = original_state[self.num_generators * 2:]

        self.net_original.sgen['p_mw'] = sgen_p_q[0::2]
        self.net_original.sgen['q_mvar'] = sgen_p_q[1::2]
        self.net_original.load['p_mw'] = load_p_q[0::2]
        self.net_original.load['q_mvar'] = load_p_q[1::2]

        physical_result = self.run_power_flow()

        if current_phase == 1 and progress == 0:
            reward = self._phase1_reward(convergence, action_reward)
        elif current_phase == 1 and progress > 0:
            reward = (1 - progress) * self._phase1_reward(convergence,
                                                          action_reward) + progress * self._phase2_reward_by_physics(
                physical_result, action_reward)
        elif current_phase == 2 and progress > 0:
            reward = (1 - progress) * self._phase2_reward_by_physics(physical_result,
                                                                     action_reward) + progress * self._phase1_reward(
                convergence, action_reward)
        else:
            reward = self._phase2_reward_by_physics(physical_result, action_reward)

            if self._check_all_constraints(physical_result):
                self.constraint_satisfied = True
                reward += 10

        if self.constraint_satisfied:
            sustain_reward = self._sustain_reward(physical_result)
            reward += sustain_reward

            if self.sustain_counter >= self.min_sustain_steps:
                done = True

        return self.get_state(), reward, done

    def _phase1_reward(self, convergence, action_reward):
        base_reward = 20 * np.tanh(5 * (convergence - 0.7))
        bonus = 10 * (convergence - 0.9) / 0.1 if convergence > 0.9 else 0
        action_penalty = action_reward * 0.5
        return base_reward + bonus + action_penalty

    def _phase2_reward_by_physics(self, physical_result, action_reward):
        if not physical_result['converged']:
            return -5

        result = physical_result
        base_reward = 30

        voltage_penalty = 0
        if result['voltage_violation_count'] > 0:
            voltage_penalty = - (result['voltage_violation_count'] * 0.5) - np.log1p(
                result['max_voltage_deviation'] * 100)

        thermal_penalty = 0
        if result['thermal_violation_count'] > 0:
            thermal_penalty = - (result['thermal_violation_count'] * 0.5) - np.log1p(
                result['max_thermal_deviation'] * 1)

        return base_reward + voltage_penalty + thermal_penalty + action_reward * 0.5

    def _sustain_reward(self, physical_result):
        reward = 5.0
        voltage_deviation = physical_result['max_voltage_deviation']
        thermal_deviation = physical_result['max_thermal_deviation']

        if voltage_deviation < 0.01:
            reward += 3.0
        elif voltage_deviation < 0.03:
            reward += 1.5

        if thermal_deviation < 5.0:
            reward += 2.0
        elif thermal_deviation < 10.0:
            reward += 1.0

        return reward

    def _check_all_constraints(self, physical_result):
        if physical_result is None:
            physical_result = self.run_power_flow()

        satisfied = (physical_result['converged'] and
                     physical_result['voltage_violation_count'] == 0 and
                     physical_result['thermal_violation_count'] == 0)

        if satisfied:
            self.sustain_counter += 1
        else:
            self.sustain_counter = 0

        return satisfied