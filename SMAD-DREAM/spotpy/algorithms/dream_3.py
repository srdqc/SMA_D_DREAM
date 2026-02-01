import random

from fontTools.misc.bezierTools import epsilon

import time

import numpy as np
import pandas as pd
from scipy import stats

from . import _algorithm
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), :]
        return x


class GroundwaterTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=1024, nhead=64, dim_feedforward=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_layers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, num_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)
        query = self.query.repeat(src.size(0), 1, 1)
        query = self.pos_encoder(query)
        output = self.decoder(query, memory)
        output = self.output_layer(output.squeeze(1))
        return output


class dream_3(_algorithm):

    def __init__(self, *args, **kwargs):

        kwargs["optimization_direction"] = "maximize"
        kwargs["algorithm_name"] = "DiffeRential Evolution Adaptive Metropolis (DREAM) algorithm"
        super(dream_3, self).__init__(*args, **kwargs)

    def check_par_validity_bound(self, par):
        if len(par) != len(self.min_bound) or len(par) != len(self.max_bound):
            raise ValueError("Bounds have not the same lengths as Parameter array")

        for i in range(len(par)):
            if par[i] < self.min_bound[i]:
                par[i] = self.min_bound[i]
            elif par[i] > self.max_bound[i]:
                par[i] = self.max_bound[i]

        return par

    def get_regular_startingpoint(self, nChains):
        randompar = np.array([self.parameter()["random"] for _ in range(10001)]).T
        initial_points = [np.percentile(randompar, (j + 1) / float(nChains + 1) * 100, axis=1) for j in range(nChains)]
        initial_points = np.array(initial_points)
        np.apply_along_axis(np.random.shuffle, 0, initial_points)
        return initial_points

    def check_par_validity_reflect(self, par):
        if len(par) != len(self.min_bound) or len(par) != len(self.max_bound):
            raise ValueError("Bounds have not the same lengths as Parameter array")

        for i in range(len(par)):
            if par[i] < self.min_bound[i]:
                par[i] = self.min_bound[i] + (self.min_bound[i] - par[i])  
                par[i] = self.max_bound[i] - (par[i] - self.max_bound[i])  

        par = np.clip(par, self.min_bound, self.max_bound)

        return par

    def _get_gamma(self, newN, nchain_pairs, cur_hat):
        thresholds = [1.5, 1.2, 1.0]
        probabilities = [0.5, 0.2, 0]

        d_star = np.sum(newN)

        for threshold, probability in zip(thresholds, probabilities):
            if cur_hat > threshold and np.random.uniform(low=0, high=1) >= probability and d_star > 0:
                return 2.38 / np.sqrt(2 * nchain_pairs * d_star)
        return 1

    def _get_lambda_(self, cur_hat):
        if cur_hat > 1.5:
            return random.uniform(0.5, 1)
        elif cur_hat > 1.2:

            source_min, source_max = 1.2, 1.5
            target_min, target_max = 0.1, 0.5
            mapped_value = target_min + (cur_hat - source_min) * (target_max - target_min) / (source_max - source_min)
            return mapped_value
        elif cur_hat > 1.0:
            return random.uniform(0, 0.1)
        else:
            return 0  

    def _get_epsilon_(self, cur_hat):
        if cur_hat <= 1.2:
            return 10e-6
        else:
            return 10e-6 + (1 - 10e-6) * (1 - np.exp(-5 * (cur_hat - 1.2)))

    def _get_w(self, cur_hat):
        if cur_hat > 1.5:
            return 0.05
        elif cur_hat > 1.2:
            return 0.1
        elif cur_hat > 1.0:
            return 0.5

    def get_other_random_chains(self, cur_chain, nchain_pairs):
        selectable_chains = list(range(self.num_chains))
        selectable_chains.remove(cur_chain)

        chain_pairs = []
        selectable_chains = list(range(self.num_chains))
        selectable_chains.remove(cur_chain)

        for _ in range(nchain_pairs):
            pair = random.sample(selectable_chains, 2)
            chain_pairs.append(pair)
            selectable_chains = [chain for chain in selectable_chains if chain not in pair]

        return chain_pairs

    def get_new_proposal_vector(self, cur_chain, newN, c, cur_r_hat):
        nchain_pairs = random.randint(1, self.delta)

        gamma = self._get_gamma(newN, nchain_pairs, cur_r_hat)
        lambda_ = self._get_lambda_(cur_r_hat)
        epsilon_ = self._get_epsilon_(cur_r_hat)
        w = self._get_w(cur_r_hat)
        chain_pairs = self.get_other_random_chains(cur_chain, nchain_pairs)
        cur_par_set = self.best_params[cur_chain][self.chain_run_counts[cur_chain] - 1]
        random_par_set1 = np.zeros(self.N)
        random_par_set2 = np.zeros(self.N)
        for pair in chain_pairs:
            random_par_set1 += self.best_params[pair[0]][self.chain_run_counts[pair[0]] - 1]
            random_par_set2 += self.best_params[pair[1]][self.chain_run_counts[pair[1]] - 1]

        guided_term = w * (self.best_dict.get("parameter", None) - cur_par_set)
        new_parameterset = []
        for i in range(self.N): 
            if newN[i] == True:
                e = np.random.uniform(-lambda_, lambda_)
                epsilon = np.random.normal(0, epsilon_)
                new_parameterset.append(cur_par_set[i] + (1 - w) * (1.0 + e) * gamma * np.array(random_par_set1[i] - random_par_set2[i]) + gamma * guided_term[i] + epsilon)
            else:
                new_parameterset.append(cur_par_set[i])

        return self.check_par_validity_reflect(new_parameterset)

    def update_mcmc_status(self, par, like, sim, cur_chain):
        self.best_params[cur_chain][self.chain_run_counts[cur_chain]] = list(par)
        self.best_like[cur_chain] = like
        self.best_simulation[cur_chain] = list(sim)

    def _rhat(self, parameter_array):
        array = np.asarray(parameter_array, dtype=float)
        _, num_samples = array.shape
        chain_mean = np.mean(array, axis=1)
        chain_var = np.var(array, axis=1, ddof=1)
        between_chain_variance = num_samples * np.nanvar(chain_mean, axis=0, ddof=1)
        within_chain_variance = np.mean(chain_var)
        if within_chain_variance == 0:
            within_chain_variance = 1e-10
        rhat_value = np.sqrt((between_chain_variance / within_chain_variance + num_samples - 1) / num_samples)

        return rhat_value

    def _backtransform_ranks(self, arr, c=3 / 8):
        arr = np.asarray(arr)
        size = arr.size
        return (arr - c) / (size - 2 * c + 1)

    def _z_scale(self, ary):
        ary = np.asarray(ary)
        rank = stats.rankdata(ary, method="average")
        rank = self._backtransform_ranks(rank)
        z = stats.norm.ppf(rank)
        return z.reshape(ary.shape)

    def get_r_hat(self, parameter_array): 
        n, N, d = parameter_array.shape

        non_nan_mask = ~np.isnan(parameter_array)
        valid_count = np.sum(non_nan_mask[0, :, 0])

        if valid_count > 3:
            parameter_array = parameter_array[:, valid_count // 2 : valid_count, :]
        else:
            parameter_array = parameter_array[:, :valid_count, :]

        R_stat = np.zeros(d)

        if n > 3:
            for i in range(d):
                parameter_data = parameter_array[:, :, i] 

                rhat_bulk = self._rhat(self._z_scale(parameter_data))

                split_ary_folded = np.abs(parameter_data - np.median(parameter_data, axis=1, keepdims=True))
                rhat_tail = self._rhat(self._z_scale(split_ary_folded))

                R_stat[i] = max(rhat_bulk, rhat_tail)

        return R_stat

    def mean_log_density(self, like_array):
        n_samples = self.chain_run_counts[0]  
        half_samples = n_samples // 2

        half_chain_samples = like_array[half_samples:n_samples, :]
        min_value = np.min(half_chain_samples, axis=0)  
        half_chain_samples = np.where(min_value < 0, half_chain_samples - min_value + 1, half_chain_samples)
        log_densities = np.log(half_chain_samples)
        mean_log_densities = np.mean(log_densities, axis=0)  

        return mean_log_densities

    def detect_outliers_iqr(self, data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        outliers = np.where(data < lower_bound)[0]
        return outliers

    def check(self, like_array):
        mean_log_densities = self.mean_log_density(like_array)
        outliers = self.detect_outliers_iqr(mean_log_densities)
        if len(outliers) > 0:
            for chain in outliers:
                self.best_params[chain][self.chain_run_counts[chain] - 1] = self.best_dict.get("parameter", None)
                self.best_like[chain] = self.best_dict.get("like", None)
                self.best_simulation[chain] = self.best_dict.get("simulation", None)

    def calculate_metropolis_hastings_ratio(self, like, best_like, nrN, option, cur_r_hat):
        if option == 1:
            logMetropHastRatio = like / best_like
        elif option in [2, 4]:
            logMetropHastRatio = np.exp(like - best_like)
        elif option == 3:
            logMetropHastRatio = (like / best_like) ** (-nrN * (1 + self._get_gamma(nrN)) / 2)
        elif option == 5:
            sigma = np.mean(np.array(self.evaluation) * 0.1)
            logMetropHastRatio = np.exp(-0.5 * (-like + best_like) / (sigma**2))
        elif option == 6:
            logMetropHastRatio = np.exp(-0.5 * (-like + best_like))
        else:
            logMetropHastRatio = 0

        return logMetropHastRatio

    def train_and_evaluate(self, model, train_loader, val_loader, criterion, optimizer, device, scaler_result, epochs=100):

        val_r2_scores = []

        best_val_r2 = float("-inf")
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_predictions = []
            train_actuals = []
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_actuals.extend(targets.cpu().numpy())

            train_predictions = scaler_result.inverse_transform(train_predictions)
            train_actuals = scaler_result.inverse_transform(train_actuals)

            model.eval()
            val_predictions = []
            val_actuals = []
            total_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    total_val_loss += val_loss.item()
                    val_predictions.extend(outputs.cpu().numpy())
                    val_actuals.extend(targets.cpu().numpy())

            val_predictions = scaler_result.inverse_transform(val_predictions)
            val_actuals = scaler_result.inverse_transform(val_actuals)

            val_r2 = r2_score(val_actuals, val_predictions)
            val_r2_scores.append(val_r2)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = model.state_dict()

        print(f"validation R²: {best_val_r2:.4f} ")

        model.load_state_dict(best_model_state)

        if best_val_r2 < 0.97:
            model = None

        return model

    def calculate_first_column_std(self, data):
        first_column = [row[0] for row in data]
        std_dev = np.std(first_column)
        return std_dev

    def load_and_preprocess_data(self, data):
        df = pd.DataFrame(data)
        df = df[(df.iloc[:, 0] != -10000000000.0) & (df.iloc[:, 0] != -1.4308224e17) & (df.iloc[:, 0] > -9999999900)]

        data = df.to_numpy()
        data = np.array(data)
        mask = (data[:, 0] != -10000000000.0) & (data[:, 0] != -1.4308224e17) & (data[:, 0] > -9999999900)
        data = data[mask]

        if data.shape[0] > 5000:
            print("Data has more than 3000 rows.")
        inputs = data[:, 1 : self.num_parameters + 1]
        targets = data[:, self.num_parameters + 1 :]

        param_train, param_val, result_train, result_val = train_test_split(inputs, targets, test_size=0.4, random_state=42)

        scaler_param = StandardScaler()
        param_train_scaled = scaler_param.fit_transform(param_train)
        param_val_scaled = scaler_param.transform(param_val)

        scaler_result = StandardScaler()
        result_train_scaled = scaler_result.fit_transform(result_train)
        result_val_scaled = scaler_result.transform(result_val)

        train_inputs = torch.FloatTensor(param_train_scaled)
        train_targets = torch.FloatTensor(result_train_scaled)
        val_inputs = torch.FloatTensor(param_val_scaled)
        val_targets = torch.FloatTensor(result_val_scaled)

        print(train_inputs.shape[0])
        print(train_targets.shape[0])
        print(val_inputs.shape[0])
        print(val_targets.shape[0])

        return train_inputs, train_targets, val_inputs, val_targets, scaler_param, scaler_result

    def rmse_loss(self, y_pred, y_true):
        return torch.sqrt(nn.MSELoss()(y_pred, y_true))

    def get_data_within_three_std_dev(self, data):
        first_column = [row[0] for row in data]
        mean = np.mean(first_column)
        std_dev = np.std(first_column)
        lower_bound = mean - 3 * std_dev
        upper_bound = mean + 3 * std_dev

        data_within_range = [row for row in data if lower_bound <= row[0] <= upper_bound]
        return data_within_range

    def get_data_within_std_dev(self, data):
        first_column = [row[0] for row in data]
        mean = np.mean(first_column)
        std_dev = np.std(first_column)
        lower_bound = mean - std_dev
        upper_bound = mean + std_dev

        data_within_range = [row for row in data if lower_bound <= row[0] <= upper_bound]
        return data_within_range

    def get_SurrogateModel(self, cur_r_hat):
        """
        model, train_loader, val_loader, criterion, optimizer, device, scaler_result
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = self.record

        df = pd.DataFrame(data)
        df = df[(df.iloc[:, 0] != -10000000000.0) & (df.iloc[:, 0] != -1.4308224e17) & (df.iloc[:, 0] > -9999999964.475117)]
        data = df.to_numpy()
        data = np.array(data)
        mask = (data[:, 0] != -10000000000.0) & (data[:, 0] != -1.4308224e17) & (data[:, 0] > -9999999900)
        data = data[mask]

        d_model = 512
        nhead = 8
        dim_feedforward = 1024

        num_layers = 1

        if self.SurrogateModel == None:

            train_inputs, train_targets, val_inputs, val_targets, scaler_param, scaler_result = self.load_and_preprocess_data(data)
            input_dim = train_inputs.shape[1]
            output_dim = train_targets.shape[1]
            train_dataset = TensorDataset(train_inputs.unsqueeze(1), train_targets)
            train_loader = DataLoader(train_dataset, 64, shuffle=True)
            val_dataset = TensorDataset(val_inputs.unsqueeze(1), val_targets)
            val_loader = DataLoader(val_dataset, 64)

            model = GroundwaterTransformer(input_dim, output_dim, d_model, nhead, dim_feedforward, num_layers).to(device)

            criterion = self.rmse_loss
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            train_model = self.train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, scaler_result)
            self.std_dev = self.calculate_first_column_std(data)
            return train_model, scaler_param, scaler_result
        else:
            std_dev = self.calculate_first_column_std(data)
            deviation = abs(std_dev - self.std_dev) / self.std_dev
            if deviation <= 0.1:
                return self.SurrogateModel, self.scaler_param, self.scaler_result
            else:
                if cur_r_hat < 1.2:
                    data = self.get_data_within_std_dev(data)
                else:
                    data = self.get_data_within_three_std_dev(data)
                train_inputs, train_targets, val_inputs, val_targets, scaler_param, scaler_result = self.load_and_preprocess_data(data)
                input_dim = train_inputs.shape[1]
                output_dim = train_targets.shape[1]
                train_dataset = TensorDataset(train_inputs.unsqueeze(1), train_targets)
                train_loader = DataLoader(train_dataset, 64, shuffle=True)
                val_dataset = TensorDataset(val_inputs.unsqueeze(1), val_targets)
                val_loader = DataLoader(val_dataset, 64)

                model = GroundwaterTransformer(input_dim, output_dim, d_model, nhead, dim_feedforward, num_layers).to(device)

                criterion = self.rmse_loss
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                train_model = self.train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, scaler_result, epochs=100)
                self.std_dev = self.calculate_first_column_std()
                return train_model, scaler_param, scaler_result

    def sample(self, repetitions, n_chains=7, n_cr=3, delta=3, c=0.1, eps=10e-6, convergence_limit=1.2, runs_after_convergence=100, acceptance_test_option=6):
        self.set_repetiton(repetitions)

        print("Starting the DREAM algorithm with " + str(repetitions) + " repetitions...")
        if n_chains < 2 * delta + 1:
            print("Please use at least n=2*delta+1 chains!")
            return None

        self.num_repetitions = int(repetitions)
        self.num_chains = int(n_chains)
        self.delta = int(delta)

        self.SurrogateModel = None
        self.scaler_param = None
        self.scaler_result = None
        self.record = []
        self.std_dev = None

        self.burn_in_period = self.num_chains
        self.num_parameters = len(self.parameter()["step"])
        interval_time = time.time()

        self.best_params = np.full((self.num_chains, self.num_repetitions, self.num_parameters), np.nan)
        self.like_array = np.full((self.num_repetitions, self.num_chains), np.nan)
        self.best_like = [[-np.inf]] * self.num_chains
        self.best_simulation = [[np.nan]] * self.num_chains
        self.accepted_array = np.zeros(self.num_chains)
        self.chain_run_counts = np.zeros(self.num_chains, dtype=int)
        self.min_bound, self.max_bound = self.parameter()["minbound"], self.parameter()["maxbound"]

        print("Initialize ", self.num_chains, " chain(s)...")
        self.iter = 0
        initial_points = self.get_regular_startingpoint(n_chains)
        param_generator = ((curChain, list(initial_points[curChain])) for curChain in range(int(self.num_chains)))

        for cur_chain, par, sim in self.repeat(param_generator):
            like = self.postprocessing(self.iter, par, sim, chains=cur_chain)
            self.update_mcmc_status(par, like, sim, cur_chain)
            self.iter += 1
            self.chain_run_counts[cur_chain] += 1
            combined_array = np.concatenate(([like], par, sim))
            self.record.append(combined_array)

        for i in range(self.num_chains):
            self.like_array[self.chain_run_counts[i] - 1][i] = self.best_like[i]


        best_chain = np.argmax(self.best_like) 
        par = self.best_params[best_chain][self.chain_run_counts[best_chain] - 1]
        like = self.best_like[best_chain]
        sim = self.best_simulation[best_chain]
        self.best_dict = {"parameter": par, "simulation": sim, "like": like} 

        print("Beginn of Random Walk")
        convergence = False
        self.r_hats = []
        self.eps = eps

        self.CR = [(i + 1) / n_cr for i in range(n_cr)]

        self.N = len(self.parameter()["random"])

        newN = [True] * self.N

        r_hat = np.full(self.num_parameters, np.inf)  

        while self.iter < self.num_repetitions:
            cur_r_hat = np.max(r_hat)
            param_generator = ((cur_chain, self.get_new_proposal_vector(cur_chain, newN, c, cur_r_hat)) for cur_chain in range(self.num_chains))
            print(f"cur_r_hat:{cur_r_hat}")
            print(f"self.iter > 0.15 * self.num_repetitions:{self.iter > 0.15 * self.num_repetitions}")
            if cur_r_hat < 2.0 and self.iter > 0.15 * self.num_repetitions:
                self.SurrogateModel, self.scaler_param, self.scaler_result = self.get_SurrogateModel(cur_r_hat)

            for c_chain, par, sim in self.repeat(param_generator, self.SurrogateModel, self.scaler_param, self.scaler_result):
                p_cr = np.random.randint(0, n_cr)
                ids = np.random.uniform(low=0, high=1, size=self.N)
                newN = ids < self.CR[p_cr]
                nrN = np.sum(newN)

                if nrN == 0:
                    ids = [np.random.randint(0, self.N)]
                    nrN = 1

                like = self.postprocessing(self.iter, par, sim, chains=c_chain)

                log_metrop_hast_ratio = self.calculate_metropolis_hastings_ratio(like, self.best_like[c_chain], nrN, acceptance_test_option, cur_r_hat)

                u = np.random.uniform(low=0.0, high=1)
                if log_metrop_hast_ratio > u:
                    self.update_mcmc_status(par, like, sim, c_chain)
                    self.accepted_array[c_chain] += 1

                else:
                    self.update_mcmc_status(self.best_params[c_chain][self.chain_run_counts[c_chain] - 1], self.best_like[c_chain], self.best_simulation[c_chain], c_chain)

                if self.status.stop:
                    self.iter = self.num_repetitions
                    print("Stopping sampling")
                    break

                self.iter += 1
                self.chain_run_counts[c_chain] += 1

                combined_array = np.concatenate(([like], par, sim))
                self.record.append(combined_array)

            for i in range(self.num_chains):
                self.like_array[self.chain_run_counts[i] - 1][i] = self.best_like[i]

            best_chain = np.argmax(self.best_like)  
            par = self.best_params[best_chain][self.chain_run_counts[best_chain] - 1]
            like = self.best_like[best_chain]
            sim = self.best_simulation[best_chain]

            best_dict_like = self.best_dict.get("like", None)

            if best_dict_like is None or like > best_dict_like:
                self.best_dict = {"parameter": par, "simulation": sim, "like": like}

            self.check(self.like_array)

            r_hat = self.get_r_hat(self.best_params)
            self.r_hats.append(r_hat)
            acttime = time.time()

            if acttime - interval_time >= 2 and self.iter >= 2 and self.chain_run_counts[-1] >= 3:
                convergence_rate = np.around(r_hat, decimals=4)
                print(f"Convergence rates = {convergence_rate}")
                interval_time = time.time()

            if (np.array(r_hat) < convergence_limit).all() and not convergence and self.chain_run_counts[-1] >= 5:
                print("#############")
                print(f"Convergence has been achieved after {self.iter} of {self.num_repetitions} runs! Finally, {runs_after_convergence} runs will be additionally sampled to form the posterior distribution")
                print("#############")
                self.num_repetitions = self.iter
                convergence = True
        self.final_call()
        return self.r_hats
