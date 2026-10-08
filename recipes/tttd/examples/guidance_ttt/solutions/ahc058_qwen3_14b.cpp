#include <bits/stdc++.h>
using namespace std;

int N, L, T, K;
vector<long long> A;
vector<vector<long long>> C;
array<array<double, 10>, 4> g_weight_log;

array<array<double, 10>, 4> g_level_priority;
bool g_use_lp = false;

array<double, 4> g_phase_weight = {1.0, 1.0, 1.0, 1.0};
bool g_use_cascade = false;

double g_discount_rate = 0.05;

array<double, 4> g_synergy_level_weight = {1.0, 1.0, 1.0, 1.0};
double g_synergy_compound_weight = 1.0;
double g_synergy_balance_weight = 1.0;
double g_diminishing_factor = 1.0;

// Dual-sigmoid synergy blend: 0 = single-level priority, 1 = multi-level synergy
double g_synergy_blend = 0.0;

// Dual-sigmoid blend with real-time state adaptation
void set_synergy_blend_dual(int turn, int total, const struct State& s);

// Legacy single-sigmoid (kept as fallback, should not be called in normal flow)
void set_synergy_blend(int turn, int total) {
    double progress = (double)turn / (double)total;
    double midpoint = 0.4;
    double k = 8.0;
    g_synergy_blend = 1.0 / (1.0 + exp(-k * (progress - midpoint)));
}

struct State;

void set_weights_uniform() {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 10; j++)
            g_weight_log[i][j] = 0.0;
}

void set_weights(int mode) {
    if (mode == 0) { set_weights_uniform(); return; }
    vector<vector<double>> W(L, vector<double>(N));
    for (int j = 0; j < N; j++) {
        W[0][j] = (double)A[j];
        for (int i = 1; i < L; i++) {
            if (mode == 1) W[i][j] = (double)A[j] / (double)C[i][j];
            else if (mode == 2) W[i][j] = 1.0 / (double)C[i][j];
            else W[i][j] = (double)A[j] * (double)A[j] / (double)C[i][j];
        }
    }
    for (int i = 0; i < L; i++) {
        double log_sum = 0;
        for (int j = 0; j < N; j++) log_sum += log(max(W[i][j], 1e-18));
        double log_mean = log_sum / N;
        double norm = exp(log_mean);
        for (int j = 0; j < N; j++) W[i][j] /= max(norm, 1e-18);
    }
    for (int i = 0; i < L; i++)
        for (int j = 0; j < 10; j++)
            g_weight_log[i][j] = log2(max(W[i][j], 1e-18));
}

void set_discount_rate_state(const State& s, int turn, int total);

void set_phase_weights(int turn, int total) {
    double progress = (double)turn / (double)total;
    if (progress < 0.12) {
        g_phase_weight = {0.5, 2.0, 1.5, 1.0};
    } else if (progress < 0.35) {
        g_phase_weight = {0.8, 1.5, 2.0, 1.5};
    } else if (progress < 0.65) {
        g_phase_weight = {1.2, 1.0, 1.2, 1.0};
    } else {
        g_phase_weight = {1.8, 1.0, 0.6, 0.4};
    }
}

struct State {
    array<array<double, 10>, 4> B;
    array<array<long long, 10>, 4> P;
    double apples;
    void init() {
        apples = (double)K;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 10; j++) { B[i][j] = 1.0; P[i][j] = 0; }
    }
    inline double production_rate() const {
        double rate = 0;
        for (int j = 0; j < N; j++)
            rate += (double)A[j] * B[0][j] * (double)P[0][j];
        return rate;
    }
};

// Dual-sigmoid blend with real-time state adaptation
void set_synergy_blend_dual(int turn, int total, const State& s) {
    double progress = (double)turn / (double)total;
    
    // First sigmoid: early phase transition around 15%
    double k1 = 12.0;
    double mid1 = 0.15;
    double sig1 = 1.0 / (1.0 + exp(-k1 * (progress - mid1)));
    
    // Second sigmoid: late phase transition around 65%
    double k2 = 12.0;
    double mid2 = 0.65;
    double sig2 = 1.0 / (1.0 + exp(-k2 * (progress - mid2)));
    
    if (progress < 0.30) {
        // Early "population growth" phase: favor single-level efficiency
        g_synergy_blend = 0.05 + 0.10 * sig1;
    } else if (progress > 0.60) {
        // Late "cascading synergy" phase: emphasize multi-level interactions
        g_synergy_blend = 0.30 + 0.70 * sig2;
    } else {
        // Transition phase (30-60%): dynamically adjust based on B ratios across levels
        double b_low = 0, b_high = 0;
        for (int j = 0; j < N; j++) {
            b_low += s.B[0][j] + s.B[1][j];
            b_high += s.B[2][j] + s.B[3][j];
        }
        double b_ratio = b_high / max(b_low, 1.0);
        double maturity = min(1.0, b_ratio);
        
        // Also consider power distribution across levels
        double p_low = 0, p_high = 0;
        for (int j = 0; j < N; j++) {
            p_low += (double)s.P[0][j] + (double)s.P[1][j];
            p_high += (double)s.P[2][j] + (double)s.P[3][j];
        }
        double p_ratio = p_high / max(p_low + p_high, 1.0);
        double power_maturity = min(1.0, p_ratio * 2.0);
        
        double dynamic_weight = 0.5 * maturity + 0.5 * power_maturity;
        
        double early_val = 0.05 + 0.10 * sig1;
        double late_val = 0.30 + 0.70 * sig2;
        g_synergy_blend = early_val + dynamic_weight * (late_val - early_val);
    }
    
    g_synergy_blend = max(0.0, min(1.0, g_synergy_blend));
}

void set_discount_rate_state(const State& s, int turn, int total) {
    double progress = (double)turn / (double)total;
    double remaining_frac = 1.0 - progress;
    double time_comp = 0.01 + 0.14 * pow(progress, 1.5);
    double prod_rate = s.production_rate();
    double state_comp;
    if (prod_rate > 1e-9) {
        double apple_turns = s.apples / prod_rate;
        double abundance_factor = 1.0 / (1.0 + apple_turns * 0.15);
        state_comp = 0.06 * abundance_factor;
    } else {
        state_comp = (s.apples > 10.0) ? 0.02 : 0.08;
    }
    double high_level_power = 0, low_level_power = 0;
    for (int j = 0; j < 10; j++) {
        for (int i = 2; i < 4; i++) high_level_power += (double)s.P[i][j];
        for (int i = 0; i < 2; i++) low_level_power += (double)s.P[i][j];
    }
    if (high_level_power < low_level_power * 0.3 && remaining_frac > 0.2) {
        state_comp *= 0.5;
    }
    double high_B_growth = 0;
    for (int j = 0; j < 10; j++) {
        for (int i = 1; i < 4; i++) {
            if (s.P[i][j] > 0) {
                high_B_growth += s.B[i][j] * (double)s.P[i][j];
            }
        }
    }
    if (high_B_growth > prod_rate * 2.0 && remaining_frac > 0.15) {
        state_comp *= 0.7;
    }
    g_discount_rate = time_comp + state_comp;
    g_discount_rate = max(0.005, min(g_discount_rate, 0.30));
}

inline void step(State& s, int i, int j) {
    if (i >= 0) {
        double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
        s.apples -= cost;
        s.P[i][j]++;
    }
    for (int j2 = 0; j2 < N; j2++)
        s.apples += (double)A[j2] * s.B[0][j2] * (double)s.P[0][j2];
    for (int j2 = 0; j2 < N; j2++)
        s.B[0][j2] += s.B[1][j2] * (double)s.P[1][j2];
    for (int j2 = 0; j2 < N; j2++)
        s.B[1][j2] += s.B[2][j2] * (double)s.P[2][j2];
    for (int j2 = 0; j2 < N; j2++)
        s.B[2][j2] += s.B[3][j2] * (double)s.P[3][j2];
}

void compute_level_priority(const State& s) {
    double level_eff[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        double sum = 0;
        for (int j = 0; j < 10; j++) {
            double eff = s.B[i][j] * (double)s.P[i][j] / max((double)C[i][j], 1.0);
            sum += eff;
        }
        level_eff[i] = sum / 10.0;
    }
    double total = 0;
    for (int i = 0; i < 4; i++) total += level_eff[i];
    double avg = total / 4.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 10; j++) {
            double eff = s.B[i][j] * (double)s.P[i][j] / max((double)C[i][j], 1.0);
            double level_ratio = avg > 1e-18 ? level_eff[i] / avg : 1.0;
            double boost = 1.0 / max(level_ratio, 0.1);
            boost = min(boost, 5.0);
            double ind_ratio = level_eff[i] > 1e-18 ? eff / level_eff[i] : 1.0;
            if (ind_ratio < 0.5) boost *= 1.3;
            else if (ind_ratio > 2.0) boost *= 0.7;
            if (i < 3) {
                double next_cap = s.B[i+1][j] * (double)s.P[i+1][j];
                double curr_cap = s.B[i][j] * (double)s.P[i][j];
                if (next_cap > curr_cap * 2 && (double)s.P[i][j] < 5) {
                    boost *= 1.2;
                }
            }
            g_level_priority[i][j] = max(boost, 0.1);
        }
    }
}

double compute_cascading_benefit(const State& s, int i, int j, int R) {
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    if (R <= i) return -1e18;
    double cascade_product = 1.0;
    for (int k = 0; k < i; k++) {
        cascade_product *= max((double)s.P[k][j], 1.0);
    }
    double marginal_rate = (double)A[j] * s.B[i][j] * cascade_product;
    int effective_turns = R - i;
    double propagation_discount = exp(-g_discount_rate * (double)i);
    double current_rate = s.production_rate();
    double base_rate = 0;
    for (int j2 = 0; j2 < N; j2++)
        base_rate += (double)A[j2] * s.B[0][j2];
    double growth = 0.01;
    if (base_rate > 0 && current_rate > base_rate) {
        growth = log(current_rate / base_rate) / max(1.0, (double)R * 0.1);
        growth = max(0.005, min(growth, 0.5));
    }
    double discounted_horizon = (double)effective_turns * propagation_discount;
    double benefit;
    if (growth > 1e-6 && effective_turns > 0) {
        double effective_horizon = max(1.0, discounted_horizon);
        double compound = (pow(1.0 + growth, effective_horizon) - 1.0) / growth;
        benefit = marginal_rate * compound;
    } else {
        benefit = marginal_rate * max(1.0, discounted_horizon);
    }
    double synergy = 1.0;
    for (int k = 0; k < i; k++) {
        synergy *= (1.0 + min((double)s.P[k][j], 100.0) * 0.005);
    }
    for (int k = i + 1; k < L; k++) {
        double pk = (double)s.P[k][j];
        if (pk > 0) synergy *= (1.0 + min(pk, 50.0) * 0.01);
    }
    benefit *= synergy;
    double dim = 1.0 / (1.0 + (double)s.P[i][j] * 0.03);
    if (s.P[i][j] > 10) dim *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.05);
    benefit *= dim;
    benefit *= g_phase_weight[i];
    if (benefit <= 0 || cost <= 0) return -1e18;
    return log2(benefit) - log2(cost);
}

pair<bool, double> eval_from(State s, const vector<pair<int,int>>& act, int start) {
    for (int t = start; t < T; t++) {
        auto [i, j] = act[t];
        if (i >= 0) {
            double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
            if (cost > s.apples + 1e-9) return {false, -1};
        }
        step(s, i, j);
    }
    return {true, s.apples};
}

pair<bool, double> eval_full(const vector<pair<int,int>>& act) {
    State s;
    s.init();
    return eval_from(s, act, 0);
}

double binom(int n, int k) {
    if (n < k || k < 0 || n < 0) return 0;
    double r = 1.0;
    for (int i = 0; i < k; i++) r *= (double)(n - i);
    for (int i = 2; i <= k; i++) r /= i;
    return r;
}

double compute_benefit(const State& s, int i, int j, int R,
                       const vector<double>& mults, double proj_factor) {
    if (R <= i) return -1e18;
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    double benefit = (double)A[j] * s.B[i][j];
    double cascade_product = 1.0;
    for (int k = 0; k < i; k++) {
        double pk = (double)s.P[k][j];
        if (pk < 1) pk = proj_factor;
        cascade_product *= pk;
    }
    benefit *= cascade_product;
    double compounding = 1.0;
    for (int k = 1; k <= i; k++) {
        double pk = (double)s.P[k][j];
        if (pk < 1) pk = proj_factor;
        double growth_ratio = s.B[k][j] * pk / max(s.B[k-1][j], 1.0);
        compounding *= (1.0 + min(growth_ratio, 10.0) * 0.0005 * (double)min(R, 200));
    }
    for (int k = 0; k < i; k++) {
        double pk = (double)s.P[k][j];
        if (pk < 1) pk = proj_factor;
        compounding *= (1.0 + min(pk, 1000.0) * 0.001 * (double)min(R, 100) / 100.0);
    }
    benefit *= compounding;
    if (i >= 1 && i < L - 1) {
        double pk_i = max((double)s.P[i][j], proj_factor);
        double next_B = s.B[i+1][j];
        double marginal = next_B * pk_i - s.B[i][j];
        if (marginal > 0) {
            double ratio = marginal / max(s.B[i][j], 1.0);
            benefit *= (1.0 + 0.05 * min(log2(1.0 + ratio), 5.0));
        }
    }
    if (i == 0 && L > 1) {
        double p0 = max((double)s.P[0][j], proj_factor);
        double marginal = s.B[1][j] * p0 - s.B[0][j];
        if (marginal > 0) {
            double ratio = marginal / max(s.B[0][j], 1.0);
            benefit *= (1.0 + 0.03 * min(log2(1.0 + ratio), 5.0));
        }
    }
    double discount = pow(0.998, (double)i);
    benefit *= discount;
    double b = binom(R, i + 1);
    benefit *= (b * mults[i]);
    if (benefit <= 0 || cost <= 0) return -1e18;
    double lp_term = 0.0;
    if (g_use_lp) {
        lp_term = log2(g_level_priority[i][j]);
    }
    double base_score = log2(benefit) - log2(cost) + g_weight_log[i][j] + lp_term;
    if (g_use_cascade) {
        set_phase_weights(T - R, T);
        set_discount_rate_state(s, T - R, T);
        double cb = compute_cascading_benefit(s, i, j, R);
        if (cb > -1e17) {
            return 0.85 * base_score + 0.15 * cb;
        }
    }
    // Dual-sigmoid phase blend: early game favors single-level, late game favors multi-level
    if (g_synergy_blend > 0.01 && i > 0) {
        double single_level = log2(max((double)A[j] * s.B[i][j], 1e-18)) - log2(max(cost, 1.0));
        return (1.0 - g_synergy_blend) * single_level + g_synergy_blend * base_score;
    }
    return base_score;
}

int compute_cascade_sensitivity(const State& s, int j, int R) {
    double best_ratio = -1e18;
    int best_level = -1;
    double current_rate = s.production_rate();
    double base_B = 0;
    for (int j2 = 0; j2 < N; j2++) base_B += (double)A[j2] * s.B[0][j2];
    double growth_rate = 0.01;
    if (base_B > 0 && current_rate > base_B) {
        growth_rate = log(current_rate / base_B) / max(1.0, (double)R * 0.1);
        growth_rate = max(growth_rate, 0.005);
        growth_rate = min(growth_rate, 0.5);
    }
    for (int i = 0; i < L; i++) {
        double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
        if (cost > s.apples) continue;
        double benefit = (double)A[j] * s.B[i][j];
        double cascade_product = 1.0;
        for (int k = 0; k < i; k++) {
            cascade_product *= max((double)s.P[k][j], 1.0);
        }
        benefit *= cascade_product;
        int effective_turns = R - i;
        double turn_ratio = (double)R > 0 ? (double)effective_turns / (double)R : 0.0;
        double time_discount;
        if (effective_turns < i) {
            time_discount = 0.05 * turn_ratio;
        } else {
            double compounding_factor = 1.0 + growth_rate * (double)effective_turns;
            double dynamic_rate_adj = 1.0 + 0.3 * ((double)R / 500.0);
            double alpha = max(0.5, growth_rate * 3.0);
            double hyperbolic_decay = (double)effective_turns / 
                ((double)effective_turns + alpha * (double)i);
            time_discount = hyperbolic_decay * turn_ratio * compounding_factor * dynamic_rate_adj;
        }
        double dim_factor = 1.0 / (1.0 + (double)s.P[i][j] * 0.03);
        if (s.P[i][j] > 10) {
            dim_factor *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.05);
        }
        benefit *= time_discount * dim_factor;
        double ratio = log2(max(benefit, 1e-18)) - log2(max(cost, 1.0));
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_level = i;
        }
    }
    return best_level;
}

pair<double, vector<double>> compute_adaptive_params(const State& s) {
    double avg_p[4] = {0, 0, 0, 0};
    for (int i = 0; i < L; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) sum += s.P[i][j];
        avg_p[i] = sum / N;
    }
    vector<double> level_importance(L, 0.0);
    vector<double> level_sensitivity_ratio(L, 0.0);
    for (int j = 0; j < N; j++) {
        double total_benefit = 0;
        array<double, 4> level_benefits = {0, 0, 0, 0};
        for (int i = 0; i < L; i++) {
            double benefit = (double)A[j] * s.B[i][j];
            double cascade_product = 1.0;
            for (int k = 0; k < i; k++) {
                cascade_product *= max((double)s.P[k][j], 1.0);
            }
            benefit *= cascade_product;
            double dim_factor = 1.0 / (1.0 + (double)s.P[i][j] * 0.03);
            if (s.P[i][j] > 10) {
                dim_factor *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.05);
            }
            benefit *= dim_factor;
            level_benefits[i] = benefit;
            total_benefit += benefit;
        }
        if (total_benefit > 0) {
            for (int i = 0; i < L; i++) {
                level_importance[i] += level_benefits[i];
                level_sensitivity_ratio[i] += level_benefits[i] / total_benefit;
            }
        }
    }
    double max_ratio = *max_element(level_sensitivity_ratio.begin(), level_sensitivity_ratio.end());
    if (max_ratio > 0) {
        for (int i = 0; i < L; i++) level_sensitivity_ratio[i] /= max_ratio;
    }
    double proj;
    if (avg_p[3] < 1.5) proj = 50.0;
    else if (avg_p[3] < 3) proj = 20.0;
    else if (avg_p[2] < 3) proj = 10.0;
    else if (avg_p[2] < 5) proj = 5.0;
    else proj = 2.0;
    double production_rate = s.production_rate();
    if (production_rate > 0 && s.apples > production_rate * 10) {
        proj *= 2.0;
    } else if (production_rate > 0 && s.apples < production_rate) {
        proj *= 0.5;
    }
    vector<double> mults(L, 1.0);
    for (int i = 1; i < L; i++) {
        if (avg_p[i] < avg_p[0] * 0.3) {
            mults[i] = max(1.0, 200.0 / (1.0 + avg_p[i]));
        } else if (avg_p[i] < avg_p[0] * 0.5) {
            mults[i] = max(1.0, 50.0 / (1.0 + avg_p[i]));
        }
        if (level_sensitivity_ratio[i] > 0.5 && avg_p[i] < avg_p[0] * 0.8) {
            mults[i] *= 1.5;
        }
        if (level_sensitivity_ratio[i] < 0.2 && avg_p[i] > avg_p[0] * 0.5) {
            mults[i] *= 0.7;
        }
    }
    double high_level_sensitivity = (level_sensitivity_ratio[2] + level_sensitivity_ratio[3]) * 0.5;
    if (high_level_sensitivity > 0.6 && avg_p[3] < 3) {
        proj *= 1.3;
    }
    return {proj, mults};
}

double mini_sim_v2(State s, int act_i, int act_j, int R,
                   const vector<double>& mults, double proj_factor) {
    set_synergy_blend_dual(T - R, T, s);
    double rate_before = s.production_rate();
    step(s, act_i, act_j);
    R--;
    int sim_turns;
    if (R > 350) sim_turns = 15;
    else if (R > 250) sim_turns = 12;
    else if (R > 150) sim_turns = 10;
    else if (R > 80) sim_turns = 8;
    else if (R > 40) sim_turns = 7;
    else sim_turns = 5;
    sim_turns = min(sim_turns, max(0, R));
    for (int t = 0; t < sim_turns && R > 0; t++, R--) {
        set_synergy_blend_dual(T - R, T, s);
        int best_i = -1, best_j = -1;
        double best_score = -1e18;
        for (int i2 = 0; i2 < L; i2++) {
            for (int j2 = 0; j2 < N; j2++) {
                double score = compute_benefit(s, i2, j2, R, mults, proj_factor);
                if (score > best_score) {
                    best_score = score;
                    best_i = i2;
                    best_j = j2;
                }
            }
        }
        step(s, best_i, best_j);
    }
    double rate_after = s.production_rate();
    double est;
    if (sim_turns > 0 && rate_before > 1.0 && rate_after > rate_before) {
        double ratio = rate_after / rate_before;
        double r = pow(ratio, 1.0 / (double)sim_turns) - 1.0;
        if (r > 1e-6 && R > 0) {
            r = min(r, 2.0);
            double growth = pow(1.0 + r, (double)R);
            double future = rate_after * (growth - 1.0) / r;
            est = s.apples + future;
        } else {
            est = s.apples + rate_after * (double)R * 0.3;
        }
    } else {
        est = s.apples + rate_after * (double)R * 0.3;
    }
    return est;
}

vector<pair<int,int>> run_greedy(const vector<double>& mults, double proj_factor,
                                  int use_sim, int target_j) {
    State s;
    s.init();
    vector<pair<int,int>> actions;
    actions.reserve(T);
    if (g_use_lp) compute_level_priority(s);
    int j_start = (target_j >= 0) ? target_j : 0;
    int j_end = (target_j >= 0) ? target_j + 1 : N;
    for (int t = 0; t < T; t++) {
        int R = T - t;
        set_synergy_blend_dual(t, T, s);
        if (g_use_lp && t > 0 && t % 5 == 0) compute_level_priority(s);
        struct Cand { int i, j; double score; };
        vector<Cand> cands;
        for (int i = 0; i < L; i++) {
            for (int j = j_start; j < j_end; j++) {
                double score = compute_benefit(s, i, j, R, mults, proj_factor);
                if (score > -1e17) cands.push_back({i, j, score});
            }
        }
        if (cands.empty()) {
            actions.push_back({-1, -1});
            step(s, -1, -1);
            continue;
        }
        sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            return a.score > b.score;
        });
        int chosen_i = cands[0].i, chosen_j = cands[0].j;
        if (use_sim > 0) {
            int top_k = min(use_sim, (int)cands.size());
            double best_sim = mini_sim_v2(s, -1, -1, R, mults, proj_factor);
            chosen_i = -1; chosen_j = -1;
            for (int c = 0; c < top_k; c++) {
                double sim_result = mini_sim_v2(s, cands[c].i, cands[c].j, R, mults, proj_factor);
                if (sim_result > best_sim) {
                    best_sim = sim_result;
                    chosen_i = cands[c].i;
                    chosen_j = cands[c].j;
                }
            }
        }
        step(s, chosen_i, chosen_j);
        actions.push_back({chosen_i, chosen_j});
    }
    g_synergy_blend = 0.0;
    return actions;
}

vector<pair<int,int>> run_greedy_cascade(int phase_mode) {
    State s;
    s.init();
    vector<pair<int,int>> actions;
    actions.reserve(T);
    for (int t = 0; t < T; t++) {
        int R = T - t;
        if (phase_mode == 0) {
            set_phase_weights(t, T);
        } else if (phase_mode == 1) {
            g_phase_weight = {1.0, 1.0, 1.0, 1.0};
        } else if (phase_mode == 2) {
            double progress = (double)t / (double)T;
            if (progress < 0.2) { g_phase_weight = {0.2, 2.0, 3.0, 2.0}; }
            else if (progress < 0.5) { g_phase_weight = {0.5, 1.5, 2.0, 2.5}; }
            else if (progress < 0.8) { g_phase_weight = {1.0, 1.0, 1.0, 1.5}; }
            else { g_phase_weight = {2.0, 0.5, 0.3, 0.2}; }
        } else if (phase_mode == 3) {
            double progress = (double)t / (double)T;
            if (progress < 0.1) { g_phase_weight = {0.3, 1.5, 2.5, 3.0}; }
            else if (progress < 0.3) { g_phase_weight = {0.5, 1.0, 2.0, 2.5}; }
            else if (progress < 0.6) { g_phase_weight = {1.0, 1.0, 1.5, 1.5}; }
            else { g_phase_weight = {1.5, 1.0, 0.5, 0.3}; }
        }
        set_discount_rate_state(s, t, T);
        int best_i = -1, best_j = -1;
        double best_score = -1e18;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                double score = compute_cascading_benefit(s, i, j, R);
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_score < -1e17) {
            actions.push_back({-1, -1});
            step(s, -1, -1);
        } else {
            actions.push_back({best_i, best_j});
            step(s, best_i, best_j);
        }
    }
    return actions;
}

double eval_state(const State& s, int R, double growth_mult = 1.0) {
    if (R <= 0) return s.apples;
    double rate = s.production_rate();
    if (rate < 1e-9) return s.apples;
    double time_weight = (double)R / (double)T;
    time_weight = max(0.1, min(1.0, time_weight));
    double d_rate = 0;
    for (int j = 0; j < N; j++)
        d_rate += (double)A[j] * s.B[1][j] * (double)s.P[1][j] * (double)s.P[0][j];
    double g0 = (rate > 1e-9) ? d_rate / rate : 0;
    double dd_rate = 0;
    for (int j = 0; j < N; j++)
        dd_rate += (double)A[j] * s.B[2][j] * (double)s.P[2][j] * (double)s.P[1][j] * (double)s.P[0][j];
    double g1 = (d_rate > 1e-9) ? dd_rate / d_rate : 0;
    double ddd_rate = 0;
    for (int j = 0; j < N; j++)
        ddd_rate += (double)A[j] * s.B[3][j] * (double)s.P[3][j] * (double)s.P[2][j] * (double)s.P[1][j] * (double)s.P[0][j];
    double g2 = (dd_rate > 1e-9) ? ddd_rate / dd_rate : 0;
    double eff_growth = g0 * (0.3 + 0.7 * time_weight);
    if (g1 > 0) eff_growth *= (1.0 + g1 * (double)R * 0.3 * time_weight);
    if (g2 > 0) eff_growth *= (1.0 + g2 * (double)R * 0.2 * time_weight);
    eff_growth *= growth_mult;
    eff_growth = min(eff_growth, 0.5);
    double future;
    if (eff_growth > 1e-6) {
        double comp = pow(1.0 + eff_growth, (double)R);
        if (comp > 1e200) comp = 1e200;
        future = rate * (comp - 1.0) / eff_growth;
    } else {
        future = rate * (double)R;
    }
    return s.apples + future;
}

vector<pair<int,int>> run_beam_search(int beam_width, double growth_mult = 1.0) {
    vector<State> cur_beam(1);
    cur_beam[0].init();
    vector<vector<int>> all_parents(T);
    vector<vector<pair<int,int>>> all_actions(T);
    for (int t = 0; t < T; t++) {
        int R = T - t;
        int cur_size = (int)cur_beam.size();
        int max_cands = cur_size * (L * N + 1);
        vector<int> cand_parent(max_cands);
        vector<pair<int,int>> cand_act(max_cands);
        vector<double> cand_eval(max_cands);
        vector<State> cand_state(max_cands);
        int ncands = 0;
        for (int b = 0; b < cur_size; b++) {
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < N; j++) {
                    double cost = (double)C[i][j] * (double)(cur_beam[b].P[i][j] + 1);
                    if (cost > cur_beam[b].apples) continue;
                    cand_parent[ncands] = b;
                    cand_act[ncands] = {i, j};
                    cand_state[ncands] = cur_beam[b];
                    step(cand_state[ncands], i, j);
                    cand_eval[ncands] = eval_state(cand_state[ncands], R - 1, growth_mult);
                    ncands++;
                }
            }
            cand_parent[ncands] = b;
            cand_act[ncands] = {-1, -1};
            cand_state[ncands] = cur_beam[b];
            step(cand_state[ncands], -1, -1);
            cand_eval[ncands] = eval_state(cand_state[ncands], R - 1, growth_mult);
            ncands++;
        }
        vector<int> indices(ncands);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return cand_eval[a] > cand_eval[b];
        });
        int next_size = min(beam_width, ncands);
        all_parents[t].resize(next_size);
        all_actions[t].resize(next_size);
        vector<State> next_beam(next_size);
        for (int b = 0; b < next_size; b++) {
            int idx = indices[b];
            all_parents[t][b] = cand_parent[idx];
            all_actions[t][b] = cand_act[idx];
            next_beam[b] = cand_state[idx];
        }
        cur_beam = move(next_beam);
    }
    vector<pair<int,int>> result(T);
    int idx = 0;
    for (int t = T - 1; t >= 0; t--) {
        result[t] = all_actions[t][idx];
        idx = all_parents[t][idx];
    }
    return result;
}

double compute_benefit_v2(const State& s, int i, int j, int R) {
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    if (R <= i) return -1e18;
    double marginal_rate = (double)A[j] * s.B[i][j];
    for (int k = 0; k < i; k++)
        marginal_rate *= max((double)s.P[k][j], 1.0);
    int effective_turns = R - i;
    double time_weight = (double)R / (double)T;
    time_weight = max(0.1, min(1.0, time_weight));
    double rate = s.production_rate();
    double g0 = 0;
    if (rate > 1e-9) {
        double d_rate = 0;
        for (int j2 = 0; j2 < N; j2++)
            d_rate += (double)A[j2] * s.B[1][j2] * (double)s.P[1][j2] * (double)s.P[0][j2];
        g0 = d_rate / rate;
    }
    double d_rate = 0;
    for (int j2 = 0; j2 < N; j2++)
        d_rate += (double)A[j2] * s.B[1][j2] * (double)s.P[1][j2] * (double)s.P[0][j2];
    double dd_rate = 0;
    for (int j2 = 0; j2 < N; j2++)
        dd_rate += (double)A[j2] * s.B[2][j2] * (double)s.P[2][j2] * (double)s.P[1][j2] * (double)s.P[0][j2];
    double g1 = (d_rate > 1e-9) ? dd_rate / d_rate : 0;
    double eff_growth = g0 * (0.3 + 0.7 * time_weight);
    if (g1 > 0) eff_growth *= (1.0 + g1 * (double)R * 0.3 * time_weight);
    eff_growth = min(eff_growth, 0.5);
    double future;
    if (eff_growth > 1e-6) {
        double comp = pow(1.0 + eff_growth, (double)effective_turns);
        if (comp > 1e200) comp = 1e200;
        future = marginal_rate * (comp - 1.0) / eff_growth;
    } else {
        future = marginal_rate * (double)effective_turns;
    }
    double dim = 1.0 / (1.0 + (double)s.P[i][j] * 0.02);
    future *= dim;
    double progress = 1.0 - (double)R / (double)T;
    if (progress > 0.7) {
        future *= 1.0 + (progress - 0.7) * (double)(L - i) * 2.0;
    } else if (progress < 0.15) {
        future *= 1.0 + (0.15 - progress) * (double)i * 1.5;
    }
    if (future <= 0 || cost <= 0) return -1e18;
    double score_v2 = log2(future) - log2(cost);
    // Dual-sigmoid phase blend
    if (g_synergy_blend > 0.01 && i > 0) {
        double single_level = log2(max((double)A[j] * s.B[i][j] * (double)effective_turns, 1e-18)) - log2(max(cost, 1.0));
        return (1.0 - g_synergy_blend) * single_level + g_synergy_blend * score_v2;
    }
    return score_v2;
}

vector<pair<int,int>> run_greedy_v2() {
    State s;
    s.init();
    vector<pair<int,int>> actions;
    actions.reserve(T);
    for (int t = 0; t < T; t++) {
        int R = T - t;
        set_synergy_blend_dual(t, T, s);
        int best_i = -1, best_j = -1;
        double best_score = -1e18;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                double score = compute_benefit_v2(s, i, j, R);
                if (score > best_score) {
                    best_score = score;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        if (best_score < -1e17) {
            actions.push_back({-1, -1});
            step(s, -1, -1);
        } else {
            actions.push_back({best_i, best_j});
            step(s, best_i, best_j);
        }
    }
    g_synergy_blend = 0.0;
    return actions;
}

void update_synergy_weights(const State& s) {
    double high_sat = 0, low_sat = 0;
    for (int j = 0; j < N; j++) {
        for (int i = 2; i < L; i++) {
            double cap = s.B[i][j] * (double)s.P[i][j];
            high_sat = max(high_sat, cap);
        }
        for (int i = 0; i < 2; i++) {
            double cap = s.B[i][j] * (double)s.P[i][j];
            low_sat = max(low_sat, cap);
        }
    }
    for (int i = 0; i < 4; i++) g_synergy_level_weight[i] = 1.0;
    g_synergy_compound_weight = 1.0;
    g_synergy_balance_weight = 1.0;

    if (high_sat > 1e6) {
        for (int i = 2; i < L; i++) g_synergy_level_weight[i] = 0.8;
        for (int i = 0; i < 2; i++) g_synergy_level_weight[i] = 1.15;
        g_synergy_compound_weight = 0.85;
        g_synergy_balance_weight = 1.2;
    }
}

double compute_cascading_impact(const State& s, int i, int j, int R) {
    if (R <= 0) return 0.0;
    double full_cascade = s.B[i][j];
    for (int k = 0; k < i; k++) {
        full_cascade *= max((double)s.P[k][j], 1.0);
    }
    double marginal_per_turn = (double)A[j] * full_cascade;
    double path_sum = 0.0;
    for (int d = 0; d <= i; d++) {
        double weight = pow(0.8, (double)d);
        int eff_turns = max(0, R - d);
        if (eff_turns <= 0) continue;
        path_sum += marginal_per_turn * (double)eff_turns * weight;
    }
    double compounding_bonus = 1.0;
    for (int k = i + 1; k < L; k++) {
        double bp = s.B[k][j] * (double)s.P[k][j];
        if (bp > 0 && s.B[i][j] > 0) {
            double ratio = bp / s.B[i][j];
            compounding_bonus *= (1.0 + min(log2(1.0 + ratio), 5.0) * 0.1);
        }
    }
    path_sum *= compounding_bonus;
    return path_sum;
}

array<double, 4> compute_level_path_sums(const State& s, int R) {
    array<double, 4> sums = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < N; j++) {
            sums[i] += compute_cascading_impact(s, i, j, R);
        }
    }
    return sums;
}

array<double, 4> compute_level_utilization(const State& s) {
    array<double, 4> weighted_p = {0, 0, 0, 0};
    array<double, 4> total_b = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < N; j++) {
            weighted_p[i] += s.B[i][j] * (double)s.P[i][j];
            total_b[i] += s.B[i][j];
        }
        weighted_p[i] /= max(total_b[i], 1.0);
    }
    double max_wp = 0;
    for (int i = 0; i < 4; i++) max_wp = max(max_wp, weighted_p[i]);
    array<double, 4> util = {0, 0, 0, 0};
    if (max_wp < 1e-18) max_wp = 1.0;
    for (int i = 0; i < 4; i++) {
        util[i] = weighted_p[i] / max_wp;
    }
    return util;
}

double compute_synergy_raw(const State& s, int i, int j, int R) {
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    double marginal = (double)A[j] * s.B[i][j];
    for (int k = 0; k < i; k++)
        marginal *= max((double)s.P[k][j], 1.0);
    marginal *= g_synergy_level_weight[i];
    double comp_factor = 1.0 + (double)i * ((double)R / (double)T) * 0.5;
    comp_factor *= g_synergy_compound_weight;
    double offspring = 0;
    for (int k = i + 1; k < L; k++)
        offspring += s.B[k][j] * (double)s.P[k][j];
    double offspring_factor = 1.0 + min(offspring / max(s.B[i][j], 1.0), 10.0) * 0.1;
    double saturation = 1.0 / (1.0 + (double)s.P[i][j] * 0.02);
    if (s.P[i][j] > 10)
        saturation *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.04);
    double level_sum = 0;
    int level_cnt = 0;
    for (int k = 0; k < L; k++) {
        if (k != i) { level_sum += (double)s.P[k][j]; level_cnt++; }
    }
    double level_avg = (level_cnt > 0) ? level_sum / level_cnt : 0;
    double balance = 1.0;
    if (level_avg > 0.5 && (double)s.P[i][j] > level_avg * 3.0)
        balance = 0.5;
    balance *= g_synergy_balance_weight;
    double cascading_impact = compute_cascading_impact(s, i, j, R);
    double impact_factor = 1.0 + log2(max(cascading_impact, 1.0)) * 0.05;
    return marginal * comp_factor * offspring_factor * saturation * balance * impact_factor;
}

double compute_cascading_potential(const State& s, int i, int j, int R) {
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    if (R <= i) return -1e18;

    double marginal_B = s.B[i][j];
    double cascade_product = 1.0;
    for (int k = 0; k < i; k++) {
        cascade_product *= max((double)s.P[k][j], 1.0);
    }
    double marginal_rate = (double)A[j] * marginal_B * cascade_product;

    int effective_turns = R - i;

    double discount_rate = 0.95;
    int compounding_horizon;
    if (effective_turns <= 30) {
        compounding_horizon = effective_turns;
    } else if (effective_turns <= 60) {
        compounding_horizon = 20 + (effective_turns - 30) / 2;
    } else {
        compounding_horizon = 35 + (effective_turns - 60) / 4;
    }
    compounding_horizon = max(1, min(compounding_horizon, effective_turns));

    double discount_multiplier = (1.0 - pow(discount_rate, (double)compounding_horizon)) / (1.0 - discount_rate);
    double future_apples = marginal_rate * discount_multiplier;

    double comp = binom(effective_turns, i + 1);
    if (comp < 1.0) comp = (double)effective_turns;
    double temporal_weight = pow(discount_rate, (double)i);
    comp *= temporal_weight;

    double compounding_bonus = 1.0;
    if (effective_turns >= 20 && effective_turns <= 50) {
        compounding_bonus = 1.0 + 0.3 * min(comp / max(discount_multiplier, 1.0), 5.0);
    } else if (effective_turns > 50) {
        compounding_bonus = 1.0 + 0.15 * min(comp / max(discount_multiplier, 1.0), 5.0);
    }
    future_apples *= compounding_bonus;

    double downstream = 1.0;
    for (int k = i + 1; k < L; k++) {
        double bp = s.B[k][j] * (double)s.P[k][j];
        if (bp > 0 && s.B[i][j] > 0) {
            double ratio = bp / s.B[i][j];
            downstream *= (1.0 + min(log2(1.0 + ratio), 5.0) * 0.1);
        }
    }
    future_apples *= downstream;

    double level_balance = 1.0;
    if (i > 0 && (double)s.P[i-1][j] > (double)s.P[i][j] * 2.0) {
        level_balance *= 1.3;
    }
    if (i < L - 1 && (double)s.P[i+1][j] > (double)s.P[i][j] * 2.0) {
        level_balance *= 1.2;
    }
    future_apples *= level_balance;

    double dim = 1.0 / (1.0 + (double)s.P[i][j] * 0.02);
    if (s.P[i][j] > 10)
        dim *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.04);
    future_apples *= dim;

    if (future_apples <= 0 || cost <= 0) return -1e18;
    return log2(max(future_apples, 1e-18)) - log2(max(cost, 1.0));
}

double compute_cascading_synergy(const State& s, int i, int j, int R) {
    double cost = (double)C[i][j] * (double)(s.P[i][j] + 1);
    if (cost > s.apples) return -1e18;
    if (R <= i) return -1e18;

    double immediate_B = s.B[i][j];
    double cascade_product = 1.0;
    for (int k = 0; k < i; k++) {
        cascade_product *= max((double)s.P[k][j], 1.0);
    }

    double base_marginal = (double)A[j] * immediate_B * cascade_product;

    int effective_turns = R - i;

    double time_to_maturity = (double)i * 0.3 + 1.0;
    double remaining_frac = (double)R / (double)T;
    double maturity_discount = 1.0 / (1.0 + time_to_maturity * (1.0 - remaining_frac) * 0.5);
    base_marginal *= maturity_discount;

    double level_amplification = 1.0;
    for (int k = 0; k < i; k++) {
        double p_k = (double)s.P[k][j];
        double b_ratio = s.B[k+1][j] / max(s.B[k][j], 1.0);
        double amp_k = 1.0 + min(p_k * b_ratio * 0.01, 2.0);
        level_amplification *= amp_k;
    }
    base_marginal *= level_amplification;

    double compounding_accel = 1.0;
    for (int k = 0; k <= i; k++) {
        double g_k = 0;
        if (k < L - 1) {
            double growth_at_k = s.B[k+1][j] * (double)s.P[k+1][j];
            g_k = growth_at_k / max(s.B[k][j], 1.0);
        }
        double term = 1.0 + min(g_k, 5.0) * (double)effective_turns / 
                       ((double)(k + 1) * (double)max(i + 1, 1));
        compounding_accel *= term;
    }

    double log_poly = 0;
    for (int k = 1; k <= i + 1; k++) {
        log_poly += log(max((double)effective_turns / (double)k, 1e-18));
    }
    double poly_base = exp(min(log_poly, 100.0));

    double benefit_poly = base_marginal * poly_base * compounding_accel;

    double discount_rate = 0.95;
    int horizon;
    if (effective_turns <= 30) {
        horizon = effective_turns;
    } else if (effective_turns <= 100) {
        horizon = 30 + (effective_turns - 30) / 3;
    } else {
        horizon = 50 + (effective_turns - 100) / 5;
    }
    horizon = max(1, min(horizon, effective_turns));

    double discount_mult = (1.0 - pow(discount_rate, (double)horizon)) / (1.0 - discount_rate);
    double benefit_discounted = base_marginal * discount_mult * compounding_accel;

    double benefit = 0.55 * benefit_poly + 0.45 * benefit_discounted;

    double downstream = 1.0;
    for (int k = i + 1; k < L; k++) {
        double bp = s.B[k][j] * (double)s.P[k][j];
        if (bp > 0 && s.B[i][j] > 0) {
            double ratio = bp / s.B[i][j];
            downstream *= (1.0 + min(log2(1.0 + ratio), 5.0) * 0.1);
        }
    }
    benefit *= downstream;

    int active_levels = 0;
    for (int k = 0; k < L; k++) {
        if (s.P[k][j] > 0) active_levels++;
    }
    double cross_synergy = 1.0 + 0.08 * (double)active_levels;
    benefit *= cross_synergy;

    double level_balance = 1.0;
    if (i > 0 && (double)s.P[i-1][j] > (double)s.P[i][j] * 2.0) level_balance *= 1.3;
    if (i < L - 1 && (double)s.P[i+1][j] > (double)s.P[i][j] * 2.0) level_balance *= 1.2;
    benefit *= level_balance;

    double baseline_gain = (double)A[j] * s.B[i][j] * (double)(s.P[i][j] + 1);
    double multi_level_impact = 0;
    int active_levels_count = 0;
    for (int k = 0; k < L; k++) {
        double level_impact = s.B[k][j] * (double)s.P[k][j];
        if (level_impact > 0) {
            multi_level_impact += level_impact;
            active_levels_count++;
        }
    }
    double synergy_threshold_mult = 1.0;
    if (active_levels_count >= 2 && multi_level_impact > baseline_gain * 0.5) {
        synergy_threshold_mult = 1.0 + min(multi_level_impact / max(baseline_gain, 1.0) * 0.1, 2.0);
    } else if (active_levels_count < 2) {
        synergy_threshold_mult = 0.7;
    }
    benefit *= synergy_threshold_mult;

    double dim = 1.0 / (1.0 + (double)s.P[i][j] * 0.02);
    if (s.P[i][j] > 10)
        dim *= 1.0 / (1.0 + (double)(s.P[i][j] - 10) * 0.04);
    benefit *= dim;

    if (benefit <= 0 || cost <= 0) return -1e18;
    return log2(max(benefit, 1e-18)) - log2(max(cost, 1.0));
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> N >> L >> T >> K;
    A.resize(N);
    for (int j = 0; j < N; j++) cin >> A[j];
    C.resize(L, vector<long long>(N));
    for (int i = 0; i < L; i++)
        for (int j = 0; j < N; j++)
            cin >> C[i][j];
    
    auto start_time = chrono::high_resolution_clock::now();
    auto elapsed = [&]() {
        return (double)chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count();
    };
    
    struct Solution {
        double score;
        vector<pair<int,int>> actions;
    };
    vector<Solution> top_solutions;
    
    auto consider = [&](vector<pair<int,int>>& actions, double score) {
        if (score <= 0) return;
        for (auto& s : top_solutions) {
            if (s.actions == actions) return;
        }
        top_solutions.push_back({score, actions});
        sort(top_solutions.begin(), top_solutions.end(), [](const Solution& a, const Solution& b) {
            return a.score > b.score;
        });
        if (top_solutions.size() > 5) top_solutions.pop_back();
    };
    
    // Phase 1: Beam search
    for (auto [bw, gm] : initializer_list<pair<int,double>>{{8, 0.5}, {8, 1.0}, {8, 2.0}, {16, 1.0}, {24, 1.0}}) {
        if (elapsed() > 500) break;
        auto actions = run_beam_search(bw, gm);
        auto [valid, score] = eval_full(actions);
        if (valid) consider(actions, score);
    }
    
    // Phase 2: Greedy v2
    {
        if (elapsed() < 550) {
            auto actions = run_greedy_v2();
            auto [valid, score] = eval_full(actions);
            if (valid) consider(actions, score);
        }
    }
    
    // Phase 3: Existing greedy
    vector<vector<double>> mult_sets = {
        {1, 1, 1, 1}, {1, 1, 1, 100}, {1, 10, 100, 1000}, {1, 5, 25, 125},
        {1, 1, 1, 10000}, {0.001, 0.01, 0.1, 1}, {1, 1, 1, 500}, {1, 2, 5, 10},
    };
    set_weights_uniform();
    for (const auto& ms : mult_sets) {
        for (double pf : {1.0, 5.0, 20.0}) {
            if (elapsed() > 650) break;
            auto actions = run_greedy(ms, pf, 0, -1);
            auto [valid, score] = eval_full(actions);
            if (valid) consider(actions, score);
        }
        if (elapsed() > 650) break;
    }
    
    // Phase 3b: Weight-based greedy
    {
        vector<vector<double>> weight_mult_sets = {{1, 1, 1, 1}, {1, 10, 100, 1000}, {1, 1, 1, 100}};
        vector<double> weight_proj_factors = {1.0, 20.0};
        for (int wm : {1, 2}) {
            if (elapsed() > 700) break;
            set_weights(wm);
            for (const auto& ms : weight_mult_sets) {
                if (elapsed() > 700) break;
                for (double pf : weight_proj_factors) {
                    if (elapsed() > 700) break;
                    auto actions = run_greedy(ms, pf, 0, -1);
                    auto [valid, score] = eval_full(actions);
                    if (valid) consider(actions, score);
                }
            }
        }
        set_weights_uniform();
    }
    
    // Phase 4: Sim greedy
    set_weights_uniform();
    for (const auto& ms : {vector<double>{1,1,1,1}, {1,1,1,100}, {1,10,100,1000}}) {
        for (double pf : {1.0, 20.0}) {
            if (elapsed() > 780) break;
            auto actions = run_greedy(ms, pf, 5, -1);
            auto [valid, score] = eval_full(actions);
            if (valid) consider(actions, score);
        }
        if (elapsed() > 780) break;
    }
    
    // Phase 5: Per-ID greedy
    set_weights_uniform();
    for (int j = 0; j < N; j++) {
        for (const auto& ms : {vector<double>{1,1,1,1}, {1,1,1,100}}) {
            for (double pf : {1.0, 20.0}) {
                if (elapsed() > 820) break;
                auto actions = run_greedy(ms, pf, 0, j);
                auto [valid, score] = eval_full(actions);
                if (valid) consider(actions, score);
            }
            if (elapsed() > 820) break;
        }
        if (elapsed() > 820) break;
    }
    
    // Phase 6: Adaptive params
    {
        if (!top_solutions.empty() && elapsed() < 880) {
            State final_s;
            final_s.init();
            for (int t = 0; t < T; t++) {
                step(final_s, top_solutions[0].actions[t].first, top_solutions[0].actions[t].second);
            }
            auto [proj, mults] = compute_adaptive_params(final_s);
            set_weights_uniform();
            for (double pf : {proj, proj * 0.5, proj * 2.0, 5.0, 20.0, 50.0}) {
                if (elapsed() > 880) break;
                auto actions = run_greedy(mults, pf, 0, -1);
                auto [valid, score] = eval_full(actions);
                if (valid) consider(actions, score);
            }
            if (elapsed() < 880) {
                auto actions2 = run_greedy(mults, proj, 5, -1);
                auto [valid2, score2] = eval_full(actions2);
                if (valid2) consider(actions2, score2);
            }
            set_weights_uniform();
        }
    }
    
    // Phase 7: Cascade greedy
    {
        for (int pm = 0; pm < 2 && elapsed() < 900; pm++) {
            auto actions = run_greedy_cascade(pm);
            auto [valid, score] = eval_full(actions);
            if (valid) consider(actions, score);
        }
    }
    
    // Phase 8: LP greedy
    {
        g_use_lp = true;
        set_weights_uniform();
        for (const auto& ms : {vector<double>{1,1,1,1}, {1,1,1,100}, {1,10,100,1000}}) {
            for (double pf : {5.0, 20.0}) {
                if (elapsed() > 920) break;
                auto actions = run_greedy(ms, pf, 0, -1);
                auto [valid, score] = eval_full(actions);
                if (valid) consider(actions, score);
            }
            if (elapsed() > 920) break;
        }
        g_use_lp = false;
        set_weights_uniform();
    }
    
    if (top_solutions.empty()) {
        vector<pair<int,int>> dn(T, {-1, -1});
        auto [valid, score] = eval_full(dn);
        top_solutions.push_back({score, dn});
    }
    
    double best_score = top_solutions[0].score;
    vector<pair<int,int>> best_actions = top_solutions[0].actions;
    
    // Phase 9: Simulated Annealing
    double sa_end = 1750;
    double sa_T0 = 3.0;
    double sa_k = log(sa_T0 / 0.001);
    int num_seeds = (int)top_solutions.size();
    double last_injection_time = elapsed();
    
    for (int seed_idx = 0; seed_idx < num_seeds && elapsed() < sa_end; seed_idx++) {
        double seed_start = elapsed();
        double remaining = sa_end - seed_start;
        double per_seed = remaining / (num_seeds - seed_idx);
        double seed_end = min(sa_end, seed_start + per_seed);
        
        vector<pair<int,int>> current_actions = top_solutions[seed_idx].actions;
        double current_score = top_solutions[seed_idx].score;
        
        vector<State> prefix(T + 1);
        prefix[0].init();
        for (int t = 0; t < T; t++) {
            prefix[t+1] = prefix[t];
            step(prefix[t+1], current_actions[t].first, current_actions[t].second);
        }
        
        mt19937 rng(42 + seed_idx * 7 + (int)elapsed());
        deque<bool> accept_window;
        const int WINDOW_SIZE = 50;
        
        int sa_iter = 0;
        vector<double> best_log_history;
        g_diminishing_factor = 1.0;
        
        auto get_acceptance_rate = [&]() -> double {
            if (accept_window.empty()) return 0.5;
            int accepts = 0;
            for (bool b : accept_window) if (b) accepts++;
            return (double)accepts / (double)accept_window.size();
        };
        auto record_attempt = [&](bool accepted) {
            accept_window.push_back(accepted);
            if ((int)accept_window.size() > WINDOW_SIZE) accept_window.pop_front();
        };
        auto get_temp = [&](double progress, double acc_rate) -> double {
            progress = max(0.0, min(1.0, progress));
            double adjusted_progress = min(1.0, progress / 0.8);
            double base_temp = sa_T0 * exp(-sa_k * pow(adjusted_progress, 0.75));
            if (acc_rate < 0.1) base_temp *= 2.0;
            else if (acc_rate < 0.2) base_temp *= 1.5;
            else if (acc_rate < 0.3) base_temp *= 1.2;
            base_temp *= g_diminishing_factor;
            return max(base_temp, 1e-6);
        };
        auto get_segment_len = [&](double temp, double acc_rate) -> int {
            double temp_factor = max(0.0, min(1.0, temp / sa_T0));
            int base = 3 + (int)(10.0 * temp_factor);
            int bonus = (int)(5.0 * acc_rate);
            return max(3, min(20, base + bonus));
        };
        auto accept_worse = [&](double delta_log, double temp) -> bool {
            if (delta_log >= 0) return true;
            double exponent = delta_log / temp;
            if (exponent < -700.0) return false;
            double prob = exp(exponent);
            return prob > (double)(rng()) / (double)4294967295.0;
        };
        vector<pair<int,int>> affordable;
        affordable.reserve(L * N + 1);
        
        while (elapsed() < seed_end) {
            sa_iter++;
            update_synergy_weights(prefix[T]);
            
            if (sa_iter % 200 == 0) {
                best_log_history.push_back(log2(max(best_score, 1.0)));
                if (best_log_history.size() > 10) {
                    double oldest = best_log_history[best_log_history.size() - 10];
                    double newest = best_log_history.back();
                    double improvement_per_iter = (newest - oldest) / (200.0 * 10);
                    if (improvement_per_iter < 0.001) {
                        g_diminishing_factor = 0.5;
                    } else {
                        g_diminishing_factor = 1.0;
                    }
                }
            }
            
            // Injection mechanism
            if (elapsed() - last_injection_time > 250 && elapsed() < sa_end - 300) {
                last_injection_time = elapsed();
                State best_final;
                best_final.init();
                for (int t = 0; t < T; t++) {
                    step(best_final, best_actions[t].first, best_actions[t].second);
                }
                auto [proj, mults] = compute_adaptive_params(best_final);
                set_weights_uniform();
                auto inj_actions = run_greedy(mults, proj, 0, -1);
                auto [inj_valid, inj_score] = eval_full(inj_actions);
                if (inj_valid) {
                    consider(inj_actions, inj_score);
                    if (inj_score > best_score) {
                        best_score = inj_score;
                        best_actions = inj_actions;
                    }
                    if (inj_score > current_score * 1.05) {
                        current_actions = inj_actions;
                        current_score = inj_score;
                        for (int t = 0; t < T; t++) {
                            prefix[t+1] = prefix[t];
                            step(prefix[t+1], current_actions[t].first, current_actions[t].second);
                        }
                    }
                }
                if (elapsed() < sa_end - 400) {
                    auto bs_actions = run_beam_search(8, 1.0);
                    auto [bs_valid, bs_score] = eval_full(bs_actions);
                    if (bs_valid) {
                        consider(bs_actions, bs_score);
                        if (bs_score > best_score) {
                            best_score = bs_score;
                            best_actions = bs_actions;
                        }
                        if (bs_score > current_score * 1.05) {
                            current_actions = bs_actions;
                            current_score = bs_score;
                            for (int t = 0; t < T; t++) {
                                prefix[t+1] = prefix[t];
                                step(prefix[t+1], current_actions[t].first, current_actions[t].second);
                            }
                        }
                    }
                }
                for (int pm = 0; pm < 2; pm++) {
                    auto inj_actions2 = run_greedy_cascade(pm);
                    auto [inj_valid2, inj_score2] = eval_full(inj_actions2);
                    if (inj_valid2) {
                        consider(inj_actions2, inj_score2);
                        if (inj_score2 > best_score) {
                            best_score = inj_score2;
                            best_actions = inj_actions2;
                        }
                        if (inj_score2 > current_score * 1.05) {
                            current_actions = inj_actions2;
                            current_score = inj_score2;
                            for (int t = 0; t < T; t++) {
                                prefix[t+1] = prefix[t];
                                step(prefix[t+1], current_actions[t].first, current_actions[t].second);
                            }
                        }
                    }
                }
                set_weights_uniform();
            }
            
            double progress = (elapsed() - seed_start) / max(1.0, per_seed);
            progress = max(0.0, min(1.0, progress));
            double acc_rate = get_acceptance_rate();
            double temp = get_temp(progress, acc_rate);
            int seg_len = get_segment_len(temp, acc_rate);
            int move_rand = rng() % 100;
            int move_type;
            
            // Three-phase move type allocation with enhanced cascade-potential in mid-game (40-60%)
            bool mid_game = (progress >= 0.4 && progress < 0.6);
            if (mid_game) {
                // Enhanced cascade-potential moves during transition phase (40-60%)
                if (move_rand < 10) move_type = 0;      // 10%
                else if (move_rand < 20) move_type = 1;  // 10%
                else if (move_rand < 30) move_type = 2;  // 10%
                else if (move_rand < 36) move_type = 3;  // 6%
                else if (move_rand < 39) move_type = 4;  // 3%
                else if (move_rand < 43) move_type = 5;  // 4%
                else if (move_rand < 47) move_type = 6;  // 4%
                else if (move_rand < 52) move_type = 7;  // 5%
                else if (move_rand < 55) move_type = 8;  // 3%
                else if (move_rand < 57) move_type = 9;  // 2%
                else if (move_rand < 59) move_type = 10; // 2%
                else if (move_rand < 60) move_type = 11; // 1%
                else if (move_rand < 62) move_type = 12; // 2%
                else if (move_rand < 70) move_type = 13; // 8% (was 3%)
                else if (move_rand < 75) move_type = 14; // 5% (was 1%)
                else move_type = 15;                      // 25% (was 15%)
            } else {
                // Standard distribution for early and late game
                if (move_rand < 14) move_type = 0;
                else if (move_rand < 28) move_type = 1;
                else if (move_rand < 42) move_type = 2;
                else if (move_rand < 50) move_type = 3;
                else if (move_rand < 54) move_type = 4;
                else if (move_rand < 59) move_type = 5;
                else if (move_rand < 64) move_type = 6;
                else if (move_rand < 70) move_type = 7;
                else if (move_rand < 74) move_type = 8;
                else if (move_rand < 77) move_type = 9;
                else if (move_rand < 80) move_type = 10;
                else if (move_rand < 82) move_type = 11;
                else if (move_rand < 85) move_type = 12;
                else if (move_rand < 88) move_type = 13;
                else if (move_rand < 91) move_type = 14;
                else move_type = 15;
            }
            
            if (move_type == 0) {
                int t = rng() % T;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                int choice = rng() % (L * N + 1);
                int new_i, new_j;
                if (choice == L * N) { new_i = -1; new_j = -1; }
                else { new_i = choice / N; new_j = choice % N; }
                if (new_i == orig_i && new_j == orig_j) { record_attempt(false); continue; }
                if (new_i >= 0) {
                    double cost = (double)C[new_i][new_j] * (double)(prefix[t].P[new_i][new_j] + 1);
                    if (cost > prefix[t].apples) { record_attempt(false); continue; }
                }
                current_actions[t] = {new_i, new_j};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 1) {
                int t1 = rng() % T, t2 = rng() % T;
                if (t1 == t2) { record_attempt(false); continue; }
                if (t1 > t2) swap(t1, t2);
                if (current_actions[t1] == current_actions[t2]) { record_attempt(false); continue; }
                swap(current_actions[t1], current_actions[t2]);
                auto [valid, new_score] = eval_from(prefix[t1], current_actions, t1);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t1; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        swap(current_actions[t1], current_actions[t2]);
                        record_attempt(false);
                    }
                } else {
                    swap(current_actions[t1], current_actions[t2]);
                    record_attempt(false);
                }
            } else if (move_type == 2) {
                int t1 = rng() % T, t2 = rng() % T;
                if (t1 == t2) { record_attempt(false); continue; }
                auto saved = current_actions[t1];
                if (t1 < t2) {
                    for (int t = t1; t < t2; t++)
                        current_actions[t] = current_actions[t+1];
                    current_actions[t2] = saved;
                } else {
                    for (int t = t1; t > t2; t--)
                        current_actions[t] = current_actions[t-1];
                    current_actions[t2] = saved;
                }
                int start = min(t1, t2);
                auto [valid, new_score] = eval_from(prefix[start], current_actions, start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        if (t1 < t2) {
                            auto tmp = current_actions[t2];
                            for (int t = t2; t > t1; t--)
                                current_actions[t] = current_actions[t-1];
                            current_actions[t1] = tmp;
                        } else {
                            auto tmp = current_actions[t2];
                            for (int t = t2; t < t1; t++)
                                current_actions[t] = current_actions[t+1];
                            current_actions[t1] = tmp;
                        }
                        record_attempt(false);
                    }
                } else {
                    if (t1 < t2) {
                        auto tmp = current_actions[t2];
                        for (int t = t2; t > t1; t--)
                            current_actions[t] = current_actions[t-1];
                        current_actions[t1] = tmp;
                    } else {
                        auto tmp = current_actions[t2];
                        for (int t = t2; t < t1; t++)
                            current_actions[t] = current_actions[t+1];
                        current_actions[t1] = tmp;
                    }
                    record_attempt(false);
                }
            } else if (move_type == 3) {
                int len = seg_len;
                int max_start = T - len;
                if (max_start < 0) { record_attempt(false); continue; }
                int t_start = (int)(rng() % (max_start + 1));
                vector<pair<int,int>> saved(current_actions.begin() + t_start, current_actions.begin() + t_start + len);
                State sim = prefix[t_start];
                for (int k = 0; k < len; k++) {
                    affordable.clear();
                    affordable.push_back({-1, -1});
                    for (int i = 0; i < L; i++)
                        for (int j = 0; j < N; j++) {
                            double cost = (double)C[i][j] * (double)(sim.P[i][j] + 1);
                            if (cost <= sim.apples) affordable.push_back({i, j});
                        }
                    int idx = (int)(rng() % affordable.size());
                    current_actions[t_start + k] = affordable[idx];
                    step(sim, affordable[idx].first, affordable[idx].second);
                }
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < len; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < len; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else if (move_type == 4) {
                int max_start = T - 4;
                if (max_start < 0) { record_attempt(false); continue; }
                int t_start = (int)(rng() % (max_start + 1));
                int dj = (int)(rng() % N);
                int start_level = 1 + (int)(rng() % 3);
                int domino_len = start_level + 1;
                if (t_start + domino_len > T) domino_len = T - t_start;
                if (domino_len <= 0) { record_attempt(false); continue; }
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + domino_len);
                bool ok = true;
                State sim = prefix[t_start];
                for (int k = 0; k < domino_len; k++) {
                    int act_i = start_level - k;
                    int act_j = dj;
                    if (act_i < 0) { domino_len = k; break; }
                    double cost = (double)C[act_i][act_j] * (double)(sim.P[act_i][act_j] + 1);
                    if (cost > sim.apples + 1e-9) { ok = false; break; }
                    current_actions[t_start + k] = {act_i, act_j};
                    step(sim, act_i, act_j);
                }
                if (!ok) {
                    for (int k = 0; k < domino_len; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                    continue;
                }
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < domino_len; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < domino_len; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else if (move_type == 5) {
                int dj = 0;
                {
                    double total_A = 0;
                    for (int j2 = 0; j2 < N; j2++) total_A += (double)A[j2];
                    double r = (double)(rng()) / 4294967295.0 * total_A;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += (double)A[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                int t = rng() % T;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                int chosen_i = -1;
                for (int i2 = L - 1; i2 >= 1; i2--) {
                    double cost = (double)C[i2][dj] * (double)(prefix[t].P[i2][dj] + 1);
                    if (cost <= prefix[t].apples) { chosen_i = i2; break; }
                }
                if (chosen_i < 0) {
                    double cost = (double)C[0][dj] * (double)(prefix[t].P[0][dj] + 1);
                    if (cost <= prefix[t].apples) { chosen_i = 0; }
                }
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && dj == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, dj};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 6) {
                int dj = 0;
                {
                    double total_A = 0;
                    for (int j2 = 0; j2 < N; j2++) total_A += (double)A[j2];
                    double r = (double)(rng()) / 4294967295.0 * total_A;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += (double)A[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                int t = rng() % T;
                int R = T - t;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                int chosen_i = compute_cascade_sensitivity(prefix[t], dj, R);
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && dj == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, dj};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 7) {
                int dj = 0;
                {
                    double total_A = 0;
                    for (int j2 = 0; j2 < N; j2++) total_A += (double)A[j2];
                    double r = (double)(rng()) / 4294967295.0 * total_A;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += (double)A[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                int t_start = (int)(rng() % T);
                int R = T - t_start;
                if (R < 4) { record_attempt(false); continue; }
                double max_cap = 0;
                for (int i2 = 0; i2 < 4; i2++)
                    for (int j2 = 0; j2 < 10; j2++)
                        max_cap = max(max_cap, prefix[t_start].B[i2][j2] * (double)prefix[t_start].P[i2][j2]);
                double avg_mat = 0;
                for (int i2 = 0; i2 < 4; i2++)
                    for (int j2 = 0; j2 < 10; j2++)
                        avg_mat += (max_cap > 1e-18 ? (prefix[t_start].B[i2][j2] * (double)prefix[t_start].P[i2][j2] / max_cap) : 0);
                avg_mat /= 40.0;
                double turn_factor = 1.0 / (1.0 + exp(-(double)(R - 150) / 50.0));
                int bucket_len = (int)(4.0 + turn_factor * 14.0 + avg_mat * 6.0 + (double)(rng() % 5));
                bucket_len = max(4, min(bucket_len, R));
                if (bucket_len < 4) { record_attempt(false); continue; }
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + bucket_len);
                State sim = prefix[t_start];
                double high_level_growth = 0.01;
                {
                    double p3_sum = 0;
                    int p3_count = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        if (sim.P[3][j2] > 0) { p3_sum += sim.P[3][j2]; p3_count++; }
                    }
                    if (p3_count > 0) {
                        double avg_p3 = p3_sum / p3_count;
                        double b3_sum = 0;
                        for (int j2 = 0; j2 < N; j2++) b3_sum += sim.B[3][j2];
                        double avg_b3 = b3_sum / N;
                        high_level_growth = max(0.01, min(2.0, avg_p3 * max(avg_b3, 1.0) / 100.0));
                    } else {
                        double current_rate = sim.production_rate();
                        double base_B = 0;
                        for (int j2 = 0; j2 < N; j2++) base_B += (double)A[j2] * sim.B[0][j2];
                        if (base_B > 0 && current_rate > base_B) {
                            high_level_growth = log(current_rate / base_B) / max(1.0, (double)R * 0.1);
                            high_level_growth = max(0.01, min(2.0, high_level_growth));
                        }
                    }
                }
                double alpha = max(0.5, high_level_growth * 3.0);
                for (int k = 0; k < bucket_len; k++) {
                    int R_local = R - k;
                    if (R_local <= 0) {
                        current_actions[t_start + k] = {-1, -1};
                        step(sim, -1, -1);
                        continue;
                    }
                    double phase = (double)k / (double)bucket_len;
                    double hyperbolic = (double)R_local / ((double)R_local + alpha * (double)(L - 1));
                    double adjusted_phase = phase * (1.0 + (1.0 - hyperbolic) * 0.5);
                    adjusted_phase = min(1.0, adjusted_phase);
                    int target_level = (int)((1.0 - adjusted_phase) * (double)L);
                    if (target_level >= L) target_level = L - 1;
                    if (target_level < 0) target_level = 0;
                    if (R_local < target_level) target_level = R_local - 1;
                    if (target_level < 0) target_level = 0;
                    int act_i = -1;
                    for (int i2 = target_level; i2 >= 0; i2--) {
                        double cost = (double)C[i2][dj] * (double)(sim.P[i2][dj] + 1);
                        if (cost <= sim.apples) { act_i = i2; break; }
                    }
                    if (act_i < 0) {
                        current_actions[t_start + k] = {-1, -1};
                    } else {
                        current_actions[t_start + k] = {act_i, dj};
                    }
                    step(sim, act_i, dj);
                }
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < bucket_len; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < bucket_len; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else if (move_type == 8) {
                int t = rng() % T;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                compute_level_priority(prefix[t]);
                int chosen_i = -1, chosen_j = -1;
                double best_lp_score = -1e18;
                for (int i2 = 0; i2 < 4; i2++) {
                    for (int j2 = 0; j2 < 10; j2++) {
                        double cost = (double)C[i2][j2] * (double)(prefix[t].P[i2][j2] + 1);
                        if (cost > prefix[t].apples) continue;
                        double marginal = (double)A[j2] * prefix[t].B[i2][j2] * g_level_priority[i2][j2];
                        double score = log2(max(marginal, 1e-18)) - log2(max(cost, 1.0));
                        if (score > best_lp_score) {
                            best_lp_score = score;
                            chosen_i = i2;
                            chosen_j = j2;
                        }
                    }
                }
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && chosen_j == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, chosen_j};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 9) {
                int dj = 0;
                {
                    double total_A = 0;
                    for (int j2 = 0; j2 < N; j2++) total_A += (double)A[j2];
                    double r = (double)(rng()) / 4294967295.0 * total_A;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += (double)A[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                int t_start = (int)(rng() % T);
                int R = T - t_start;
                if (R < 4) { record_attempt(false); continue; }
                int window_size = 4 + (int)(rng() % 8);
                window_size = min(window_size, R);
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + window_size);
                State sim = prefix[t_start];
                for (int k = 0; k < window_size; k++) {
                    int R_k = R - k;
                    set_phase_weights(t_start + k, T);
                    set_discount_rate_state(sim, t_start + k, T);
                    int act_i = -1;
                    double best_cb = -1e18;
                    for (int i2 = 0; i2 < L; i2++) {
                        double cost = (double)C[i2][dj] * (double)(sim.P[i2][dj] + 1);
                        if (cost > sim.apples) continue;
                        double cb = compute_cascading_benefit(sim, i2, dj, R_k);
                        if (cb > best_cb) {
                            best_cb = cb;
                            act_i = i2;
                        }
                    }
                    current_actions[t_start + k] = {act_i, dj};
                    step(sim, act_i, dj);
                }
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < window_size; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < window_size; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else if (move_type == 10) {
                int t = rng() % T;
                int R = T - t;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                double time_weight = max(0.1, min(1.0, (double)R / (double)T));
                int chosen_i = -1, chosen_j = -1;
                double best_comp = -1e18;
                for (int i2 = 0; i2 < L; i2++) {
                    for (int j2 = 0; j2 < N; j2++) {
                        double cost = (double)C[i2][j2] * (double)(prefix[t].P[i2][j2] + 1);
                        if (cost > prefix[t].apples) continue;
                        double marginal = (double)A[j2] * prefix[t].B[i2][j2];
                        for (int k = 0; k < i2; k++)
                            marginal *= max((double)prefix[t].P[k][j2], 1.0);
                        double level_weight = 1.0 + (double)i2 * time_weight * 0.5;
                        level_weight *= g_synergy_level_weight[i2];
                        double synergy = 1.0;
                        for (int k = i2 + 1; k < L; k++) {
                            double pk = (double)prefix[t].P[k][j2];
                            if (pk > 0) synergy *= (1.0 + min(pk, 50.0) * 0.01);
                        }
                        synergy *= g_synergy_balance_weight;
                        double comp_score = log2(max(marginal * level_weight * synergy, 1e-18)) - log2(max(cost, 1.0));
                        if (comp_score > best_comp) {
                            best_comp = comp_score;
                            chosen_i = i2;
                            chosen_j = j2;
                        }
                    }
                }
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && chosen_j == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, chosen_j};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 11) {
                int t_start = (int)(rng() % T);
                int R = T - t_start;
                if (R < 4) { record_attempt(false); continue; }
                int window_size = 4 + (int)(rng() % 8);
                window_size = min(window_size, R);
                
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + window_size);
                State sim = prefix[t_start];
                
                for (int k = 0; k < window_size; k++) {
                    int R_k = R - k;
                    if (R_k <= 0) {
                        current_actions[t_start + k] = {-1, -1};
                        step(sim, -1, -1);
                        continue;
                    }
                    
                    update_synergy_weights(sim);
                    
                    double max_synergy = 0;
                    for (int i2 = 0; i2 < L; i2++) {
                        for (int j2 = 0; j2 < N; j2++) {
                            double raw = compute_synergy_raw(sim, i2, j2, R_k);
                            if (raw > max_synergy) max_synergy = raw;
                        }
                    }
                    
                    double threshold = max_synergy * 0.3;
                    
                    int best_i = -1, best_j = -1;
                    double best_ratio = -1e18;
                    for (int i2 = 0; i2 < L; i2++) {
                        for (int j2 = 0; j2 < N; j2++) {
                            double raw = compute_synergy_raw(sim, i2, j2, R_k);
                            if (raw < threshold || raw <= 0) continue;
                            double cost = (double)C[i2][j2] * (double)(sim.P[i2][j2] + 1);
                            if (cost > sim.apples) continue;
                            double ratio = log2(max(raw, 1e-18)) - log2(max(cost, 1.0));
                            if (ratio > best_ratio) {
                                best_ratio = ratio;
                                best_i = i2;
                                best_j = j2;
                            }
                        }
                    }
                    
                    if (best_i < 0) {
                        current_actions[t_start + k] = {-1, -1};
                    } else {
                        current_actions[t_start + k] = {best_i, best_j};
                    }
                    step(sim, best_i, best_j);
                }
                
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < window_size; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < window_size; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else if (move_type == 12) {
                int t = rng() % T;
                int R = T - t;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                
                auto util = compute_level_utilization(prefix[t]);
                
                int bottleneck_level = 0;
                double min_util = util[0];
                for (int i2 = 1; i2 < 4; i2++) {
                    if (util[i2] < min_util) {
                        min_util = util[i2];
                        bottleneck_level = i2;
                    }
                }
                
                double max_util = *max_element(util.begin(), util.end());
                double util_imbalance = (max_util > 1e-18) ? (max_util - min_util) / max_util : 0.0;
                
                auto level_sums = compute_level_path_sums(prefix[t], R);
                double max_sum = *max_element(level_sums.begin(), level_sums.end());
                
                int chosen_i = -1, chosen_j = -1;
                double best_combined = -1e18;
                
                for (int i2 = 0; i2 < L; i2++) {
                    for (int j2 = 0; j2 < N; j2++) {
                        double cost = (double)C[i2][j2] * (double)(prefix[t].P[i2][j2] + 1);
                        if (cost > prefix[t].apples) continue;
                        
                        double impact = compute_cascading_impact(prefix[t], i2, j2, R);
                        if (impact <= 0) continue;
                        
                        double benefit_score = log2(max(impact, 1.0)) - log2(max(cost, 1.0));
                        
                        double util_priority = 0.0;
                        if (max_util > 1e-18) {
                            util_priority = 1.0 - util[i2] / max(max_util, 1e-18);
                        }
                        
                        double bottleneck_bonus = (i2 == bottleneck_level) ? util_imbalance : 0.0;
                        
                        double path_imbalance_score = 0.0;
                        if (max_sum > 1e-18) {
                            path_imbalance_score = 1.0 - level_sums[i2] / max_sum;
                        }
                        
                        double combined = benefit_score 
                            + 0.5 * util_priority 
                            + 0.4 * bottleneck_bonus
                            + 0.2 * path_imbalance_score;
                        
                        if (combined > best_combined) {
                            best_combined = combined;
                            chosen_i = i2;
                            chosen_j = j2;
                        }
                    }
                }
                
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && chosen_j == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, chosen_j};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 13) {
                int dj = 0;
                {
                    double total_potential = 0;
                    array<double, 10> potential = {};
                    for (int j2 = 0; j2 < N; j2++) {
                        int sample_t = rng() % T;
                        double pp = 0;
                        for (int i2 = 1; i2 < L; i2++) {
                            pp += prefix[sample_t].B[i2][j2] * (double)prefix[sample_t].P[i2][j2];
                        }
                        potential[j2] = pp + 1.0;
                        total_potential += potential[j2];
                    }
                    double r = (double)(rng()) / 4294967295.0 * total_potential;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += potential[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                
                int t = rng() % T;
                int R = T - t;
                int orig_i = current_actions[t].first, orig_j = current_actions[t].second;
                
                int chosen_i = -1, chosen_j = -1;
                double best_cascade_score = -1e18;
                
                for (int i2 = 0; i2 < L; i2++) {
                    double cost = (double)C[i2][dj] * (double)(prefix[t].P[i2][dj] + 1);
                    if (cost > prefix[t].apples) continue;
                    
                    double score = compute_cascading_potential(prefix[t], i2, dj, R);
                    if (score > best_cascade_score) {
                        best_cascade_score = score;
                        chosen_i = i2;
                        chosen_j = dj;
                    }
                }
                
                if (chosen_i < 0) { record_attempt(false); continue; }
                if (chosen_i == orig_i && chosen_j == orig_j) { record_attempt(false); continue; }
                current_actions[t] = {chosen_i, chosen_j};
                auto [valid, new_score] = eval_from(prefix[t], current_actions, t);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    if (accept_worse(delta_log, temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2 = t; t2 < T; t2++) {
                            prefix[t2+1] = prefix[t2];
                            step(prefix[t2+1], current_actions[t2].first, current_actions[t2].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        current_actions[t] = {orig_i, orig_j};
                        record_attempt(false);
                    }
                } else {
                    current_actions[t] = {orig_i, orig_j};
                    record_attempt(false);
                }
            } else if (move_type == 14) {
                int dj = 0;
                {
                    double total_potential = 0;
                    array<double, 10> potential = {};
                    for (int j2 = 0; j2 < N; j2++) {
                        int sample_t = rng() % T;
                        double pp = 0;
                        for (int i2 = 0; i2 < L; i2++) {
                            pp += prefix[sample_t].B[i2][j2] * (double)prefix[sample_t].P[i2][j2];
                        }
                        potential[j2] = pp + 1.0;
                        total_potential += potential[j2];
                    }
                    double r = (double)(rng()) / 4294967295.0 * total_potential;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += potential[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                
                int t_start = (int)(rng() % T);
                int R = T - t_start;
                if (R < 4) { record_attempt(false); continue; }
                
                int window_size = 4 + (int)(rng() % 8);
                window_size = min(window_size, R);
                
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + window_size);
                State sim = prefix[t_start];
                
                double total_synergy = 0;
                int synergy_count = 0;
                
                for (int k = 0; k < window_size; k++) {
                    int R_k = R - k;
                    if (R_k <= 0) {
                        current_actions[t_start + k] = {-1, -1};
                        step(sim, -1, -1);
                        continue;
                    }
                    
                    int best_i = -1;
                    double best_score = -1e18;
                    
                    for (int i2 = 0; i2 < L; i2++) {
                        double cost = (double)C[i2][dj] * (double)(sim.P[i2][dj] + 1);
                        if (cost > sim.apples) continue;
                        
                        double score = compute_cascading_synergy(sim, i2, dj, R_k);
                        if (score > best_score) {
                            best_score = score;
                            best_i = i2;
                        }
                    }
                    
                    if (best_score > -1e17) {
                        total_synergy += best_score;
                        synergy_count++;
                    }
                    
                    current_actions[t_start + k] = {best_i, dj};
                    step(sim, best_i, dj);
                }
                
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    
                    double synergy_boost = 1.0;
                    if (synergy_count > 0) {
                        double avg_synergy = total_synergy / (double)synergy_count;
                        synergy_boost = 1.0 + min(max(avg_synergy * 0.02, 0.0), 0.5);
                    }
                    double adjusted_temp = temp * synergy_boost;
                    
                    if (accept_worse(delta_log, adjusted_temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < window_size; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < window_size; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            } else {
                // Move type 15: Coordinated multi-level strengthening sequence
                int dj = 0;
                {
                    double total_potential = 0;
                    array<double, 10> potential = {};
                    for (int j2 = 0; j2 < N; j2++) {
                        double pp = 0;
                        for (int i2 = 0; i2 < L; i2++) {
                            pp += prefix[0].B[i2][j2];
                        }
                        potential[j2] = pp + 1.0;
                        total_potential += potential[j2];
                    }
                    double r = (double)(rng()) / 4294967295.0 * total_potential;
                    double cum = 0;
                    for (int j2 = 0; j2 < N; j2++) {
                        cum += potential[j2];
                        if (r <= cum) { dj = j2; break; }
                    }
                }
                
                int t_start = (int)(rng() % T);
                int R = T - t_start;
                if (R < L) { record_attempt(false); continue; }
                
                int window_size = L + (int)(rng() % 4);
                window_size = min(window_size, R);
                
                vector<pair<int,int>> saved(current_actions.begin() + t_start,
                                            current_actions.begin() + t_start + window_size);
                State sim = prefix[t_start];
                
                double total_synergy = 0;
                int synergy_count = 0;
                
                for (int k = 0; k < window_size; k++) {
                    int R_k = R - k;
                    if (R_k <= 0) {
                        current_actions[t_start + k] = {-1, -1};
                        step(sim, -1, -1);
                        continue;
                    }
                    
                    double seq_progress = (double)k / (double)window_size;
                    int preferred_level = (int)((1.0 - seq_progress) * (double)(L - 1));
                    preferred_level = max(0, min(L - 1, preferred_level));
                    
                    int best_i = -1;
                    double best_score = -1e18;
                    
                    for (int i2 = 0; i2 < L; i2++) {
                        double cost = (double)C[i2][dj] * (double)(sim.P[i2][dj] + 1);
                        if (cost > sim.apples) continue;
                        
                        double score = compute_cascading_synergy(sim, i2, dj, R_k);
                        
                        if (i2 == preferred_level) {
                            score += 0.5;
                        }
                        
                        if (score > best_score) {
                            best_score = score;
                            best_i = i2;
                        }
                    }
                    
                    if (best_i < 0) {
                        current_actions[t_start + k] = {-1, -1};
                    } else {
                        current_actions[t_start + k] = {best_i, dj};
                        if (best_score > -1e17) {
                            total_synergy += best_score;
                            synergy_count++;
                        }
                    }
                    step(sim, best_i, dj);
                }
                
                auto [valid, new_score] = eval_from(prefix[t_start], current_actions, t_start);
                if (valid && new_score > 0 && current_score > 0) {
                    double delta_log = log2(new_score) - log2(current_score);
                    
                    double synergy_boost = 1.0;
                    if (synergy_count > 0) {
                        double avg_synergy = total_synergy / (double)synergy_count;
                        synergy_boost = 1.0 + min(max(avg_synergy * 0.02, 0.0), 0.5);
                    }
                    double adjusted_temp = temp * synergy_boost;
                    
                    if (accept_worse(delta_log, adjusted_temp)) {
                        current_score = new_score;
                        record_attempt(true);
                        for (int t2b = t_start; t2b < T; t2b++) {
                            prefix[t2b+1] = prefix[t2b];
                            step(prefix[t2b+1], current_actions[t2b].first, current_actions[t2b].second);
                        }
                        if (current_score > best_score) {
                            best_score = current_score;
                            best_actions = current_actions;
                        }
                    } else {
                        for (int k = 0; k < window_size; k++)
                            current_actions[t_start + k] = saved[k];
                        record_attempt(false);
                    }
                } else {
                    for (int k = 0; k < window_size; k++)
                        current_actions[t_start + k] = saved[k];
                    record_attempt(false);
                }
            }
        }
    }
    
    // Phase 10: Final hill climbing
    {
        vector<State> prefix(T + 1);
        prefix[0].init();
        for (int t = 0; t < T; t++) {
            prefix[t+1] = prefix[t];
            step(prefix[t+1], best_actions[t].first, best_actions[t].second);
        }
        bool improved = true;
        while (improved && elapsed() < 1980) {
            improved = false;
            for (int t = 0; t < T && elapsed() < 1980; t++) {
                int orig_i = best_actions[t].first, orig_j = best_actions[t].second;
                int best_repl_i = orig_i, best_repl_j = orig_j;
                double best_repl_score = best_score;
                if (orig_i >= 0) {
                    best_actions[t] = {-1, -1};
                    auto [valid, score] = eval_from(prefix[t], best_actions, t);
                    if (valid && score > best_repl_score) {
                        best_repl_score = score;
                        best_repl_i = -1; best_repl_j = -1;
                    }
                }
                for (int i = 0; i < L; i++) {
                    for (int j = 0; j < N; j++) {
                        if (i == orig_i && j == orig_j) continue;
                        double cost = (double)C[i][j] * (double)(prefix[t].P[i][j] + 1);
                        if (cost > prefix[t].apples) continue;
                        best_actions[t] = {i, j};
                        auto [valid, score] = eval_from(prefix[t], best_actions, t);
                        if (valid && score > best_repl_score) {
                            best_repl_score = score;
                            best_repl_i = i; best_repl_j = j;
                        }
                    }
                }
                best_actions[t] = {best_repl_i, best_repl_j};
                if (best_repl_i != orig_i || best_repl_j != orig_j) {
                    best_score = best_repl_score;
                    improved = true;
                    for (int t2 = t; t2 < T; t2++) {
                        prefix[t2+1] = prefix[t2];
                        step(prefix[t2+1], best_actions[t2].first, best_actions[t2].second);
                    }
                }
            }
            for (int t = 0; t < T - 1 && elapsed() < 1980; t++) {
                if (best_actions[t] == best_actions[t+1]) continue;
                swap(best_actions[t], best_actions[t+1]);
                auto [valid, score] = eval_from(prefix[t], best_actions, t);
                if (valid && score > best_score) {
                    best_score = score;
                    improved = true;
                    for (int t2 = t; t2 < T; t2++) {
                        prefix[t2+1] = prefix[t2];
                        step(prefix[t2+1], best_actions[t2].first, best_actions[t2].second);
                    }
                } else {
                    swap(best_actions[t], best_actions[t+1]);
                }
            }
        }
    }
    
    for (const auto& [i, j] : best_actions) {
        if (i >= 0) cout << i << " " << j << "\n";
        else cout << "-1\n";
    }
    
    return 0;
}