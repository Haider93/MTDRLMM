#include "../../include/rl/state.h"

#include "../../include/rl/tiles.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <armadillo>
using namespace arma;
using namespace rl;

State::State(long memory_size, int n_actions, int n_tilings):
    MEMORY_SIZE(memory_size),
    N_TILINGS(n_tilings),
    N_ACTIONS(n_actions),
    state_vars(),
    features(n_actions, std::vector<int>(3*n_tilings, 0)),
	//features(n_actions, std::vector<int>(4 * n_tilings, 0)),
    potential(0.0)//,
	//mean_vector(8, fill::zeros),
	//covar_mat(8, 8, fill::zeros),
	//variable_values(1, 8, fill::zeros)
{}

State::State(Config &c):
    State(c["learning"]["memory_size"].as<long>(),
          c["learning"]["n_actions"].as<int>(),
          c["learning"]["n_tilings"].as<int>())
{}

void State::initialise()
{
    state_vars.clear();

    for (int a = 0; a < N_ACTIONS; a++)
        features[a].clear();
}


void State::newState(environment::Base& env)
{
    state_vars.clear();
    env.getState(state_vars);
	ofstream outfile;
	outfile.open("state_variables.csv", ios::app);
	for (int i = 0; i < state_vars.size(); i++) {
		//cout << state_vars[i]<<" ";
		outfile << state_vars[i] << ",";
	}
	outfile << endl;
	outfile.close();
	//last value of state-vars correspond to market sentiment
	//unsigned int reg = unsigned int(state_vars.at(state_vars.size() - 1));
	//env.setRegime(reg);
    populateFeatures();

    // for univariate
	//normalized_mean_val = normalize_mean(state_vars);
	//normalized_vars(state_vars);
	//state_vars = normalized_vars_subset(state_vars);

	// for multivariate
	//v_mvd = vars_multi(state_vars);

	//update means and covariances of state variables as state variable values accumulate
	//update_mean_covariances(trans(v_mvd));
	
	potential = env.getPotential();
}

vector<float>& State::getStateVars() {
	return state_vars;
}

float State::getNormalizedMeanVal() {
	return normalized_mean_val;
}
vector<float> State::getNormalizedVars() {
	return state_vars;
}
vec State::get_vars_for_mvd() {
	return v_mvd;
}
//rowvec State::get_mean_vector_state_values() {
//	return mean_vector;
//}
//
//bool State::isZero(mat m) {
//	int c = 0;
//	for (int ii = 0; ii < m.n_rows; ii++) {
//		for (int jj = 0; jj < m.n_cols; jj++) {
//			if (m(ii, jj) == 0)
//				c++;
//		}
//	}
//	if (c == (m.n_rows * m.n_cols))
//		return true;
//	else
//		return false;
//}
//
//mat State::get_covariances_state_values() {
//	//check if all elements of matrix are zero then return idemtity
//	if (isZero(covar_mat))
//		return eye<mat>(covar_mat.n_rows, covar_mat.n_cols);
//	return covar_mat;
//}
//
//
//double trend;
//void State::newState(vector<float>& _vars, double _potential)
//{
//    state_vars = _vars;
//    populateFeatures();//starat herer
//
//	if(state_vars.size() == 9)
//		trend = state_vars.at(8);
//
//    potential = _potential;
//}
//
void State::populateFeatures()
{
	
    for (int a = 0; a < N_ACTIONS; a++) {
		::tiles(&features[a][0], N_TILINGS, MEMORY_SIZE,
			&state_vars[0], 3, a);

		::tiles(&features[a][N_TILINGS], N_TILINGS, MEMORY_SIZE,
			&state_vars[3], state_vars.size() - 3, N_ACTIONS + a);

		::tiles(&features[a][2 * N_TILINGS], N_TILINGS, MEMORY_SIZE,
			&state_vars[0], state_vars.size(), (2 * N_ACTIONS) + a);

    }
}
//
std::vector<int>& State::getFeatures(int action)
{
    return features[action];
}
//
double State::getPotential()
{
    return potential;
}
//
//vector<float>& State::toVector()
//{
//    return state_vars;
//}
//
//void State::printState()
//{
//	for (size_t i = 0; i < state_vars.size(); i++)
//		cout << "[" << i << "] " << state_vars[i] << endl;
//		/*<< " \t\t-> " <<
//            floor(state_vars[i] * N_TILINGS) << endl;*/
//
//	cout << "\n Normalize and weighted mean of state vars : " << normalize_mean(state_vars) << endl;
//    cout << "\n--Press any key to continue--\n" << flush;
//    cin.sync();
//    cin.get();
//}
//
//void State::update_mean_covariances(rowvec vars) {
//	variable_values.row(variable_values.n_rows-1) = vars;
//	//variable_values.print("State variables values matrix::");
//	mean_vector = sum(variable_values, 0);// returns a rowvec
//	mean_vector = mean_vector / variable_values.n_rows;
//	for (int ii = 0; ii < covar_mat.n_rows; ii++) {
//		for (int jj = 0; jj < covar_mat.n_cols; jj++) {
//			covar_mat(ii, jj) = calc_cov(variable_values.col(ii), variable_values.col(jj));
//		}
//	}
//	variable_values.resize(variable_values.n_rows + 1, variable_values.n_cols);
//}
//double State::calc_cov(colvec a, colvec b) {
//	colvec ab = a % b;
//	return expectation_val(ab) - (expectation_val(a) * expectation_val(b));
//}
//double State::expectation_val(colvec x) {
//	return sum(x) / x.size();
//}
//
//double State::normalize_mean(vector<float> vars){
//	float average = 0.0f;
//	auto n = vars.size();
//	sort(vars.begin(), vars.end());
//	float min = vars.at(0);
//	float max = vars.at(vars.size() - 1);
//	//normalize between -1 to 1
//	float new_min = -1.0;
//	float new_max = 1.0;
//	for (int i = 0; i < vars.size(); i++) {
//		//cout << "state var " << i << " before normalization " << vars[i] << endl;
//		vars[i] = new_min + (vars[i] - min) * (new_max - new_min) / (max - min);
//		//cout << "state var " << i << " after normalization " << vars[i] << endl;
//	}
//	//cout << "--------------------------------------------------------------" << endl;
//	if (n != 0) {
//		average = accumulate(vars.begin(), vars.end(), 0.0) / n;
//		//cout << "Normalized state ::" << average << endl;
//	}
//	return average;
//}
void State::normalized_vars(vector<float>& vars) {
	auto n = vars.size();
	sort(vars.begin(), vars.end());
	float min = vars.at(0);
	float max = vars.at(vars.size() - 1);
	//normalize between -1 to 1
	float new_min = -1.0;
	float new_max = 1.0;
	for (int i = 0; i < vars.size(); i++) {
		//cout << "state var " << i << " before normalization " << vars[i] << endl;
		vars[i] = new_min + (vars[i] - min) * (new_max - new_min) / (max - min);
		//cout << "state var " << i << " after normalization " << vars[i] << endl;
	}
}
//vector<float> State::normalized_vars_subset(vector<float> vars) {
//	auto n = vars.size();
//	int subsets = 2;
//	int subset_size[2] = { 3, 5 };
//	vector<float> state_subsets;
//	auto vars_new = vars;
//	sort(vars.begin(), vars.end());
//	float min = vars.at(0);
//	float max = vars.at(vars.size() - 1);
//	vars = vars_new;
//	//normalize between -1 to 1
//	float new_min = -1.0;
//	float new_max = 1.0;
//	for (int i = 0; i < vars.size(); i++) {
//		//cout << "state var " << i << " before normalization " << vars[i] << endl;
//		vars[i] = new_min + (vars[i] - min) * (new_max - new_min) / (max - min);
//		//cout << "state var " << i << " after normalization " << vars[i] << endl;
//	}
//	state_subsets.push_back(accumulate(vars.begin(), vars.begin()+2, 0.0)/subset_size[0]);
//	state_subsets.push_back(accumulate(vars.begin()+3, vars.end(), 0.0) / subset_size[1]);
//	return state_subsets;
//}
//vec State::vars_multi(vector<float> vars) {
//	auto n = vars.size();
//	vec normalized_vars(n, fill::zeros);
//	sort(vars.begin(), vars.end());
//	float min = vars.at(0);
//	float max = vars.at(vars.size() - 1);
//	//normalize between -1 to 1
//	float new_min = -1.0;
//	float new_max = 1.0;
//	for (int i = 0; i < vars.size(); i++) {
//		//cout << "state var " << i << " before normalization " << vars[i] << endl;
//		normalized_vars(i) = new_min + (vars[i] - min) * (new_max - new_min) / (max - min);
//		//cout << "state var " << i << " after normalization " << vars[i] << endl;
//	}
//	return normalized_vars;
//}
