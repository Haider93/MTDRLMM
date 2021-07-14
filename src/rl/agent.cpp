#include "../../include/rl/agent.h"

#include <cmath>
#include <memory>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <spdlog\sinks\rotating_file_sink.h>
#include <Python.h>
#include <exception>

using namespace std;
using namespace rl;


Agent::Agent(std::unique_ptr<Policy> policy, Config &c):
    MEMORY_SIZE(c["learning"]["memory_size"].as<long>()),
    N_TILINGS(c["learning"]["n_tilings"].as<int>()),
    N_ACTIONS(c["learning"]["n_actions"].as<int>()),

    group_weights(make_tuple(1.0/3, 1.0/3, 1.0/3)),

    traces(MEMORY_SIZE, N_TILINGS, N_ACTIONS),
	eligibility_trace(N_ACTIONS),

    alpha_start(c["learning"]["alpha_start"].as<double>(0.2)),
    alpha_floor(c["learning"]["alpha_floor"].as<double>(0.001)),
    omega(c["learning"]["omega"].as<double>(1.0)),
    alpha(alpha_start),

    gamma(c["learning"]["gamma"].as<double>()),
    lambda(c["learning"]["lambda"].as<double>()),

    gen(c["debug"]["random_seed"].as<unsigned>(random_device{}())),
	//weight_incr(c["learning"]["weight_incr"].as<double>()),
    unif_dist(0.0, 1.0),

    policy(std::move(policy))
{
	//d = new Distribution(0.0, 1.0, 0, 0.0);
	//multivariate gauss
	vec m;
	m << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0;
	mat cov = eye<mat>(8, 8);
	d = new Distribution(8, m, cov, 0, 0.0);
	action_distributions.resize(N_ACTIONS);
	for (int i = 0; i < (N_ACTIONS); i++) {
		action_distributions[i] = *d;
		action_distributions[i].setWeight(1.0 / N_ACTIONS);
	}

	//Initialize python interpreter
	Py_Initialize();

	//PyEval_InitThreads();

	// Convert the file name to a Python string.

	string mod_name = c["learning"]["python_module"].as<string>("dqn_mlp_shared_nn");
	pName = PyUnicode_FromString(mod_name.c_str());

	// Import the file as a Python module.

	pModule = PyImport_Import(pName);

	//action_seq.open("test_log.txt");

	//try{
	//	if (pModule) {
	//		train_model = PyObject_GetAttrString(pModule, "train_model");
	//		call_model = PyObject_GetAttrString(pModule, "get_q_values");
	//	}
	//	else {
	//		//raise an exception
	//		cout << "Python script not loaded." << endl;
	//	}
	//}
	//catch(std::exception& e){}
	
	//initialization ends

    theta = new double[MEMORY_SIZE];

    if (c["learning"]["random_init"].as<bool>(false))
        generate(&theta[0], &theta[MEMORY_SIZE],
                 [this]() { return 2.0*unif_dist(gen)-1.0; });
    else
        std::fill(&theta[0], &theta[MEMORY_SIZE], 0.0);

    if (c["learning"]["group_weights"]) {
        get<0>(group_weights) = c["learning"]["group_weights"][0].as<double>();
        get<1>(group_weights) = c["learning"]["group_weights"][1].as<double>();

        double alt = 1.0 - (get<0>(group_weights) + get<1>(group_weights));
        get<2>(group_weights) =
            c["learning"]["group_weights"][2].as<double>(alt);
    }

    // Register loggers for TD-error:
    if (c["logging"] and c["logging"]["log_learning"].as<bool>(true)) {
        try {
            auto log = spdlog::rotating_logger_mt("model_log",
                                                  c["output_dir"].as<string>() + "model_log",
                                                  c["logging"]["max_size"].as<size_t>(), 1);
        } catch (spdlog::spdlog_ex& e) {}
    }
}

Agent::~Agent()
{
    delete [] theta;
	//destroy python interpreter
	//Py_XDECREF(pName);
	//Py_XDECREF(pModule);
	//Py_Finalize();
	//action_seq.close();
}

unsigned int Agent::action(State& s)
{
    std::vector<double> qs(N_ACTIONS, 0.0);
	// for univariate
	//std::vector<float> state_vars = s.getNormalizedVars();
	//float state_normalized_val = s.getNormalizedMeanVal();
	//std::vector<float> subset_state_vars(2, 0.0);

	std::vector<double> prob_vals(N_ACTIONS, 0.0);

	//for multivariate
	vec v_mvd = s.get_vars_for_mvd();
	//vec v_means = trans(s.get_mean_vector_state_values());//returns rowvec type transpose to convert it to vec type (colvec)
	//mat v_covariances = s.get_covariances_state_values();
	//v_means.print("mean vector from data:");
	//v_covariances.print("Matrix from data:");
	/*if (det(v_covariances) <= 0){
		v_covariances = get_PD(v_covariances);
	}*/
		

	//multivariate gauss
	for (int a = 0; a < (N_ACTIONS); a++) {
		//upadate means and covariances of action distributions from new state obervation v_mvd
		/*if (v_covariances.is_sympd() == 1) {
			action_distributions[a].setMeanVec(v_means);
			action_distributions[a].setCovMat(v_covariances);
		}*/
		//calculate distance of new state observation v_mvd from mean vector of each distribution
		//prob_vals[a] = (action_distributions[a].getWeight() * action_distributions[a].pdf_multi(v_mvd));
	}

	//univariate
	/*for (int s_ind = 0; s_ind < state_vars.size(); s_ind++) {
		for (int a = 0; a < (N_ACTIONS); a++) {
			prob_vals[a] += (action_distributions[a].getWeight() * action_distributions[a].pdf(state_vars[s_ind]));
		}
	}*/
	
	for (int a = 0; a < N_ACTIONS; a++) {
		//prob_vals[a] = action_distributions[a].pdf(state_normalized_val);
		//prob_vals[a] += action_distributions[a].pdf(state_normalized_val);
		
		//qs[a] = getQ(s, a);
	}
	//auto max_prob = std::max_element(std::begin(prob_vals), std::end(prob_vals));
	//int max_prob_index = std::distance(prob_vals.begin(), max_prob);
	//auto max_q = std::max_element(std::begin(qs), std::end(qs));

	//call python model and get q values of actions in qs

	try{
		if (pModule) {
			call_model = PyObject_GetAttrString(pModule, "get_q_values");
			if (call_model && PyCallable_Check(call_model)) {
				vector<float> st = s.getStateVars();
				if (st.size() > 0) {
					call_model_args = PyTuple_New(1);
					call_model_state = PyList_New(st.size());
					for (size_t ii = 0; ii < st.size(); ii++) {
						PyList_SET_ITEM(call_model_state, ii, PyFloat_FromDouble(st.at(ii)));
					}
					PyTuple_SetItem(call_model_args, 0, call_model_state);
					//PyEval_AcquireLock();
					PyObject* pResult = PyObject_CallObject(call_model, call_model_args);
					//PyEval_ReleaseLock();
					if (pResult != NULL)
					{
						//cout << "Call model." << endl;
						Py_ssize_t n = PyList_Size(pResult);
						PyObject* item = NULL;
						for (int j = 0; j < n; j++)
						{
							item = PyList_GetItem(pResult, j);
							//cout << PyFloat_AsDouble(item) << endl;
							qs[j] = PyFloat_AsDouble(item);
						}
						Py_XDECREF(item);
						//cout << "----------------------" << endl;
					}
					//Py_XDECREF(call_model_args);
					//Py_XDECREF(call_model_state);
					Py_XDECREF(pResult);
				}
			}
			//Py_XDECREF(call_model);
		}
			
		
	}
	catch(std::exception& e){}

	unsigned int choosen_action_prob = policy->Sample(qs);
	//unsigned int choosen_action_prob = policy->Sample(prob_vals);
	//cout << choosen_action_prob << endl;
	//action_seq << to_string(choosen_action_prob) << endl;
	//unsigned int choosen_action_simple = policy->Sample(qs);
	//if (choosen_action_simple == choosen_action_prob)
		//return choosen_action_simple;
	//else
	return choosen_action_prob;
}

void Agent::GoGreedy()
{
    this->policy = std::unique_ptr<Policy>(new Greedy(N_ACTIONS));
}

void Agent::SetPolicy(std::unique_ptr<Policy> policy)
{
    this->policy = std::move(policy);
}

void Agent::HandleTransition(State& from_state, int action,
                             double reward, State& to_state)
{
    UpdateTraces(from_state, action);

    double delta = UpdateWeights(from_state, action, reward, to_state);

    _agg_delta += abs(delta);

    if (++_update_counter % 1000 == 0) {
        spdlog::get("model_log")->info(_agg_delta / 1000);

        _agg_delta = 0.0;
        _update_counter = 0;
    }
}

void Agent::HandleTerminal(int episode)
{
    traces.decay(0.0);
	eligibility_trace.decay(0.0);
    alpha = max(alpha_floor, alpha_start * pow(omega, (double) episode));

    this->policy->HandleTerminal(episode);
}

void Agent::UpdateTraces(State& from_state, int action)
{
    traces.decay(gamma*lambda);
    traces.update(from_state, action);//set value of traces
	eligibility_trace.decay(gamma*lambda);
	eligibility_trace.update(from_state, action);
}

double Agent::getQ(State& state, int action)
{
    auto features = state.getFeatures(action);
    double Q = 0.0;
	// for unuvariate
	double state_val = state.getNormalizedMeanVal();
	//double prob_q_val = action_distributions[action].pdf(state_val);
	vector<float> state_vars = state.getNormalizedVars();
	double prob_q_val = 0.0;
	/*for (int s_ind = 0; s_ind < state_vars.size(); s_ind++) {
		prob_q_val += (action_distributions[action].getWeight() * action_distributions[action].pdf(state_vars[s_ind]));
	}*/
	//prob_q_val += action_distributions[action].pdf(state_val);

	// for multivariate
	//vec v_mvd = state.get_vars_for_mvd();
	//prob_q_val = action_distributions[action].getWeight() * action_distributions[action].pdf_multi(v_mvd);


	//for agent vars
    double w = get<0>(group_weights);
    for (int i = 0; i < N_TILINGS; i++)
        Q += w*theta[features[i]];

	//marker vars
    w = get<1>(group_weights);
    for (int i = N_TILINGS; i < 2*N_TILINGS; i++)
        Q += w*theta[features[i]];

	//combined state vars
    w = get<2>(group_weights);
    for (int i = N_TILINGS; i < 3*N_TILINGS; i++)
        Q += w*theta[features[i]];


    return Q;
	//return prob_q_val;
}

//univariate
void Agent::update_action_dist(double update, int action) {
	double scaled_update = 0.0;
	//update weights of distributions
	/*for (int ii = 0; ii < N_ACTIONS; ii++) {
		if (ii == action) {
			action_distributions[ii].setWeight(action_distributions[ii].getWeight() + weight_incr);
		}
		action_distributions[ii].setWeight(action_distributions[ii].getWeight() - (weight_incr / (N_ACTIONS - 1)));
	}*/
	for (auto it = eligibility_trace.begin(); it != eligibility_trace.end(); it++)
		scaled_update += update * eligibility_trace.get(*it);
	update = scaled_update;
	double prev_num_vals = action_distributions[action].getObservation();
	action_distributions[action].setObservation(action_distributions[action].getObservation() + 1);
	double old_m = action_distributions[action].getMean();
	double new_m = old_m + ((update - old_m) / action_distributions[action].getObservation());
	double old_var = pow(action_distributions[action].getDeviation(), 2.0);
	double new_var = (prev_num_vals / (prev_num_vals + 1)) * (old_var + (pow((update - old_m), 2.0) / (prev_num_vals + 1)));
	if (new_var < 0)
		throw std::invalid_argument("Invalid value for variance!!");
	else if (new_var == 0)
		new_var = old_var;
	action_distributions[action].setMean(new_m);
	action_distributions[action].setDeviation(sqrt(new_var));
}
////multivariate
void Agent::update_action_multi(double update, int action) {
	double scaled_update = 0.0;

	/*for (int ii = 0; ii < N_ACTIONS; ii++) {
		if (ii == action) {
			action_distributions[ii].setWeight(action_distributions[ii].getWeight() + weight_incr);
		}
		action_distributions[ii].setWeight(action_distributions[ii].getWeight() - (weight_incr / (N_ACTIONS - 1)));
	}*/

	double obs = action_distributions[action].getObservation();
	action_distributions[action].setObservation(action_distributions[action].getObservation() + 1);
	/*for (auto it = eligibility_trace.begin(); it != eligibility_trace.end(); it++)
		scaled_update += update * eligibility_trace.get(*it);*/
	update = scaled_update;
	update = update / action_distributions[action].getDimension();
	vec old_means = action_distributions[action].getMeanVec();
	mat old_cov_matr = action_distributions[action].getCovMat();
	//old_means.print("Old Means::");
	//old_cov_matr.print("Old covariance matrix::");
	vec new_means = old_means + ((update - old_means) / action_distributions[action].getObservation());
	vec xx = update - old_means;
	double exp_val = pow(xx(0)+xx(1)+xx(2)+xx(3)+xx(4)+xx(5)+xx(6)+xx(7), 2.0);
	//double exp_val = pow(xx(0), 2.0);
	mat new_cov_matr(old_cov_matr.n_rows, old_cov_matr.n_cols);

	//if using diagonal cov matrix in the problem
	new_cov_matr.fill(0.0);
	for (int jj = 0; jj < action_distributions[action].getDimension(); jj++) {
		new_cov_matr(jj, jj) = (obs / (obs + 1)) * (old_cov_matr(jj, jj) + (exp_val / (obs + 1)));
	}

	//add update to diagonal elements of cov matrix, when using non-diagonal cov matrix
	/*new_cov_matr = old_cov_matr;
	for (int jj = 0; jj < action_distributions[action].getDimension(); jj++) {
		new_cov_matr(jj, jj) = (obs / (obs + 1)) * (old_cov_matr(jj, jj) + (exp_val / (obs + 1)));
	}*/

	//new_means.print("mean vector after td update:");
	//new_cov_matr.print("Cov matrix after td update::");

	/*if (det(new_cov_matr) <= 0)
		new_cov_matr = get_PD(new_cov_matr);
	action_distributions[action].setMeanVec(new_means);
	if(det(new_cov_matr) > 0)
		action_distributions[action].setCovMat(new_cov_matr);*/

	if (det(new_cov_matr) <= 0)
		new_cov_matr = old_cov_matr;
	action_distributions[action].setMeanVec(new_means);
	action_distributions[action].setCovMat(new_cov_matr);

}
//
//mat Agent::get_PD(mat a) {
//	double epsd = std::numeric_limits<double>::epsilon();
//	mat b = (a + trans(a)) / 2;
//	mat U;
//	vec s;
//	mat V;
//	svd(U, s, V, b);
//	mat H = trans(V) * (diagmat(s) * V);
//	mat a2 = (b + H) / 2;
//	mat a3 = (a2 + trans(a2)) / 2;
//	if (a3.is_sympd() == 1)
//		return a3;
//	mat I = eye<mat>(a.n_rows, a.n_cols);
//	int k = 1;
//	vec eigval;
//	mat closest_mat = eye<mat>(a.n_rows, a.n_cols);
//	//double spacing = eps(norm(a));
//	double spacing = std::nextafter(double(norm(a, "fro")), epsd) - double(norm(a, "fro"));
//	if (spacing < 0)
//		spacing *= -1;
//	while (a3.is_sympd() == 0) {
//		try {
//			eigval = eig_sym(a3);
//			double mineig = min(eigval);
//			a3 += I * (mineig * pow(k, 2.0) + spacing);
//			if (det(a3) > 0){
//				closest_mat = a3;
//				//return closest_mat;
//			}
//			k += 1;
//		}
//		catch (exception e) {
//			//return closest_mat;
//		}
//	}
//	return a3;
//}

void Agent::updateQ(double update)
{
    double scaled_update = update / N_TILINGS;
    for (auto it = traces.begin(); it != traces.end(); it++)
        theta[*it] += scaled_update * traces.get(*it);
}

int Agent::argmaxQ(State& state)
{
    int index = 0;
    int n_ties = 1;
    double currMaxQ = getQ(state, index);

    for (int a = 1; a < N_ACTIONS; a++) {
        double val = getQ(state, a);

        if (val >= currMaxQ) {
            if (val > currMaxQ) {
                currMaxQ = val;
                index = a;
            } else {
                n_ties++;

                if (0 == rand() % n_ties) {
                    currMaxQ = val;
                    index = a;
                }
            }
        }
    }

    return index;
}

double Agent::maxQ(State& state)
{
    return getQ(state, argmaxQ(state));
}

void Agent::write_theta(string path, string filename)
{
    //ofstream file(filename.c_str(), std::ofstream::binary | std::ofstream::out);
	//filename = filename.c_str();
	std::ofstream file(filename.c_str(), std::ios::app);
	//std::ofstream os(filename.c_str(), std::ios::app);
	
    file.write((char *) theta, MEMORY_SIZE * sizeof(double));
	//file.write(reinterpret_cast<const char*>(theta), std::streamsize(MEMORY_SIZE * sizeof(double)));
	/*for (int i = 0; i < MEMORY_SIZE; i++)
		cout << theta[i] << " ";*/
	//file << *theta;
    file.close();

	//write parameters of distributions
	//ofstream file1(path+"dist_parms.txt", std::ios::binary | std::ios::out);
	//for (int ii = 0; ii < N_ACTIONS; ii++)
	//{
	//	//cout<< action_distributions[ii].getMean() << " " << action_distributions[ii].getDeviation() << endl;
	//	file1 << action_distributions[ii].getMean() << " " << action_distributions[ii].getDeviation() << endl;
	//}
	//file1.close();
}

double* Agent::get_theta(string filename)
{
	//ifstream file(filename.c_str(), std::ifstream::binary | std::ifstream::in);
	ifstream file(filename.c_str(), std::ios::binary | std::ios::in);
	double *t = NULL;
	if(file.is_open()){
		//file.seekg(0, file.end);
		//long content_length = file.tellg();
		//file.seekg(0);
		//char* buffer = new char[MEMORY_SIZE * sizeof(double)];
		//file.read(buffer, content_length * sizeof(double));
		//file.read(buffer, MEMORY_SIZE * sizeof(double));
		//*theta = atof(buffer);
		//*theta = stod(buffer);
		file.read(reinterpret_cast<char*>(theta), std::streamsize(MEMORY_SIZE * sizeof(double)));
	}
	
	file.close();
	return t;
}
// ---------------

DoubleAgent::DoubleAgent(std::unique_ptr<Policy> policy, Config& c):
    Agent(std::move(policy), c)
{
    theta_b = new double[MEMORY_SIZE];

    if (c["learning"]["random_init"].as<bool>(false))
        generate(&theta_b[0], &theta_b[MEMORY_SIZE],
                 [this]() { return 2.0*unif_dist(gen)-1.0; });
    else
        std::fill(&theta_b[0], &theta_b[MEMORY_SIZE], 0.0);
}

DoubleAgent::~DoubleAgent()
{
    delete [] theta_b;
}

unsigned int DoubleAgent::action(State& s)
{
    std::vector<double> qs(N_ACTIONS, 0.0);
    for (int a = 0; a < N_ACTIONS; a++)
        qs[a] = (getQ(s, a) + getQb(s, a)) / 2.0f;

    return policy->Sample(qs);
}

double DoubleAgent::getQb(State& state, int action)
{
    auto features = state.getFeatures(action);

    double Q = 0.0;

    double w = get<0>(group_weights);
    for (int i = 0; i < N_TILINGS; i++)
        Q += w*theta[features[i]];

    w = get<1>(group_weights);
    for (int i = N_TILINGS; i < 2*N_TILINGS; i++)
        Q += w*theta[features[i]];

    w = get<2>(group_weights);
    for (int i = N_TILINGS; i < 3*N_TILINGS; i++)
        Q += w*theta[features[i]];

    return Q;
}

void DoubleAgent::updateQb(double update)
{
    double scaled_update = update / N_TILINGS;
    for (auto it = traces.begin(); it != traces.end(); it++)
        theta[*it] += scaled_update * traces.get(*it);
}

int DoubleAgent::argmaxQb(State& state)
{
    int index = 0;
    int n_ties = 1;
    double currMaxQ = getQb(state, index);

    for (int a = 1; a < N_ACTIONS; a++) {
        double val = getQb(state, a);

        if (val >= currMaxQ) {
            if (val > currMaxQ) {
                currMaxQ = val;
                index = a;
            } else {
                n_ties++;

                if (0 == rand() % n_ties) {
                    currMaxQ = val;
                    index = a;
                }
            }
        }
    }

    return index;
}

// ---------------

QLearn::QLearn(std::unique_ptr<Policy> policy, Config& c):
    Agent(std::move(policy), c)
{}

void QLearn::UpdateTraces(State& from_state, int action)
{
    int amax = argmaxQ(from_state);

    if (action != amax) traces.decay(0.0);
    else traces.decay(gamma*lambda);

    traces.update(from_state, action);
}

double QLearn::UpdateWeights(State& from_state, int action, double reward,
                             State& to_state)
{
    double Q = getQ(from_state, action),
           F_term = gamma*to_state.getPotential() - from_state.getPotential(),
           delta = reward + F_term + gamma*maxQ(to_state) - Q;

    updateQ(alpha * delta);

    return delta;
}

// ---------------

SARSA::SARSA(std::unique_ptr<Policy> policy, Config& c):
    Agent(std::move(policy), c)
{}

double SARSA::UpdateWeights(State& from_state, int action, double reward,
                            State& to_state)
{
    double Q1 = getQ(from_state, action),
          Q2 = getQ(to_state, this->action(to_state)),
          F = gamma*to_state.getPotential() - from_state.getPotential(),
          delta = reward + F + gamma*Q2 - Q1;

	//state_q_map.insert(pair<State&, double>(from_state, alpha * delta));40
	//update_action_dist(alpha * delta, action);
	//cout << (alpha * delta) << endl;

    //updateQ(alpha * delta);

	//update_action_multi(alpha * delta, action);

	//call train model python function to update NN based model

	try{
		if (pModule) {
			train_model = PyObject_GetAttrString(pModule, "train_model");
			if (train_model && PyCallable_Check(train_model)) {
				vector<float> c_state = from_state.getStateVars();
				vector<float> n_state = to_state.getStateVars();
				if (c_state.size() > 0) {
					train_model_args = PyTuple_New(4);
					curr_state = PyList_New(c_state.size());
					next_state = PyList_New(n_state.size());
					for (size_t ii = 0; ii < c_state.size(); ii++) {
						PyList_SET_ITEM(curr_state, ii, PyFloat_FromDouble(c_state.at(ii)));
						PyList_SET_ITEM(next_state, ii, PyFloat_FromDouble(n_state.at(ii)));
					}
					reward_arg = PyFloat_FromDouble(reward);
					PyTuple_SetItem(train_model_args, 0, curr_state);
					PyTuple_SetItem(train_model_args, 1, next_state);
					PyTuple_SetItem(train_model_args, 2, reward_arg);
					PyTuple_SetItem(train_model_args, 3, PyFloat_FromDouble(action));
					//PyEval_AcquireLock();
					PyObject* pResult = PyObject_CallObject(train_model, train_model_args);
					//PyEval_ReleaseLock();
					//Py_XDECREF(train_model_args);
					//Py_XDECREF(curr_state);
					//Py_XDECREF(next_state);
					//Py_XDECREF(reward_arg);
					//Py_XDECREF(pResult);
				}
			}
			//Py_XDECREF(train_model);
		}
		else {
			cout << "Failed to call train model python function." << endl;
			//raise an exception.
		}
	}
	catch(std::exception& e){}

    return delta;
	//return 1.0;
}

// ---------------

DoubleQLearn::DoubleQLearn(std::unique_ptr<Policy> policy, Config& c):
    DoubleAgent(std::move(policy), c)
{}

void DoubleQLearn::UpdateTraces(State& from_state, int action)
{
    int amax = argmaxQ(from_state);

    if (action != amax) traces.decay(0.0);
    else traces.decay(gamma*lambda);

    traces.update(from_state, action);
}

double DoubleQLearn::UpdateWeights(State& from_state, int action, double reward,
                                   State& to_state)
{
    double delta;
    double F_term = gamma*to_state.getPotential() - from_state.getPotential();
    if (unif_dist(gen) > 0.5) { // UPDATE(A)

        double Qa = getQ(from_state, action);
        delta = reward + F_term +
            gamma*getQb(to_state, argmaxQ(to_state)) - Qa;

        updateQ(alpha * delta);

    } else { // UPDATE(B)

        double Qb = getQb(from_state, action);
        delta = reward + F_term +
            gamma*getQ(to_state, argmaxQb(to_state)) - Qb;

        updateQb(alpha * delta);

    }

    return delta;
}

// ---------------

RLearn::RLearn(std::unique_ptr<Policy> policy, Config& c):
    Agent(std::move(policy), c),

    beta(c["learning"]["beta"].as<double>())
{}

void RLearn::UpdateTraces(State& from_state, int action)
{
    int amax = argmaxQ(from_state);

    if (action != amax) traces.decay(0.0);
    else traces.decay(gamma*lambda);

    traces.update(from_state, action);
}

double RLearn::UpdateWeights(State& from_state, int action, double reward,
                             State& to_state)
{
    double Q = getQ(from_state, action),
          mQ = maxQ(to_state),
          delta = reward - rho + mQ - Q,
          update = alpha*delta;

    updateQ(update);

    double nQ = Q + update;
    if (nQ - maxQ(from_state) < 1e-7)
        rho += beta * (reward - rho + mQ - nQ);

    return delta;
}

// ---------------

OnlineRLearn::OnlineRLearn(std::unique_ptr<Policy> policy, Config& c):
    Agent(std::move(policy), c),

    beta(c["learning"]["beta"].as<double>())
{}

double OnlineRLearn::UpdateWeights(State& from_state, int action, double reward,
                                   State& to_state)
{
    double Q = getQ(from_state, action),
          gQ = getQ(to_state, this->action(to_state)),
          delta = reward - rho + gQ - Q,
          update = alpha*delta;

    updateQ(update);

    double nQ = Q + update;
    if (nQ - maxQ(from_state) < 1e-7)
        rho += beta * (reward - rho + gQ - nQ);

    return delta;
}
// ---------------

DoubleRLearn::DoubleRLearn(std::unique_ptr<Policy> policy, Config& c):
    DoubleAgent(std::move(policy), c),

    beta(c["learning"]["beta"].as<double>())
{}

void DoubleRLearn::UpdateTraces(State& from_state, int action)
{
    int amax = argmaxQ(from_state);

    if (action != amax) traces.decay(0.0);
    else traces.decay(gamma*lambda);

    traces.update(from_state, action);
}

double DoubleRLearn::UpdateWeights(State& from_state, int action, double reward,
                                   State& to_state)
{
    double delta, Q, mQ;
    if (unif_dist(gen) > 0.5) { // UPDATE(A)

        Q = getQ(from_state, action);
        mQ = getQb(to_state, argmaxQ(to_state));
        delta = reward - rho + mQ - Q;

        updateQ(alpha*delta);

    } else { // UPDATE(B)

        Q = getQb(from_state, action);
        mQ = getQ(to_state, argmaxQb(to_state));
        delta = reward - rho + mQ - Q;

        updateQb(alpha*delta);

    }

    mQ = -DBL_MAX;
    for (int i = 0; i < N_ACTIONS; i++) {
        double val = (getQ(from_state, i) + getQb(from_state, i)) / 2.0;

        if (val > mQ)
            mQ = val;
    }

    double nQ = Q + alpha*delta;
    if (nQ - mQ < 1e-7)
        rho += beta * (reward - rho + mQ - nQ);

    return delta;
}
