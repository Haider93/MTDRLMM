#include "../../include/rl/eligibility_trace.h"

#include <iostream>
#include <iterator>

using namespace rl;

EligibilityTrace::EligibilityTrace(int actions):N_ACTIONS(actions),
tolerance(0.01),
n_nonzero_traces(0) {
	eligibility = new float[N_ACTIONS];
	nonzero_traces_inverse = new int[N_ACTIONS];
	for (long i = 0; i < N_ACTIONS; i++) {
		eligibility[i] = 0.0;
		nonzero_traces_inverse[i] = 0;
	}
}

EligibilityTrace::~EligibilityTrace()
{
	delete[] eligibility;
	delete[] nonzero_traces_inverse;
}

void EligibilityTrace::decay(float rate)
{
	for (int loc = n_nonzero_traces - 1; loc >= 0; loc--) {
		int f = nonzero_traces[loc];
		eligibility[f] *= rate;

		if (eligibility[f] < tolerance) clearExisting(f, loc);
	}
}

int* EligibilityTrace::begin()
{
	return nonzero_traces;
}

int* EligibilityTrace::end()
{
	return &nonzero_traces[n_nonzero_traces];
}

void EligibilityTrace::update(State& state, int action)
{
	for (int a = 0; a < N_ACTIONS; a++) {

		if (a != action)
			clear(a);
		else
			set(a, 1.0);
	}
}

float EligibilityTrace::get(int action)
{
	return eligibility[action];
}

void EligibilityTrace::set(int action, float value)
{
	if (eligibility[action] >= tolerance) eligibility[action] = value;
	else {
		while (n_nonzero_traces >= MAX_NONZERO_TRACES) increaseTolerance();

		eligibility[action] = value;
		nonzero_traces[n_nonzero_traces] = action;
		nonzero_traces_inverse[action] = n_nonzero_traces;
		n_nonzero_traces++;
	}
}
void EligibilityTrace::clear(int action)
{
	if (eligibility[action] != 0.0)
		clearExisting(action, nonzero_traces_inverse[action]);
}

void EligibilityTrace::clearExisting(int action, int loc)
{
		eligibility[action] = 0.0;
		n_nonzero_traces--;
		nonzero_traces[loc] = nonzero_traces[n_nonzero_traces];
		nonzero_traces_inverse[nonzero_traces[loc]] = loc;
}

void EligibilityTrace::increaseTolerance()
{
	tolerance *= 1.1;
	for (int loc = n_nonzero_traces - 1; loc >= 0; loc--) {
		int f = nonzero_traces[loc];

		if (eligibility[f] < tolerance) clearExisting(f, loc);
	}
}