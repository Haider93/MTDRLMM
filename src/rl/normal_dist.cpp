#include "../../include/rl/normal_dist.h"
#include <math.h>
#include <armadillo>
#define e 2.71828
using namespace arma;

using namespace rl;
Distribution::~Distribution(){}
double Distribution::getMean() {
	return mean;
}
double Distribution::getDeviation() {
	return deviation;
}
double Distribution::pdf(double x) {
	double prob_x = (1 / (deviation * sqrt(2 * 3.14))) * pow(e, -1 / 2.0 * pow(((x - mean) / deviation), 2));
	return prob_x;
}
void Distribution::setMean(double m) {
	mean = m;
}
void Distribution::setDeviation(double sd) {
	deviation = sd;
}
void Distribution::setObservation(int x){
	observations = x;
}
int Distribution::getObservation(){
	return observations;
}
void Distribution::setWeight(double w) {
	weight = w;
}
double Distribution::getWeight() {
	return weight;
}

//multivariate
vec Distribution::getMeanVec() {
	return means;
}
mat Distribution::getCovMat() {
	return covariance;
}
void Distribution::setMeanVec(vec m) {
	means = m;
}
void Distribution::setCovMat(mat c) {
	covariance = c;
}
double Distribution::pdf_multi(vec X) {
	double density = 0.0;
	if(det(covariance) <= 0)
		throw std::invalid_argument("Covriance matrix is not symmetricpositive definite!!");
	if (X.size() == means.size()) {
		vec val = trans(X - means) * (inv(covariance) * (X - means));
		density = (1 / (pow((2 * 3.14), (X.size() / 2)) * sqrt(det(covariance)))) * pow(e, -0.5 * val(0));
	}
	return density;
}
uword Distribution::getDimension() {
	return dimension;
}