/* Multiclass Linear Programming Boosting.
 *
 * Copyright (C) 2008 -- Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>
 */

#ifndef BOOSTING_LPBOOSTMULTICLASS_H
#define BOOSTING_LPBOOSTMULTICLASS_H

#include <vector>
#include <string>

#include <OsiSolverInterface.hpp>

namespace Boosting {

class LPBoostMulticlassClassifier {
public:
	/* number_classes: Number of classes in the multiclass problem.
	 * nu: Nu parameter, 0 < nu <= 1.
	 * weight_sharing: if true, there is a single K-vector alpha, if false,
	 *    each class i has an individual weight vector alpha_i.
	 */
	LPBoostMulticlassClassifier(int number_classes, double nu,
		bool weight_sharing = true);
	~LPBoostMulticlassClassifier();

	/* labels: (N,1) vector of class id, 0 <= labels[i] < number_classes.
	 * interior_point: If true, we use the interior-point solver of Clp in
	 *   order to produce central solutions.  If false, we use the simplex
	 *   solver.
	 */
	void InitializeBoosting(const std::vector<int>& labels,
		bool interior_point = false,
		const std::string& solver = "clp");

	/* response: N vector of number_classes responses
	 */
	void AddMulticlassClassifier(
		const std::vector<std::vector<double> >& response);

	/* (Re-)solve the LPBoost multiclass problem
	 */
	void Update();

	/* Write current problem as MPS file.
	 */
	void WriteMPS(const std::string& mpsfile) const;

	/* Obtain the current classifier weights.
	 *
	 * If weight sharing is used, a 1-vector with one M-vector alpha is
	 * returned.  Then, the prediction function is:
	 *    f(x) = argmax_i [H(x) alpha]_i,
	 * where H(x) is the (K,M) prediction matrix for sample x.
	 *
	 * If individual weights are used, a K-vector with one M-vector each is
	 * returned.  Then, the prediction function is:
	 *    f(x) = argmax_i (H(x)_{i,.} alpha_i),
	 * where H(x)_{i,.} is the i'th row of the (K,M) prediction matrix for
	 * sample x.
	 */
	const std::vector<std::vector<double> >& ClassifierWeights() const;

	/* Rho, the margin.
	 * Gamma, the soft margin.
	 */
	double Rho() const;
	double Gamma() const;

private:
	bool initialized;	// Safety flag to ensure correct call order
	int number_classes;	// Number of classes in the multiclass problem
	double nu;	// LPBoost nu, D = 1.0 / (N * nu)
	bool weight_sharing;	// true: global alpha, false: A_{.,class}
	unsigned int norm1_rows;	// Number of |.|_1 = 1 constraints.
	unsigned int number_classifiers;	// M, number of weak learners

	double rho;	// achieved soft margin
	double gamma;	// achieved objective in max-view: rho - D \sum_i \xi_i

	std::vector<int> sample_labels;	// Multiclass labels, N samples
	// If weight sharing is used: 1 vector of M-vectors,
	// If no weight sharing is used: K vector of M-vectors.
	std::vector<std::vector<double> > classifier_weights;

	OsiSolverInterface* si;
	bool use_interior_point;
};

}

#endif

