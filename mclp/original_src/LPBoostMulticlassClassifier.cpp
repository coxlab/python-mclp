/* Multiclass Linear Programming Boosting.
 *
 * Copyright (C) 2008 -- Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>
 */

#include <iostream>
#include <algorithm>

#include <assert.h>
#include <stdlib.h>

#ifdef HAS_MOSEK
#include <mosek.h>
#endif

#include <CoinPackedMatrix.hpp>
#include <CoinPackedVector.hpp>
#include <OsiSolverParameters.hpp>
#include <OsiClpSolverInterface.hpp>

#ifdef HAS_MOSEK
#include <OsiMskSolverInterface.hpp>
#endif

//#include <OsiCpxSolverInterface.hpp>

#include "LPBoostMulticlassClassifier.h"

namespace Boosting {

LPBoostMulticlassClassifier::LPBoostMulticlassClassifier(
	int number_classes, double nu, bool weight_sharing)
	: initialized(false), number_classes(number_classes), nu(nu),
		weight_sharing(weight_sharing), number_classifiers(0),
		rho(0), gamma(0), use_interior_point(false) {
	assert(nu > 0.0);
	assert(nu <= 1.0);
}

LPBoostMulticlassClassifier::~LPBoostMulticlassClassifier() {
	if (si != NULL)
		delete si;
}

void LPBoostMulticlassClassifier::InitializeBoosting(
	const std::vector<int>& labels, bool interior_point,
	const std::string& solver) {
	assert(*std::min_element(labels.begin(), labels.end()) >= 0);
	assert(*std::max_element(labels.begin(), labels.end()) < number_classes);
	sample_labels = labels;

	// Initialize LP solver
	if (solver == "cplex") {
//		si = new OsiCpxSolverInterface;
		assert(0);
	} else if (solver == "clp") {
		si = new OsiClpSolverInterface;
	} else if (solver == "mosek") {
	#ifdef  HAS_MOSEK
		OsiMskSolverInterface* mosek_si = new OsiMskSolverInterface;
		si = mosek_si;

		/* Disable crossover to a basic solution
		 */
		MSKtask_t msk = mosek_si->getLpPtr();
		MSKrescodee msk_res = MSK_putintparam(msk,
			MSK_IPAR_INTPNT_BASIS, MSK_BI_NEVER);
		assert(msk_res == MSK_RES_OK);
	#else
	        std::cerr << "Unsupported solver type \"" << solver << "\"."
			<< std::endl;
		exit(EXIT_FAILURE);
	#endif
	
	} else {
		std::cerr << "Unsupported solver type \"" << solver << "\"."
			<< std::endl;
		exit(EXIT_FAILURE);
	}

	/* We use the variable order for no weight-sharing:
	 * [rho, xi, A_{1,1}, ..., A_{1,K}, A_{2,1}, ..., A_{2,K}, ..., A_{M,K}]
	 * where A_{j,i} is the i'th class weight for the j'th weak learner.
	 *
	 * For weight-sharing the order is [rho, xi, alpha_1, ..., alpha_M].
	 *
	 * The ordering is chosen such that we can append variables easily (column
	 * generation).  For now, assume zero weak learners.  Also, the constraint
	 * matrix has column ordering, appending columns will be cheaper than
	 * appending rows.
	 */
	CoinPackedMatrix* matrix = new CoinPackedMatrix(true, 0, 0);
	matrix->setDimensions(0, 1 + sample_labels.size());

	// rho unconstrained, xi >= 0 constraints
	double* varLB = new double[1 + sample_labels.size()];
	double* varUB = new double[1 + sample_labels.size()];
	varLB[0] = -si->getInfinity();
	varUB[0] = si->getInfinity();
	for (unsigned int n = 0 ; n < sample_labels.size () ; ++n) {
		varLB[1 + n] = 0;
		varUB[1 + n] = si->getInfinity();
	}

	// Setup objective function: min -\rho + D \sum_i \xi_i
	double* objective = new double[1 + sample_labels.size()];
	objective[0] = -1.0;	// min: -\rho
	double slack_penalty = 1.0 /
		(nu * static_cast<double>(sample_labels.size()));
	for (unsigned int n = 0 ; n < sample_labels.size () ; ++n)
		objective[1 + n] = slack_penalty;

	// Constraint structure, if weight sharing is used:
	//      \sum_i \alpha_i = 1.
	// If no weight sharing is used:
	//      \sum_i A_{j,i} = 1,   for all j = 1, ..., K.
	norm1_rows = 0;
	if (weight_sharing) {
		CoinPackedVector alpha_sum;
		matrix->appendRow(alpha_sum);
		norm1_rows = 1;
	} else {
		CoinPackedVector alpha_sum;
		for (int cl = 0; cl < number_classes; ++cl)
			matrix->appendRow(alpha_sum);
		norm1_rows = number_classes;
	}

	// Add empty rows.  We need N (K-1) rows for the major constraint system.
	// We use a row-ordered temporary matrix and reserve the correct amount of
	// space so we can skip any reallocations.
	CoinPackedMatrix margin_matrix(false, 0, 0);
	margin_matrix.setDimensions(0, 1 + sample_labels.size());
	margin_matrix.reserve(sample_labels.size() * (number_classes - 1),
		2 * (sample_labels.size() * number_classes), false);
	for (unsigned int n = 0; n < sample_labels.size(); ++n) {
		CoinPackedVector margin_constraint;
		margin_constraint.insert(0, -1.0);	// - rho
		margin_constraint.insert(1 + n, 1.0);	// + xi_n
		for (int y = 0; y < number_classes; ++y) {
			if (y == sample_labels[n])
				continue;
			margin_matrix.appendRow(margin_constraint);
		}
	}
	matrix->bottomAppendPackedMatrix(margin_matrix);

	unsigned int margin_rows = sample_labels.size() * (number_classes - 1);
	double* rowLB = new double[norm1_rows + margin_rows];
	double* rowUB = new double[norm1_rows + margin_rows];
	for (unsigned int n = 0; n < norm1_rows; ++n)
		rowLB[n] = rowUB[n] = 1.0;
	for (unsigned int n = 0; n < margin_rows; ++n) {
		rowLB[norm1_rows + n] = 0.0;
		rowUB[norm1_rows + n] = si->getInfinity();
	}

	// Set constraint matrix
	si->assignProblem(matrix, varLB, varUB, objective, rowLB, rowUB);
	si->setObjSense(1);	// minimize \gamma

	// Use interior-point
	if (solver == "clp" && interior_point) {
		std::cout << "Using barrier solver" << std::endl;
		ClpSolve lp_options;
		// Don't do crossover to a feasible basis.
		lp_options.setSolveType(ClpSolve::useBarrierNoCross);
		//lp_options.setPresolveType(ClpSolve::presolveOn);
		lp_options.setPresolveType(ClpSolve::presolveOff);
		lp_options.setSpecialOption(6, 1);
		dynamic_cast<OsiClpSolverInterface*>(si)->setSolveOptions(lp_options);
	}
	use_interior_point = interior_point;

	initialized = true;
}

void LPBoostMulticlassClassifier::AddMulticlassClassifier(
	const std::vector<std::vector<double> >& response) {
	assert(initialized);
	assert(response.size() == sample_labels.size());
	assert(response[0].size() == static_cast<unsigned int>(number_classes));

	if (weight_sharing) {
		// Add one alpha, hence one column.
		// There are (norm1_rows + N*(K-1)) rows.
		CoinPackedVector col;

		// First, the one-norm constraint
		col.insert(0, 1.0);

		// Second, the margin constraints
		unsigned int idx = norm1_rows;
		for (unsigned int n = 0; n < sample_labels.size(); ++n) {
			for (int cl = 0; cl < number_classes; ++cl) {
				if (cl == sample_labels[n])
					continue;

				// H_{y_n,.}(x_n)' \alpha - H_{cl,.}(x_n)' \alpha
				//    - \rho + \xi_n >= 0
				col.insert(idx, response[n][sample_labels[n]]
					- response[n][cl]);
				idx += 1;
			}
		}
		si->addCol(col, 0.0, si->getInfinity(), 0.0);
	} else {
		/* [rho, xi, A_{1,1}, ..., A_{1,K},
		 *  A_{2,1}, ..., A_{2,K}, A_{M,1}, ..., A_{M,K}]
		 * where A_{j,i} is the i'th class weight for the j'th weak learner.
		 */
		CoinPackedVector cols[number_classes];

		// One-norm constraints
		for (int cl = 0; cl < number_classes; ++cl)
			cols[cl].insert(cl, 1.0);	// ... + A_{M,cl} = 1.

		// Margin constraints
		unsigned int idx = norm1_rows;
		for (unsigned int n = 0; n < sample_labels.size(); ++n) {
			for (int cl = 0; cl < number_classes; ++cl) {
				if (cl == sample_labels[n])
					continue;

				// H_{y_n,.}(x_n)' A_{.,y_n} - H_{cl,.}(x_n)' A_{.,cl}
				//     - \rho + \xi_n >= 0
				cols[sample_labels[n]].insert(idx, response[n][sample_labels[n]]);
				cols[cl].insert(idx, -response[n][cl]);
				idx += 1;
			}
		}
		double colLB[number_classes];
		double colUB[number_classes];
		double col_obj[number_classes];
		for (int cl = 0; cl < number_classes; ++cl) {
			colLB[cl] = 0.0;	// A_{M,cl} >= 0
			colUB[cl] = si->getInfinity();
			col_obj[cl] = 0.0;
		}

		CoinPackedVectorBase* cols_p[number_classes];
		for (int cl = 0; cl < number_classes; ++cl)
			cols_p[cl] = &cols[cl];
		si->addCols(number_classes, cols_p, colLB, colUB, col_obj);
	}
	number_classifiers += 1;
}

void LPBoostMulticlassClassifier::Update() {
	assert(initialized);

	//si->writeMps ("toughone", "mps", si->getObjSense());

	//si->messageHandler()->setLogLevel(0);	// no verbosity
	if (use_interior_point)
		si->initialSolve();	// Complete initial solve.
	else
		si->resolve();	// Warm-start solving (we only add constraints)

	if (si->isProvenOptimal() == false) {
		std::cerr << "Linear Program Solver failed." << std::endl;
		std::cerr << "Problem: " << si->getNumCols() << " variables, "
			<< si->getNumRows() << " rows." << std::endl;
		si->writeMps("LP-CRASH", "mps", si->getObjSense());
		std::cerr << "Written problem to file \"LP-CRASH.MPS\" for analysis."
			<< std::endl;
		std::cerr << "         problem sense: "
			<< si->getObjSense() << std::endl;
		std::cerr << "STATUS:  numerical difficulties: "
			<< (si->isAbandoned() ? "YES" : "no") << std::endl;
		std::cerr << "STATUS:       primal infeasible: "
			<< (si->isProvenPrimalInfeasible() ? "YES" : "no") << std::endl;
		std::cerr << "STATUS:         dual infeasible: "
			<< (si->isProvenDualInfeasible() ? "YES" : "no") << std::endl;
		std::cerr << "STATUS: iteration limit reached: "
			<< (si->isIterationLimitReached() ? "YES" : "no") << std::endl;
		assert(0);
		return;
	}

	const double* primal = si->getColSolution();
	rho = primal[0];
	gamma = -si->getObjValue();	// rho - D \sum_i \xi_i
	if (weight_sharing) {
		classifier_weights.resize(1);
		classifier_weights[0].resize(number_classifiers);
		std::copy(primal + 1 + sample_labels.size(),
			primal + 1 + sample_labels.size() + number_classifiers,
			classifier_weights[0].begin());
	} else {
		// [rho, xi, A_{1,1}, ..., A_{1,K}, o o o, A_{M,1}, ..., A_{M,K}]
		// where A_{j,i} is the i'th class weight for the j'th weak learner.
		classifier_weights.resize(number_classes);
		for (int cl1 = 0; cl1 < number_classes; ++cl1) {
			classifier_weights[cl1].resize(number_classifiers);
			for (unsigned int m = 0; m < number_classifiers; ++m) {
				classifier_weights[cl1][m] = primal[1 + sample_labels.size()
					+ m * number_classes + cl1];
			}
		}
	}
}

void LPBoostMulticlassClassifier::WriteMPS(const std::string& mpsfile) const {
	si->writeMps(mpsfile.c_str(), "mps", si->getObjSense());
}

const std::vector<std::vector<double> >&
LPBoostMulticlassClassifier::ClassifierWeights() const {
	return (classifier_weights);
}

double LPBoostMulticlassClassifier::Rho() const {
	return (rho);
}

double LPBoostMulticlassClassifier::Gamma() const {
	return (gamma);
}

}

