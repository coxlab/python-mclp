/* Simple command line utility for multiclass LPBoost.
 *
 * Copyright (C) 2008 -- Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <ext/numeric>

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "mex.h"

#include "LPBoostMulticlassClassifier.h"


static void read_matlab_data(Boosting::LPBoostMulticlassClassifier& mlp,
							  const mxArray *scores) {
	// responses[n][cl], the n'th sample, cl'th class
	// response
	std::vector<std::vector<double> > responses;
	
	const mwSize *dims = mxGetDimensions(scores);

	assert(dims[0] && "Input should be 3D array");
	assert(dims[1] && "Input should be 3D array");
	assert(dims[2] && "Input should be 3D array");

	unsigned int number_examples = dims[0];
	unsigned int number_classifiers = dims[1];
	unsigned int number_classes = dims[2];
	double *data = mxGetPr(scores);

	printf("nClassifier = %d\n",number_classifiers);
	printf("nExamples = %d\n",number_examples);
	printf("nClasses = %d\n",number_classes);

	// We build and add the weak learners one after another 
	for (unsigned int m = 0, m2=0; m < number_classifiers; ++m, m2+=number_examples){
		responses.resize(number_examples);
		for (unsigned int n = 0; n < number_examples; ++n)
			responses[n].resize(number_classes);
	
		for (unsigned int cl = 0, cl2=0; cl < number_classes; ++cl,cl2+=number_classifiers*number_examples)
			for (unsigned int n = 0; n < number_examples; ++n) 
				responses[n][cl] = data[cl2+m2+n];

		// Add weak learner responses to multiclass LP
		mlp.AddMulticlassClassifier(responses);
		responses.clear();	// free some memory now
	}
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	bool verbose = true;
	bool interior_point = true;
	double nu;
	bool weight_sharing;
	std::string solver;
	std::vector<int> labels;

	if (nrhs<4)
		mexErrMsgTxt("Too few input arguments.");

	if (mxGetNumberOfDimensions(prhs[0])!=3)
		mexErrMsgTxt("scores must be in format [DxMxN]");

	const mwSize *dims = mxGetDimensions(prhs[0]);
	unsigned int number_examples = dims[0];
	unsigned int number_classifiers = dims[1];
	unsigned int number_classes = dims[2];

	if (mxGetM(prhs[1])*mxGetN(prhs[1]) != number_examples)
		mexErrMsgTxt("Dimension mismatch between labels and scores");
	
	nu = (double)(*(mxGetPr(prhs[2])));
	printf("reg parameters nu = %.4f\n",nu);

	weight_sharing = (double)(*(mxGetPr(prhs[3])));

	// set some default values
	interior_point = true;
	solver = "mosek";

	// parse labels
	double *y = mxGetPr(prhs[1]);
	int max_label = -1;
	labels.clear();
	y = mxGetPr(prhs[1]);

	int min_label = INT_MAX;
	for ( unsigned int i=0 ; i<mxGetM(prhs[1]) ; i++ ) {
		labels.push_back((int)y[i]);
		if (max_label < y[i])
			max_label = (int)y[i];
		if (min_label > y[i])
			min_label = (int)y[i];
	}

	if (min_label>0)
		std::transform(labels.begin(), labels.end(), labels.begin(), 
					   std::bind2nd(std::plus<int>(),-min_label));

	if (verbose)
		mexPrintf("%d samples found\n",labels.size());

	if (number_classes <= 0)
		mexErrMsgTxt("number of classes less than 1");

	// Instantiate multiclass classifier and fill it with training data
	Boosting::LPBoostMulticlassClassifier mlp(number_classes, nu, weight_sharing);
	mlp.InitializeBoosting(labels, interior_point, solver);

	// read the matlab data
	read_matlab_data(mlp, prhs[0]);

	// Solve
	if (verbose) mexPrintf("Solving linear program...");
	mlp.Update();
	if (verbose) mexPrintf("Done.");
	if (verbose) mexPrintf("Soft margin %.4f, objective = %.4f\n",mlp.Rho(),mlp.Gamma());

	const std::vector<std::vector<double> >& clw = mlp.ClassifierWeights();

	// Return weights
	plhs[0] = mxCreateDoubleMatrix((weight_sharing ? 1 : number_classes), number_classifiers, mxREAL);
	double *B = mxGetPr(plhs[0]);

	for (unsigned int aidx = 0; aidx < clw.size(); ++aidx) {
		for (unsigned int bidx = 0; bidx < clw[aidx].size(); ++bidx) {
			B[bidx*clw.size()+aidx] = clw[aidx][bidx];
		}
	}
}


