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
#include <boost/program_options.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/filesystem.hpp>

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "LPBoostMulticlassClassifier.h"

namespace po = boost::program_options;

// Return number of classes found or -1 on failure
static int
read_problem(const std::string& train_filename,
	std::vector<int>& labels,
	std::vector<std::vector<std::string> >& data_S_M) {
	labels.clear();
	data_S_M.clear();

	std::ifstream in(train_filename.c_str());
	if (in.fail())
		return (-1);

	// Read linewise
	std::string line;
	std::vector<std::string> current_data;
	int max_label = -1;
	unsigned int partitions = 0;
	while (in.eof() == false) {
		std::getline(in, line);
		if (line.size() == 0)
			continue;	// skip over empty lines

		// Line format: label text1.txt text2.txt ... textP.txt
		std::istringstream is(line);
		int label;
		is >> label;
		labels.push_back(label);
		if (label > max_label)
			max_label = label;

		current_data.clear();
		while (is.eof() == false) {
			std::string value;
			is >> value;
			current_data.push_back(value);
		}
		// Ensure the equal number of data files for all samples
		if (partitions == 0)
			partitions = current_data.size();
		assert(partitions == current_data.size());
		data_S_M.push_back(current_data);
	}
	in.close();

	return (max_label + 1);
}

static unsigned int count_lines(const std::string& filename) {
	std::ifstream in(filename.c_str());
	if (in.fail()) {
		std::cerr << "count_lines, failed to open \"" << filename
			<< "\"." << std::endl;
		exit(EXIT_FAILURE);
	}
	unsigned int lines = 0;
	std::string line;
	while (in.eof() == false) {
		std::getline(in, line);
		if (line.size() == 0)
			continue;
		lines += 1;
	}
	in.close();
	return (lines);
}

static void read_problem_data(Boosting::LPBoostMulticlassClassifier& mlp,
	std::vector<std::vector<std::string> >& data_S_M, int number_classes) {
	// responses[m][n][cl], the m'th weak learner, n'th sample, cl'th class
	// response
	std::vector<std::vector<std::vector<double> > > responses;

	// Obtain number of weak learners
	unsigned int number_classifiers = 0;
	for (unsigned int j = 0; j < data_S_M[0].size(); ++j) {
		unsigned int linecount_cur = count_lines(data_S_M[0][j]);
		std::cout << "   input partition " << j << ", "
			<< linecount_cur << " weak learners" << std::endl;
		number_classifiers += linecount_cur;
	}

	std::cout << "Reading problem data" << std::endl;
	std::cout << "Total number of weak learners is "
		<< number_classifiers << std::endl;

	responses.resize(number_classifiers);
	for (unsigned int mi = 0; mi < number_classifiers; ++mi) {
		responses[mi].resize(data_S_M.size());
		for (unsigned int n = 0; n < data_S_M.size(); ++n)
			responses[mi][n].resize(number_classes);
	}

	int fixed_m = -1;
	for (unsigned int n = 0; n < data_S_M.size(); ++n) {
		//std::cout << "Sample " << n << " / " << data_S_M.size() << std::endl;
		// For each sample, read in all weak learner responses
		int m = 0;
		for (unsigned int didx = 0; didx < data_S_M[n].size(); ++didx) {
			std::ifstream in(data_S_M[n][didx].c_str());
			if (in.fail()) {
				std::cerr << "Failed to open file \""
					<< data_S_M[n][didx] << "\" for sample "
					<< n << "." << std::endl;
				exit(EXIT_FAILURE);
			}
			// M rows, K responses each, linewise read
			std::string line;
			while (in.eof() == false) {
				std::getline(in, line);
				if (line.size() == 0)
					continue;	// skip over empty lines

				std::istringstream is(line);
				for (int cl = 0; cl < number_classes; ++cl) {
					assert(is.eof() == false);
					is >> responses[m][n][cl];
				}
				m += 1;	// next weak learner
			}
			in.close();
		}
		if (fixed_m == -1)
			fixed_m = m;
		if (fixed_m != m) {
			std::cerr << "Inconsistent number of weak learners across samples, "
				<< "sample " << n << " has " << m << ", previous had "
				<< fixed_m << std::endl;
		}
	}

	// Add weak learner responses to multiclass LP
	for (unsigned int m = 0; m < number_classifiers; ++m) {
		mlp.AddMulticlassClassifier(responses[m]);
		responses[m].clear();	// free some memory now
	}
}

int main(int argc, char* argv[]) {
	bool verbose;
	bool interior_point;
	double nu;
	bool weight_sharing;
	bool force;
	std::string train_filename;
	std::string output_filename;
	std::string solver;
	std::string mpsfile;

	// Command line options
	po::options_description generic("Generic Options");
	generic.add_options()
		("help", "Produce help message")
		("verbose", "Verbose output")
		;

	po::options_description input_options("Input/Output Options");
	input_options.add_options()
		("train", po::value<std::string>
			(&train_filename)->default_value("training.txt"),
			"Training file in \"label s0-m0.txt s0-m1.txt ...\" format, "
			"one sample per row.")
		("output", po::value<std::string>
			(&output_filename)->default_value("output.txt"),
			"File to write weight matrix to.  If \"--weight_sharing 1\" is "
			"used, this is a single line containing the alpha vector.  If "
			"no weight sharing is used, it is a matrix with number-of-classes "
			"rows and number-of-weak-learners columns.")
		("force", po::value<bool>(&force)->default_value(false),
			"Force overwriting the output file.  Otherwise, if the "
			"output file already exists, the program is aborted immediately.")
		("writemps", po::value<std::string>(&mpsfile)->default_value(""),
			"Write linear programming problem as MPS file.")
		;

	po::options_description lpboost_options("LPBoost Options");
	lpboost_options.add_options()
		("nu", po::value<double>(&nu)->default_value(0.1),
			"nu-parameter for 2-class LPBoost.  A larger value "
			"indicates stronger regularization")
		("weight_sharing", po::value<bool>(&weight_sharing)->default_value(true),
			"Share classifier weights among all classes.")
		("interior_point",
			po::value<bool>(&interior_point)->default_value(true),
			"Use interior point (true) or simplex method (false) to "
			"solve the LPBoost master problem")
		("solver", po::value<std::string>(&solver)->default_value("clp"),
			"LP solver to use.  One of \"clp\" or \"mosek\".")
		;

	po::options_description all_options;
	all_options.add(generic).add(input_options).add(lpboost_options);
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
	po::notify(vm);

	// Boolean flags
	verbose = vm.count("verbose");

	if (vm.count("help")) {
		std::cerr << "mclp $Id: mclp.cpp 1229 2008-03-10 10:26:34Z nowozin $" << std::endl;
		std::cerr << "===================================================="
			<< "===========================" << std::endl;
		std::cerr << "Copyright (C) 2008 -- "
			<< "Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>"
			<< std::endl;
		std::cerr << std::endl;
		std::cerr << "Usage: mclp [options]" << std::endl;
		std::cerr << std::endl;
		std::cerr << "Train a multiclass LPBoost model for given and fixed multiclass "
			<< "weak learners." << std::endl;
		std::cerr << all_options << std::endl;

		exit(EXIT_SUCCESS);
	}

	// Check if output file already exists
	if (boost::filesystem::exists(boost::filesystem::path(output_filename))
		&& force == false) {
		std::cout << "Output file \"" << output_filename << "\" "
			<< "already exists, exiting." << std::endl;
		exit(EXIT_SUCCESS);
	}

	// Read in training data
	std::cout << "Training file: " << train_filename << std::endl;
	std::vector<int> labels;	// discrete class labels, >= 0, < K.
	std::vector<std::vector<std::string> > data_S_M;	// [n][m]
	int number_classes = read_problem(train_filename, labels, data_S_M);
	if (number_classes <= 0) {
		std::cerr << "Failed to read in training data." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << labels.size() << " samples, "
		<< number_classes << " classes." << std::endl;

	// Instantiate multiclass classifier and fill it with training data
	Boosting::LPBoostMulticlassClassifier mlp(number_classes, nu, weight_sharing);
	mlp.InitializeBoosting(labels, interior_point, solver);
	read_problem_data(mlp, data_S_M, number_classes);

	if (mpsfile.empty() == false)
		mlp.WriteMPS(mpsfile);

	// Solve
	std::cout << "Solving linear program..." << std::endl;
	mlp.Update();
	std::cout << "Done." << std::endl;
	std::cout << "Soft margin " << mlp.Rho() << ", objective "
		<< mlp.Gamma() << std::endl;

	// Print weights
	const std::vector<std::vector<double> >& clw = mlp.ClassifierWeights();
	std::cout << "Writing (K,M) weight matrix to \""
		<< output_filename << "\", K = "
		<< (weight_sharing ? 1 : number_classes)
		<< ", M = " << clw[0].size() << std::endl;

	std::ofstream wout(output_filename.c_str());
	if (wout.fail()) {
		std::cerr << "Failed to open \"" << output_filename
			<< "\" for writing." << std::endl;
		exit(EXIT_FAILURE);
	}
	wout << std::setprecision(12);
	for (unsigned int aidx = 0; aidx < clw.size(); ++aidx) {
		for (unsigned int bidx = 0; bidx < clw[aidx].size(); ++bidx) {
			wout << (bidx == 0 ? "" : " ") << clw[aidx][bidx];
		}
		wout << std::endl;
	}
	wout.close();

	exit(EXIT_SUCCESS);
}


