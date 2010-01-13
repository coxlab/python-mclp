/* Simple random multiclass test data generator to evaluate scaling behaviour.
 *
 * Copyright (C) 2008 -- Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/random.hpp>

#include <assert.h>
#include <stdlib.h>
#include <math.h>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
	unsigned int number_samples;
	unsigned int number_classifiers;
	unsigned int number_classes;
	std::string training_output_filename;
	std::string training_prefix;

	po::options_description generic("Generic Options");
	generic.add_options()
		("help", "Produce help message")
		("verbose", "Verbose output")
		;

	po::options_description input_options("Input/Output Options");
	input_options.add_options()
		("training_output", po::value<std::string>
			(&training_output_filename)->default_value("training.txt"),
			"The master training.txt to be used with mclp.")
		("training_prefix", po::value<std::string>
			(&training_prefix)->default_value("data"),
			"The data file prefix, the filenames will be prefix_n.txt")
		;

	po::options_description gen_options("Sample Generation Options");
	gen_options.add_options()
		("number_samples", po::value<unsigned int>
			(&number_samples)->default_value(1000),
			"Number of samples to generate, totally.  The sample count is "
			"split evenly over the classes.")
		("number_classifiers", po::value<unsigned int>
			(&number_classifiers)->default_value(64),
			"Number of multiclass weak learners to use.")
		("number_classes", po::value<unsigned int>
			(&number_classes)->default_value(100))
		;

	// Parse options
	po::options_description all_options;
	all_options.add(generic).add(input_options).add(gen_options);
	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).options(all_options).run(), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cerr << "mclpgen $Id: mclpgen.cpp 1211 2008-03-03 21:59:08Z nowozin $" << std::endl;
		std::cerr << "===================================================="
			<< "===========================" << std::endl;
		std::cerr << "Copyright (C) 2008 -- "
			<< "Sebastian Nowozin <sebastian.nowozin@tuebingen.mpg.de>"
			<< std::endl;
		std::cerr << std::endl;
		std::cerr << "Usage: mclpgen [options]" << std::endl;
		std::cerr << std::endl;
		std::cerr << all_options << std::endl;

		exit(EXIT_SUCCESS);
	}

	// Create master training file and keep it open until the end
	std::ofstream training_out(training_output_filename.c_str());
	if (training_out.fail()) {
		std::cerr << "Failed to open output file \""
			<< training_output_filename << "\"." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Generate random number generators, "out of thin air"
	boost::mt19937 rng;
	boost::uniform_int<> dist_classlabel(0, number_classes - 1);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> >
		rand_class_label(rng, dist_classlabel);
	boost::uniform_real<double> dist_uniform(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >
		rand_sample_element(rng, dist_uniform);

	// Generate individual samples
	for (unsigned int n = 0; n < number_samples; ++n) {
		std::ostringstream cur_data_filename;
		cur_data_filename << training_prefix << "_" << n << ".txt";
		int cur_class_label = rand_class_label();
		training_out << cur_class_label << " "
			<< cur_data_filename.str() << std::endl;

		// Generate data file
		std::ofstream data_out(cur_data_filename.str().c_str());
		if (data_out.fail()) {
			std::cerr << "Failed to open data output file \""
				<< cur_data_filename << "\" for writing." << std::endl;
			exit(EXIT_FAILURE);
		}
		for (unsigned int m = 0; m < number_classifiers; ++m) {
			for (unsigned int k = 0; k < number_classes; ++k) {
				if (k > 0)
					data_out << " ";
				data_out << rand_sample_element();
			}
			data_out << std::endl;
		}
		data_out.close();
	}
	training_out.close();

	std::cout << "Written " << number_samples << " files, "
		<< number_classifiers << " weak learners, "
		<< number_classes << " classes." << std::endl;
	std::cout << "Constraint matrix will have approximately "
		<< (number_samples * number_classifiers * number_classes)
		<< " non-zero elements." << std::endl;
}

