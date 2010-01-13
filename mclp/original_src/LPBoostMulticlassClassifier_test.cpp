
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/config/SourcePrefix.h>

#include <vector>
#include <iostream>
#include <algorithm>

#include "LPBoostMulticlassClassifier.h"
#include "LPBoostMulticlassClassifier_test.h"

CPPUNIT_TEST_SUITE_REGISTRATION(LPBoostMulticlassClassifierTest);

// Simple test: three samples, one perfect classifier
// classifier
void LPBoostMulticlassClassifierTest::LearningSimple() {
	Boosting::LPBoostMulticlassClassifier mlp(3, 0.1, true);

	std::vector<int> labels;
	labels.push_back(0);
	labels.push_back(1);
	labels.push_back(2);
	mlp.InitializeBoosting(labels);

	// Multiclass classifier 1: perfect for all classes
	std::vector<std::vector<double> > resp1;
	resp1.push_back(std::vector<double>(3));
	resp1.push_back(std::vector<double>(3));
	resp1.push_back(std::vector<double>(3));
	resp1[0][0] = 1.0;
	resp1[0][1] = 0.0;
	resp1[0][2] = 0.0;
	resp1[1][0] = 0.0;
	resp1[1][1] = 1.0;
	resp1[1][2] = 0.0;
	resp1[2][0] = 0.0;
	resp1[2][1] = 0.0;
	resp1[2][2] = 1.0;
	mlp.AddMulticlassClassifier(resp1);
	mlp.Update();

	// One classifier which is perfect -> result should be perfect
	const std::vector<std::vector<double> >& clw1 = mlp.ClassifierWeights();
	CPPUNIT_ASSERT_DOUBLES_EQUAL(clw1[0][0], 1.0, 1e-8);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(mlp.Rho(), 1.0, 1e-8);
}

// Simple test with noise: three samples, two noisy and one perfect classifier
void LPBoostMulticlassClassifierTest::LearningNoise() {
	Boosting::LPBoostMulticlassClassifier mlp(3, 0.1, true);

	std::vector<int> labels;
	labels.push_back(0);
	labels.push_back(1);
	labels.push_back(2);
	mlp.InitializeBoosting(labels);

	// Multiclass classifier 1: perfect for all classes
	std::vector<std::vector<double> > resp1;
	resp1.push_back(std::vector<double>(3));
	resp1.push_back(std::vector<double>(3));
	resp1.push_back(std::vector<double>(3));
	resp1[0][0] = 1.0;
	resp1[0][1] = 0.75;
	resp1[0][2] = 0.23;
	resp1[1][0] = 0.9;
	resp1[1][1] = 0.8;
	resp1[1][2] = 0.7;
	resp1[2][0] = 0.8;
	resp1[2][1] = 0.9;
	resp1[2][2] = 0.5;
	mlp.AddMulticlassClassifier(resp1);
	mlp.Update();
	const std::vector<std::vector<double> >& clw1 = mlp.ClassifierWeights();
	for (unsigned int n = 0; n < clw1[0].size(); ++n)
		std::cout << "CLW1: alpha_" << n << " " << clw1[0][n] << std::endl;

	std::vector<std::vector<double> > resp2;
	resp2.push_back(std::vector<double>(3));
	resp2.push_back(std::vector<double>(3));
	resp2.push_back(std::vector<double>(3));
	resp2[0][0] = 0.3;
	resp2[0][1] = 0.3;
	resp2[0][2] = 0.3;
	resp2[1][0] = 0.4;
	resp2[1][1] = 0.4;
	resp2[1][2] = 0.4;
	resp2[2][0] = 0.5;
	resp2[2][1] = 0.5;
	resp2[2][2] = 0.5;
	mlp.AddMulticlassClassifier(resp2);
	mlp.Update();
	const std::vector<std::vector<double> >& clw2 = mlp.ClassifierWeights();
	for (unsigned int n = 0; n < clw2[0].size(); ++n)
		std::cout << "CLW2: alpha_" << n << " " << clw2[0][n] << std::endl;

	// Perfect classifier
	std::vector<std::vector<double> > resp3;
	resp3.push_back(std::vector<double>(3));
	resp3.push_back(std::vector<double>(3));
	resp3.push_back(std::vector<double>(3));
	resp3[0][0] = 1.0;
	resp3[0][1] = 0.0;
	resp3[0][2] = 0.0;
	resp3[1][0] = 0.0;
	resp3[1][1] = 1.0;
	resp3[1][2] = 0.0;
	resp3[2][0] = 0.0;
	resp3[2][1] = 0.0;
	resp3[2][2] = 1.0;
	mlp.AddMulticlassClassifier(resp3);
	mlp.Update();
	const std::vector<std::vector<double> >& clw3 = mlp.ClassifierWeights();
	for (unsigned int n = 0; n < clw3[0].size(); ++n)
		std::cout << "CLW3: alpha_" << n << " " << clw3[0][n] << std::endl;

	std::vector<std::vector<double> > resp4;
	resp4.push_back(std::vector<double>(3));
	resp4.push_back(std::vector<double>(3));
	resp4.push_back(std::vector<double>(3));
	resp4[0][0] = 0.5;
	resp4[0][1] = 0.8;
	resp4[0][2] = 0.2;
	resp4[1][0] = 0.0;
	resp4[1][1] = 0.2;
	resp4[1][2] = 0.1;
	resp4[2][0] = 0.8;
	resp4[2][1] = 0.0;
	resp4[2][2] = 0.2;
	mlp.AddMulticlassClassifier(resp4);
	mlp.Update();
	const std::vector<std::vector<double> >& clw4 = mlp.ClassifierWeights();
	for (unsigned int n = 0; n < clw4[0].size(); ++n)
		std::cout << "CLW4: alpha_" << n << " " << clw4[0][n] << std::endl;

	// One classifier which is perfect -> result should be perfect
	CPPUNIT_ASSERT_DOUBLES_EQUAL(clw4[0][2], 1.0, 1e-8);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(mlp.Rho(), 1.0, 1e-8);
}

int main(int argc, char **argv) {
	CPPUNIT_NS::TestResult controller;
	CPPUNIT_NS::TestResultCollector result;
	controller.addListener(&result);        
	CPPUNIT_NS::BriefTestProgressListener progress;
	controller.addListener(&progress);      

	CPPUNIT_NS::TestRunner runner;
	runner.addTest(CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest());
	runner.run(controller);

	CPPUNIT_NS::CompilerOutputter outputter(&result, CPPUNIT_NS::stdCOut());
	outputter.write(); 

	return (result.wasSuccessful() ? 0 : 1);
}

