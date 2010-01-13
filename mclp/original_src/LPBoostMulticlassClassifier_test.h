
#ifndef LPBOOSTMULTICLASSCLASSIFIER_TEST_H
#define LPBOOSTMULTICLASSCLASSIFIER_TEST_H

#include <cppunit/extensions/HelperMacros.h>

class LPBoostMulticlassClassifierTest : public CPPUNIT_NS::TestFixture {
	CPPUNIT_TEST_SUITE(LPBoostMulticlassClassifierTest);
	CPPUNIT_TEST(LearningSimple);
	CPPUNIT_TEST(LearningNoise);
	CPPUNIT_TEST_SUITE_END();

protected:
	void LearningSimple();
	void LearningNoise();
};

#endif


