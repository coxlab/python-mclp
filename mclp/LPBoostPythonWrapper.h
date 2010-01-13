#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "LPBoostMulticlassClassifier.h"
#include "container_conversions.h"

using namespace Boosting;
using namespace boost::python;
using namespace container_conversions;


BOOST_PYTHON_MODULE(_mclp)
{

    typedef std::vector<int> IntVector; 
    class_<IntVector>("IntVector")
        .def(vector_indexing_suite<IntVector>())
        .def(init<const IntVector&>());
    
    
    typedef std::vector<double> DoubleVector; 
    class_<DoubleVector>("DoubleVector")
        .def(vector_indexing_suite<DoubleVector>())
        .def(init<const DoubleVector&>());
        
    typedef std::vector<DoubleVector> DVVector; 
    class_<DVVector>("DVVector") 
        .def(vector_indexing_suite<DVVector>())
        .def(init<const DVVector&>());
        
    
    from_python_sequence<std::vector<int>, variable_capacity_policy>();
    from_python_sequence<std::vector<double>, variable_capacity_policy>();
    from_python_sequence<std::vector<DoubleVector>, variable_capacity_policy>();
    
    class_<LPBoostMulticlassClassifier>("LPBoostMulticlassClassifier_wrap", init<int, double, bool>())
        .def("initialize_boosting", &LPBoostMulticlassClassifier::InitializeBoosting, "(Re)initialize the object to allow boosting")
        .def("add_multiclass_classifier",  &LPBoostMulticlassClassifier::AddMulticlassClassifier)
        .def("update",  &LPBoostMulticlassClassifier::Update, "Solve for the optimal blend of the weak learners")
        .def("get_weights", &LPBoostMulticlassClassifier::ClassifierWeights, return_value_policy<copy_const_reference>())
        .add_property("rho", &LPBoostMulticlassClassifier::Rho)
        .add_property("gamma", &LPBoostMulticlassClassifier::Gamma)
    ;
    
    

}
