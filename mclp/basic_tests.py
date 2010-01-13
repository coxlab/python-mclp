
def simple_learning_test():
    import numpy
    from mclp import *

    c = LPBoostMulticlassClassifier(3, 0.1)

    c.initialize_boosting([0,1,2], False, "clp")
    c.add_multiclass_classifier([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])

    c.update()

    assert(c.weights[0][0] - 1.0 <  1e-8)
    assert(c.rho - 1.0 < 1e-8)

def noise_learning_test():
    
    import numpy
    from mclp import LPBoostMulticlassClassifier
    
    # do all of the setup implicitly this time
    c = LPBoostMulticlassClassifier(3, 0.1)

    # ok to use numpy arrays
    c.add_multiclass_classifier(numpy.eye(3))

    # junk / noise
    c.add_multiclass_classifier([[1.0, 0.4, 0.2],
                                 [0.4, 0.4, 0.4],
                                 [0.6, 0.1, 0.2]])
                  
    # junk / noise              
    c.add_multiclass_classifier([[0.3, 0.3, 0.3],
                                 [0.4, 0.4, 0.4],
                                 [0.5, 0.5, 0.5]])

    c.update()

    assert(c.weights[0][0] - 1.0 <  1e-8)
    assert(c.rho - 1.0 < 1e-8)