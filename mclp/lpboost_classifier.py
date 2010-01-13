from _mclp import LPBoostMulticlassClassifier_wrap
import numpy

# A Python layer to add syntactic sugar, error checking, etc. to the 
# boost-wrapped object


class LPBoostMulticlassClassifier (LPBoostMulticlassClassifier_wrap):

    
    def __init__(self, number_of_classes, nu, **kwargs):
    
        self.number_of_classes = number_of_classes
        self.nu = nu
        self.weight_sharing = kwargs.get("weight_sharing", True)
        self.labels = kwargs.get("labels", range(0, self.number_of_classes))
        if(self.labels.__class__ == numpy.ndarray):
            self.labels = self.labels.tolist()
        self.interior_point = kwargs.get("interior_point", False)
        self.solver = kwargs.get("solver", "clp")
        
        
        LPBoostMulticlassClassifier_wrap.__init__(self, self.number_of_classes, self.nu, self.weight_sharing)
        self.initialize_boosting(self.labels, self.interior_point, self.solver)

    @property
    def weights(self): 
        return self.get_weights()

    def add_multiclass_classifier(self, classifier):
        """
        Add information about an additional weak learner to the queue of classifiers
        to be blended.
        """
        
        formatted = classifier
        
        if(formatted.__class__ == numpy.ndarray):
            formatted = formatted.tolist()
        
        # TODO: check the size of the input
        
        LPBoostMulticlassClassifier_wrap.add_multiclass_classifier(self, formatted)
