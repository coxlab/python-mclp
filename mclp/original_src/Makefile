# The used C++ compiler
CPP=g++-4.4.1-sd

# Additional compilation flags#
#CPPFLAGS=-Wall -Werror -std=c++98 -O2 -msse -msse2 -funroll-loops -fPIC
#CPPFLAGS=-Wall -Werror -std=c++98 -O2 -march=native -msse -msse2 -funroll-loops
CPPFLAGS=-Wall -Werror -std=c++98 -g -ggdb -fPIC


####
#### External dependency configuration
####

# COIN-OR Open Solver Interface (Osi), version 0.97.1 or higher
#COIN_INCLUDE=-I/scratch_net/biwidl07/opt/coin-osi-0.100.1/include/coin
#COIN_LIB=-L/scratch_net/biwidl07/opt/coin-osi-0.100.1/lib \
#	-lOsiMsk \
#	-lOsiClp -lOsi -lClp -lCoinUtils -lm \
#	-L/usr/lib/sse2  /usr/lib/libatlas.so.3  /usr/lib/libcblas.so.3 -lg2c \
	-L/usr/lib/atlas -lblas /usr/lib/libf77blas.so.3.0

COIN_INCLUDE=-I/scratch_net/biwidl07/opt/coin-Osi-0.100.1/include/coin -I/scratch_net/biwidl07/opt/coin-Clp-1.10/include/coin
COIN_LIB=-L/scratch_net/biwidl07/opt/coin-Osi-0.100.1/lib -L/scratch_net/biwidl07/opt/coin-Clp-1.10/lib \
	-lOsiMsk \
	-lOsiClp -lOsi -lClp -lCoinUtils -lm \
	-L/usr/lib/sse2  /usr/lib/libatlas.so.3  /usr/lib/libcblas.so.3 -lg2c \
	-L/usr/lib/atlas -lblas /usr/lib/libf77blas.so.3.0
#-llapack_atlas -lf77blas -latlas -lcblas

#CPLEX_INCLUDE=-I/opt/cplex-9.1/include
#CPLEX_LIB=-L/opt/cplex-9.1/lib -lOsiCpx -lcplex -lpthread

MOSEK_INCLUDE=-I/scratch_net/biwidl07/opt/mosek/5/tools/platform/linux64x86/h
MOSEK_LIB=-L/scratch_net/biwidl07/opt/mosek/5/tools/platform/linux64x86/bin -lmosek64 -lguide -lirc -limf -lpthread -lsvml #-lunwind 

#COIN_LIB=-L/opt/coin-osi-0.97.1/lib -lOsiClp -lOsi -lClp -lCoinUtils -lm \
#	-L/usr/lib/sse2 -llapack_atlas -lf77blas -latlas -lcblas -lg2c \
#	-L/usr/lib/atlas/sse2 -lblas

# boost library
#BOOST_LIB=-L/scratch_net/biwidl07/opt/boost-1.39.0/lib -lboost_filesystem-gcc41-mt-1_35 -lboost_program_options-gcc41-mt-1_35 -lboost_system-gcc41-mt-1_35
BOOST_LIB=-L/scratch_net/biwidl07/opt/boost-1.39.0/lib -lboost_filesystem-gcc43-mt-1_39 -lboost_program_options-gcc43-mt-1_39 -lboost_system-gcc43-mt-1_39
BOOST_INCLUDE=-I/scratch_net/biwidl07/opt/boost-1.39.0/include/boost-1_39

MATLAB_LIB=-L/usr/pack/matlab-7.8r2009a-sd/bin/glnxa64 -lmex
MATLAB_INCLUDE=-I/usr/pack/matlab-7.8r2009a-sd/extern/include

# CPP unit
CPPUNIT_LIB=-lcppunit

###
INCLUDE=$(COIN_INCLUDE) $(MOSEK_INCLUDE) $(BOOST_INCLUDE) 
#$(CPLEX_INCLUDE)

####
#### Build rules
####

# default rule

%.o: %.cpp
	$(CPP) $(CPPFLAGS) $(INCLUDE) -c $^ -o $@

all:	libmclasslpboost mclp mclp_mex.mexa64

mclp_mex.o: mclp_mex.cpp
	$(CPP) $(CPPFLAGS) $(INCLUDE) $(MATLAB_INCLUDE) -c mclp_mex.cpp -o mclp_mex.o

check:	LPBoostMulticlassClassifier_test
	./LPBoostMulticlassClassifier_test

clean:
	rm -f *.o
	rm -f libmclasslpboost.a
	rm -f LPBoostMulticlassClassifier_test
	rm -f mclp

libmclasslpboost:	LPBoostMulticlassClassifier.o
	ar rc libmclasslpboost.a LPBoostMulticlassClassifier.o
	ranlib libmclasslpboost.a

mclp:	libmclasslpboost mclp.o
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o mclp mclp.o libmclasslpboost.a \
		$(COIN_LIB) $(BOOST_LIB) $(MOSEK_LIB)
		#$(CPLEX_LIB)

mclp_mex.mexa64:	libmclasslpboost mclp_mex.o
	$(CPP) $(CPPFLAGS) $(INCLUDE) $(MATLAB_INCLUDE) -shared -o mclp_mex.mexa64 mclp_mex.o libmclasslpboost.a \
		$(COIN_LIB) $(MOSEK_LIB) $(BOOST_LIB) $(MATLAB_LIB)

mclpgen:	libmclasslpboost mclpgen.o
	$(CPP) $(CPPFLAGS) $(INCLUDE) -static -o mclpgen mclpgen.o \
		$(COIN_LIB) $(BOOST_LIB)

LPBoostMulticlassClassifier_test: libmclasslpboost LPBoostMulticlassClassifier_test.o
	$(CPP) $(CPPFLAGS) $(INCLUDE) -o LPBoostMulticlassClassifier_test \
		LPBoostMulticlassClassifier_test.o libmclasslpboost.a \
		$(COIN_LIB) $(CPPUNIT_LIB) $(MOSEK_LIB)


