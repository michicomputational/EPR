PROGRAM = main

SRCS = code_objects/after_run_neurongroup_spike_thresholder_codeobject.cpp code_objects/neurongroup_spike_resetter_codeobject.cpp code_objects/neurongroup_spike_thresholder_codeobject.cpp code_objects/neurongroup_stateupdater_codeobject.cpp code_objects/spikemonitor_codeobject.cpp main.cpp network.cpp objects.cpp run.cpp synapses_classes.cpp
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/dynamic_array.h code_objects/after_run_neurongroup_spike_thresholder_codeobject.h code_objects/neurongroup_spike_resetter_codeobject.h code_objects/neurongroup_spike_thresholder_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/spikemonitor_codeobject.h network.h objects.h run.h synapses_classes.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = -I"/home/michael_lmu/EPR/.venv/include" -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -L"/home/michael_lmu/EPR/.venv/lib" -Wl,--enable-new-dtags,-rpath,"/home/michael_lmu/EPR/.venv/lib" 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CXX) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CXX) $(CXXFLAGS) $< -o $@