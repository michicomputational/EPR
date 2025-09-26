PROGRAM = main

SRCS = brianlib/randomkit/randomkit.c code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.cpp code_objects/before_run_synapses_4_pre_push_spikes.cpp code_objects/before_run_synapses_5_pre_push_spikes.cpp code_objects/before_run_synapses_6_pre_push_spikes.cpp code_objects/before_run_synapses_7_pre_push_spikes.cpp code_objects/neurongroup_1_spike_resetter_codeobject.cpp code_objects/neurongroup_1_spike_thresholder_codeobject.cpp code_objects/neurongroup_1_stateupdater_codeobject.cpp code_objects/poissoninput_1_codeobject.cpp code_objects/spikemonitor_2_codeobject.cpp code_objects/spikemonitor_3_codeobject.cpp code_objects/synapses_4_pre_codeobject.cpp code_objects/synapses_4_pre_push_spikes.cpp code_objects/synapses_4_stateupdater_codeobject.cpp code_objects/synapses_4_summed_variable_I_syn_ee_post_codeobject.cpp code_objects/synapses_4_synapses_create_generator_codeobject.cpp code_objects/synapses_5_pre_codeobject.cpp code_objects/synapses_5_pre_push_spikes.cpp code_objects/synapses_5_stateupdater_codeobject.cpp code_objects/synapses_5_summed_variable_I_syn_ei_post_codeobject.cpp code_objects/synapses_5_synapses_create_generator_codeobject.cpp code_objects/synapses_6_pre_codeobject.cpp code_objects/synapses_6_pre_push_spikes.cpp code_objects/synapses_6_stateupdater_codeobject.cpp code_objects/synapses_6_summed_variable_I_syn_ie_post_codeobject.cpp code_objects/synapses_6_synapses_create_generator_codeobject.cpp code_objects/synapses_7_pre_codeobject.cpp code_objects/synapses_7_pre_push_spikes.cpp code_objects/synapses_7_stateupdater_codeobject.cpp code_objects/synapses_7_summed_variable_I_syn_ii_post_codeobject.cpp code_objects/synapses_7_synapses_create_generator_codeobject.cpp main.cpp network.cpp objects.cpp run.cpp synapses_classes.cpp
H_SRCS = brianlib/clocks.h brianlib/common_math.h brianlib/dynamic_array.h code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h code_objects/before_run_synapses_4_pre_push_spikes.h code_objects/before_run_synapses_5_pre_push_spikes.h code_objects/before_run_synapses_6_pre_push_spikes.h code_objects/before_run_synapses_7_pre_push_spikes.h code_objects/neurongroup_1_spike_resetter_codeobject.h code_objects/neurongroup_1_spike_thresholder_codeobject.h code_objects/neurongroup_1_stateupdater_codeobject.h code_objects/poissoninput_1_codeobject.h code_objects/spikemonitor_2_codeobject.h code_objects/spikemonitor_3_codeobject.h code_objects/synapses_4_pre_codeobject.h code_objects/synapses_4_pre_push_spikes.h code_objects/synapses_4_stateupdater_codeobject.h code_objects/synapses_4_summed_variable_I_syn_ee_post_codeobject.h code_objects/synapses_4_synapses_create_generator_codeobject.h code_objects/synapses_5_pre_codeobject.h code_objects/synapses_5_pre_push_spikes.h code_objects/synapses_5_stateupdater_codeobject.h code_objects/synapses_5_summed_variable_I_syn_ei_post_codeobject.h code_objects/synapses_5_synapses_create_generator_codeobject.h code_objects/synapses_6_pre_codeobject.h code_objects/synapses_6_pre_push_spikes.h code_objects/synapses_6_stateupdater_codeobject.h code_objects/synapses_6_summed_variable_I_syn_ie_post_codeobject.h code_objects/synapses_6_synapses_create_generator_codeobject.h code_objects/synapses_7_pre_codeobject.h code_objects/synapses_7_pre_push_spikes.h code_objects/synapses_7_stateupdater_codeobject.h code_objects/synapses_7_summed_variable_I_syn_ii_post_codeobject.h code_objects/synapses_7_synapses_create_generator_codeobject.h network.h objects.h run.h synapses_classes.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
OPTIMISATIONS = -Ibrianlib/randomkit -I/usr/include -w -O3 -ffast-math -fno-finite-math-only -march=native -std=c++11
CXXFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/usr/lib -Wl,--enable-new-dtags,-R/usr/lib 
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