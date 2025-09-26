#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "randomkit.h"

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/poissoninput_1_codeobject.h"
#include "code_objects/spikemonitor_2_codeobject.h"
#include "code_objects/spikemonitor_3_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_stateupdater_codeobject.h"
#include "code_objects/synapses_4_summed_variable_I_syn_ee_post_codeobject.h"
#include "code_objects/synapses_4_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_stateupdater_codeobject.h"
#include "code_objects/synapses_5_summed_variable_I_syn_ei_post_codeobject.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_stateupdater_codeobject.h"
#include "code_objects/synapses_6_summed_variable_I_syn_ie_post_codeobject.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_7_pre_codeobject.h"
#include "code_objects/synapses_7_pre_push_spikes.h"
#include "code_objects/synapses_7_stateupdater_codeobject.h"
#include "code_objects/synapses_7_summed_variable_I_syn_ii_post_codeobject.h"
#include "code_objects/synapses_7_synapses_create_generator_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    for (int i=0; i<1; i++)
	    rk_randomseed(brian::_mersenne_twister_states[i]);  // Note that this seed can be potentially replaced in main.cpp
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


