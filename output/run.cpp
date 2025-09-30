#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/poissoninput_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/spikemonitor_1_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_1_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_2_pre_codeobject.h"
#include "code_objects/synapses_2_pre_push_spikes.h"
#include "code_objects/synapses_2_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject_1.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    brian::statemonitor_1_clock_1.timestep = brian::_array_statemonitor_1_clock_1_timestep;
    brian::statemonitor_1_clock_1.dt = brian::_array_statemonitor_1_clock_1_dt;
    brian::statemonitor_1_clock_1.t = brian::_array_statemonitor_1_clock_1_t;
    brian::statemonitor_clock_1.timestep = brian::_array_statemonitor_clock_1_timestep;
    brian::statemonitor_clock_1.dt = brian::_array_statemonitor_clock_1_dt;
    brian::statemonitor_clock_1.t = brian::_array_statemonitor_clock_1_t;
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


