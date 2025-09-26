
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network network_1;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikemonitor_2_i;
extern std::vector<double> _dynamic_array_spikemonitor_2_t;
extern std::vector<int32_t> _dynamic_array_spikemonitor_3_i;
extern std::vector<double> _dynamic_array_spikemonitor_3_t;
extern std::vector<int32_t> _dynamic_array_synapses_4__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_4__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_4_A_SE;
extern std::vector<double> _dynamic_array_synapses_4_delay;
extern std::vector<int32_t> _dynamic_array_synapses_4_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_4_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_4_tau_inact;
extern std::vector<double> _dynamic_array_synapses_4_y;
extern std::vector<int32_t> _dynamic_array_synapses_5__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_5__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_5_A_SE;
extern std::vector<double> _dynamic_array_synapses_5_delay;
extern std::vector<int32_t> _dynamic_array_synapses_5_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_5_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_5_tau_inact;
extern std::vector<double> _dynamic_array_synapses_5_y;
extern std::vector<int32_t> _dynamic_array_synapses_6__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_6__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_6_A_SE;
extern std::vector<double> _dynamic_array_synapses_6_delay;
extern std::vector<int32_t> _dynamic_array_synapses_6_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_6_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_6_tau_inact;
extern std::vector<double> _dynamic_array_synapses_6_y;
extern std::vector<int32_t> _dynamic_array_synapses_7__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_7__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_7_A_SE;
extern std::vector<double> _dynamic_array_synapses_7_delay;
extern std::vector<int32_t> _dynamic_array_synapses_7_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_7_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_7_tau_inact;
extern std::vector<double> _dynamic_array_synapses_7_y;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern int32_t *_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double *_array_neurongroup_1_I_syn_ee;
extern const int _num__array_neurongroup_1_I_syn_ee;
extern double *_array_neurongroup_1_I_syn_ei;
extern const int _num__array_neurongroup_1_I_syn_ei;
extern double *_array_neurongroup_1_I_syn_ie;
extern const int _num__array_neurongroup_1_I_syn_ie;
extern double *_array_neurongroup_1_I_syn_ii;
extern const int _num__array_neurongroup_1_I_syn_ii;
extern double *_array_neurongroup_1_lastspike;
extern const int _num__array_neurongroup_1_lastspike;
extern char *_array_neurongroup_1_not_refractory;
extern const int _num__array_neurongroup_1_not_refractory;
extern double *_array_neurongroup_1_R;
extern const int _num__array_neurongroup_1_R;
extern int32_t *_array_neurongroup_1_subgroup_1__sub_idx;
extern const int _num__array_neurongroup_1_subgroup_1__sub_idx;
extern int32_t *_array_neurongroup_1_subgroup__sub_idx;
extern const int _num__array_neurongroup_1_subgroup__sub_idx;
extern double *_array_neurongroup_1_tau_mem;
extern const int _num__array_neurongroup_1_tau_mem;
extern double *_array_neurongroup_1_v;
extern const int _num__array_neurongroup_1_v;
extern int32_t *_array_spikemonitor_2__source_idx;
extern const int _num__array_spikemonitor_2__source_idx;
extern int32_t *_array_spikemonitor_2_count;
extern const int _num__array_spikemonitor_2_count;
extern int32_t *_array_spikemonitor_2_N;
extern const int _num__array_spikemonitor_2_N;
extern int32_t *_array_spikemonitor_3__source_idx;
extern const int _num__array_spikemonitor_3__source_idx;
extern int32_t *_array_spikemonitor_3_count;
extern const int _num__array_spikemonitor_3_count;
extern int32_t *_array_spikemonitor_3_N;
extern const int _num__array_spikemonitor_3_N;
extern int32_t *_array_synapses_4_N;
extern const int _num__array_synapses_4_N;
extern int32_t *_array_synapses_5_N;
extern const int _num__array_synapses_5_N;
extern int32_t *_array_synapses_6_N;
extern const int _num__array_synapses_6_N;
extern int32_t *_array_synapses_7_N;
extern const int _num__array_synapses_7_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////
// synapses_4
extern SynapticPathway synapses_4_pre;
// synapses_5
extern SynapticPathway synapses_5_pre;
// synapses_6
extern SynapticPathway synapses_6_pre;
// synapses_7
extern SynapticPathway synapses_7_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


