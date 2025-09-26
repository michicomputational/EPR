
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network network_1;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 5001;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 5000;
double * _array_neurongroup_1_I_syn_ee;
const int _num__array_neurongroup_1_I_syn_ee = 5000;
double * _array_neurongroup_1_I_syn_ei;
const int _num__array_neurongroup_1_I_syn_ei = 5000;
double * _array_neurongroup_1_I_syn_ie;
const int _num__array_neurongroup_1_I_syn_ie = 5000;
double * _array_neurongroup_1_I_syn_ii;
const int _num__array_neurongroup_1_I_syn_ii = 5000;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 5000;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 5000;
double * _array_neurongroup_1_R;
const int _num__array_neurongroup_1_R = 5000;
int32_t * _array_neurongroup_1_subgroup_1__sub_idx;
const int _num__array_neurongroup_1_subgroup_1__sub_idx = 1000;
int32_t * _array_neurongroup_1_subgroup__sub_idx;
const int _num__array_neurongroup_1_subgroup__sub_idx = 4000;
double * _array_neurongroup_1_tau_mem;
const int _num__array_neurongroup_1_tau_mem = 5000;
double * _array_neurongroup_1_v;
const int _num__array_neurongroup_1_v = 5000;
int32_t * _array_spikemonitor_2__source_idx;
const int _num__array_spikemonitor_2__source_idx = 4000;
int32_t * _array_spikemonitor_2_count;
const int _num__array_spikemonitor_2_count = 4000;
int32_t * _array_spikemonitor_2_N;
const int _num__array_spikemonitor_2_N = 1;
int32_t * _array_spikemonitor_3__source_idx;
const int _num__array_spikemonitor_3__source_idx = 1000;
int32_t * _array_spikemonitor_3_count;
const int _num__array_spikemonitor_3_count = 1000;
int32_t * _array_spikemonitor_3_N;
const int _num__array_spikemonitor_3_N = 1;
int32_t * _array_synapses_4_N;
const int _num__array_synapses_4_N = 1;
int32_t * _array_synapses_5_N;
const int _num__array_synapses_5_N = 1;
int32_t * _array_synapses_6_N;
const int _num__array_synapses_6_N = 1;
int32_t * _array_synapses_7_N;
const int _num__array_synapses_7_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_spikemonitor_2_i;
std::vector<double> _dynamic_array_spikemonitor_2_t;
std::vector<int32_t> _dynamic_array_spikemonitor_3_i;
std::vector<double> _dynamic_array_spikemonitor_3_t;
std::vector<int32_t> _dynamic_array_synapses_4__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_4__synaptic_pre;
std::vector<double> _dynamic_array_synapses_4_A_SE;
std::vector<double> _dynamic_array_synapses_4_delay;
std::vector<int32_t> _dynamic_array_synapses_4_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_4_N_outgoing;
std::vector<double> _dynamic_array_synapses_4_tau_inact;
std::vector<double> _dynamic_array_synapses_4_y;
std::vector<int32_t> _dynamic_array_synapses_5__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_5__synaptic_pre;
std::vector<double> _dynamic_array_synapses_5_A_SE;
std::vector<double> _dynamic_array_synapses_5_delay;
std::vector<int32_t> _dynamic_array_synapses_5_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_5_N_outgoing;
std::vector<double> _dynamic_array_synapses_5_tau_inact;
std::vector<double> _dynamic_array_synapses_5_y;
std::vector<int32_t> _dynamic_array_synapses_6__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_6__synaptic_pre;
std::vector<double> _dynamic_array_synapses_6_A_SE;
std::vector<double> _dynamic_array_synapses_6_delay;
std::vector<int32_t> _dynamic_array_synapses_6_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_6_N_outgoing;
std::vector<double> _dynamic_array_synapses_6_tau_inact;
std::vector<double> _dynamic_array_synapses_6_y;
std::vector<int32_t> _dynamic_array_synapses_7__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_7__synaptic_pre;
std::vector<double> _dynamic_array_synapses_7_A_SE;
std::vector<double> _dynamic_array_synapses_7_delay;
std::vector<int32_t> _dynamic_array_synapses_7_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_7_N_outgoing;
std::vector<double> _dynamic_array_synapses_7_tau_inact;
std::vector<double> _dynamic_array_synapses_7_y;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////
// synapses_4
SynapticPathway synapses_4_pre(
		_dynamic_array_synapses_4__synaptic_pre,
		0, 4000);
// synapses_5
SynapticPathway synapses_5_pre(
		_dynamic_array_synapses_5__synaptic_pre,
		0, 4000);
// synapses_6
SynapticPathway synapses_6_pre(
		_dynamic_array_synapses_6__synaptic_pre,
		4000, 5000);
// synapses_7
SynapticPathway synapses_7_pre(
		_dynamic_array_synapses_7__synaptic_pre,
		4000, 5000);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup_1__spikespace = new int32_t[5001];
    
	for(int i=0; i<5001; i++) _array_neurongroup_1__spikespace[i] = 0;

	_array_neurongroup_1_i = new int32_t[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_i[i] = 0;

	_array_neurongroup_1_I_syn_ee = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_I_syn_ee[i] = 0;

	_array_neurongroup_1_I_syn_ei = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_I_syn_ei[i] = 0;

	_array_neurongroup_1_I_syn_ie = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_I_syn_ie[i] = 0;

	_array_neurongroup_1_I_syn_ii = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_I_syn_ii[i] = 0;

	_array_neurongroup_1_lastspike = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_lastspike[i] = 0;

	_array_neurongroup_1_not_refractory = new char[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_not_refractory[i] = 0;

	_array_neurongroup_1_R = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_R[i] = 0;

	_array_neurongroup_1_subgroup_1__sub_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_1_subgroup_1__sub_idx[i] = 0;

	_array_neurongroup_1_subgroup__sub_idx = new int32_t[4000];
    
	for(int i=0; i<4000; i++) _array_neurongroup_1_subgroup__sub_idx[i] = 0;

	_array_neurongroup_1_tau_mem = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_tau_mem[i] = 0;

	_array_neurongroup_1_v = new double[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_v[i] = 0;

	_array_spikemonitor_2__source_idx = new int32_t[4000];
    
	for(int i=0; i<4000; i++) _array_spikemonitor_2__source_idx[i] = 0;

	_array_spikemonitor_2_count = new int32_t[4000];
    
	for(int i=0; i<4000; i++) _array_spikemonitor_2_count[i] = 0;

	_array_spikemonitor_2_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_2_N[i] = 0;

	_array_spikemonitor_3__source_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor_3__source_idx[i] = 0;

	_array_spikemonitor_3_count = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor_3_count[i] = 0;

	_array_spikemonitor_3_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_3_N[i] = 0;

	_array_synapses_4_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_4_N[i] = 0;

	_array_synapses_5_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_5_N[i] = 0;

	_array_synapses_6_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_6_N[i] = 0;

	_array_synapses_7_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_7_N[i] = 0;

	_dynamic_array_synapses_4_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_4_delay[i] = 0;

	_dynamic_array_synapses_5_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_5_delay[i] = 0;

	_dynamic_array_synapses_6_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_6_delay[i] = 0;

	_dynamic_array_synapses_7_delay.resize(1);
    
	for(int i=0; i<1; i++) _dynamic_array_synapses_7_delay[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_1_i = new int32_t[5000];
    
	for(int i=0; i<5000; i++) _array_neurongroup_1_i[i] = 0 + i;

	_array_neurongroup_1_subgroup_1__sub_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_1_subgroup_1__sub_idx[i] = 4000 + i;

	_array_neurongroup_1_subgroup__sub_idx = new int32_t[4000];
    
	for(int i=0; i<4000; i++) _array_neurongroup_1_subgroup__sub_idx[i] = 0 + i;

	_array_spikemonitor_2__source_idx = new int32_t[4000];
    
	for(int i=0; i<4000; i++) _array_spikemonitor_2__source_idx[i] = 0 + i;

	_array_spikemonitor_3__source_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor_3__source_idx[i] = 0 + i;


	// static arrays

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_1978099143", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_2669362164", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_144223508", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_1__spikespace;
	outfile__array_neurongroup_1__spikespace.open("results/_array_neurongroup_1__spikespace_3155027917", ios::binary | ios::out);
	if(outfile__array_neurongroup_1__spikespace.is_open())
	{
		outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 5001*sizeof(_array_neurongroup_1__spikespace[0]));
		outfile__array_neurongroup_1__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_1_i;
	outfile__array_neurongroup_1_i.open("results/_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_i.is_open())
	{
		outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 5000*sizeof(_array_neurongroup_1_i[0]));
		outfile__array_neurongroup_1_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
	}
	ofstream outfile__array_neurongroup_1_I_syn_ee;
	outfile__array_neurongroup_1_I_syn_ee.open("results/_array_neurongroup_1_I_syn_ee_3788673398", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_I_syn_ee.is_open())
	{
		outfile__array_neurongroup_1_I_syn_ee.write(reinterpret_cast<char*>(_array_neurongroup_1_I_syn_ee), 5000*sizeof(_array_neurongroup_1_I_syn_ee[0]));
		outfile__array_neurongroup_1_I_syn_ee.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_I_syn_ee." << endl;
	}
	ofstream outfile__array_neurongroup_1_I_syn_ei;
	outfile__array_neurongroup_1_I_syn_ei.open("results/_array_neurongroup_1_I_syn_ei_3898924381", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_I_syn_ei.is_open())
	{
		outfile__array_neurongroup_1_I_syn_ei.write(reinterpret_cast<char*>(_array_neurongroup_1_I_syn_ei), 5000*sizeof(_array_neurongroup_1_I_syn_ei[0]));
		outfile__array_neurongroup_1_I_syn_ei.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_I_syn_ei." << endl;
	}
	ofstream outfile__array_neurongroup_1_I_syn_ie;
	outfile__array_neurongroup_1_I_syn_ie.open("results/_array_neurongroup_1_I_syn_ie_1298652794", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_I_syn_ie.is_open())
	{
		outfile__array_neurongroup_1_I_syn_ie.write(reinterpret_cast<char*>(_array_neurongroup_1_I_syn_ie), 5000*sizeof(_array_neurongroup_1_I_syn_ie[0]));
		outfile__array_neurongroup_1_I_syn_ie.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_I_syn_ie." << endl;
	}
	ofstream outfile__array_neurongroup_1_I_syn_ii;
	outfile__array_neurongroup_1_I_syn_ii.open("results/_array_neurongroup_1_I_syn_ii_1154585169", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_I_syn_ii.is_open())
	{
		outfile__array_neurongroup_1_I_syn_ii.write(reinterpret_cast<char*>(_array_neurongroup_1_I_syn_ii), 5000*sizeof(_array_neurongroup_1_I_syn_ii[0]));
		outfile__array_neurongroup_1_I_syn_ii.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_I_syn_ii." << endl;
	}
	ofstream outfile__array_neurongroup_1_lastspike;
	outfile__array_neurongroup_1_lastspike.open("results/_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_lastspike.is_open())
	{
		outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 5000*sizeof(_array_neurongroup_1_lastspike[0]));
		outfile__array_neurongroup_1_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_1_not_refractory;
	outfile__array_neurongroup_1_not_refractory.open("results/_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_not_refractory.is_open())
	{
		outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 5000*sizeof(_array_neurongroup_1_not_refractory[0]));
		outfile__array_neurongroup_1_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_1_R;
	outfile__array_neurongroup_1_R.open("results/_array_neurongroup_1_R_1779030929", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_R.is_open())
	{
		outfile__array_neurongroup_1_R.write(reinterpret_cast<char*>(_array_neurongroup_1_R), 5000*sizeof(_array_neurongroup_1_R[0]));
		outfile__array_neurongroup_1_R.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_R." << endl;
	}
	ofstream outfile__array_neurongroup_1_subgroup_1__sub_idx;
	outfile__array_neurongroup_1_subgroup_1__sub_idx.open("results/_array_neurongroup_1_subgroup_1__sub_idx_1584208906", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_subgroup_1__sub_idx.is_open())
	{
		outfile__array_neurongroup_1_subgroup_1__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_1__sub_idx), 1000*sizeof(_array_neurongroup_1_subgroup_1__sub_idx[0]));
		outfile__array_neurongroup_1_subgroup_1__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_subgroup_1__sub_idx." << endl;
	}
	ofstream outfile__array_neurongroup_1_subgroup__sub_idx;
	outfile__array_neurongroup_1_subgroup__sub_idx.open("results/_array_neurongroup_1_subgroup__sub_idx_1166957185", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_subgroup__sub_idx.is_open())
	{
		outfile__array_neurongroup_1_subgroup__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup__sub_idx), 4000*sizeof(_array_neurongroup_1_subgroup__sub_idx[0]));
		outfile__array_neurongroup_1_subgroup__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_subgroup__sub_idx." << endl;
	}
	ofstream outfile__array_neurongroup_1_tau_mem;
	outfile__array_neurongroup_1_tau_mem.open("results/_array_neurongroup_1_tau_mem_126505239", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_tau_mem.is_open())
	{
		outfile__array_neurongroup_1_tau_mem.write(reinterpret_cast<char*>(_array_neurongroup_1_tau_mem), 5000*sizeof(_array_neurongroup_1_tau_mem[0]));
		outfile__array_neurongroup_1_tau_mem.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_tau_mem." << endl;
	}
	ofstream outfile__array_neurongroup_1_v;
	outfile__array_neurongroup_1_v.open("results/_array_neurongroup_1_v_1443512128", ios::binary | ios::out);
	if(outfile__array_neurongroup_1_v.is_open())
	{
		outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 5000*sizeof(_array_neurongroup_1_v[0]));
		outfile__array_neurongroup_1_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
	}
	ofstream outfile__array_spikemonitor_2__source_idx;
	outfile__array_spikemonitor_2__source_idx.open("results/_array_spikemonitor_2__source_idx_1793786228", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2__source_idx.is_open())
	{
		outfile__array_spikemonitor_2__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_2__source_idx), 4000*sizeof(_array_spikemonitor_2__source_idx[0]));
		outfile__array_spikemonitor_2__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_2_count;
	outfile__array_spikemonitor_2_count.open("results/_array_spikemonitor_2_count_3621222387", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2_count.is_open())
	{
		outfile__array_spikemonitor_2_count.write(reinterpret_cast<char*>(_array_spikemonitor_2_count), 4000*sizeof(_array_spikemonitor_2_count[0]));
		outfile__array_spikemonitor_2_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2_count." << endl;
	}
	ofstream outfile__array_spikemonitor_2_N;
	outfile__array_spikemonitor_2_N.open("results/_array_spikemonitor_2_N_2352936276", ios::binary | ios::out);
	if(outfile__array_spikemonitor_2_N.is_open())
	{
		outfile__array_spikemonitor_2_N.write(reinterpret_cast<char*>(_array_spikemonitor_2_N), 1*sizeof(_array_spikemonitor_2_N[0]));
		outfile__array_spikemonitor_2_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_2_N." << endl;
	}
	ofstream outfile__array_spikemonitor_3__source_idx;
	outfile__array_spikemonitor_3__source_idx.open("results/_array_spikemonitor_3__source_idx_3078478065", ios::binary | ios::out);
	if(outfile__array_spikemonitor_3__source_idx.is_open())
	{
		outfile__array_spikemonitor_3__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_3__source_idx), 1000*sizeof(_array_spikemonitor_3__source_idx[0]));
		outfile__array_spikemonitor_3__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_3__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_3_count;
	outfile__array_spikemonitor_3_count.open("results/_array_spikemonitor_3_count_1906342983", ios::binary | ios::out);
	if(outfile__array_spikemonitor_3_count.is_open())
	{
		outfile__array_spikemonitor_3_count.write(reinterpret_cast<char*>(_array_spikemonitor_3_count), 1000*sizeof(_array_spikemonitor_3_count[0]));
		outfile__array_spikemonitor_3_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_3_count." << endl;
	}
	ofstream outfile__array_spikemonitor_3_N;
	outfile__array_spikemonitor_3_N.open("results/_array_spikemonitor_3_N_2382143331", ios::binary | ios::out);
	if(outfile__array_spikemonitor_3_N.is_open())
	{
		outfile__array_spikemonitor_3_N.write(reinterpret_cast<char*>(_array_spikemonitor_3_N), 1*sizeof(_array_spikemonitor_3_N[0]));
		outfile__array_spikemonitor_3_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_3_N." << endl;
	}
	ofstream outfile__array_synapses_4_N;
	outfile__array_synapses_4_N.open("results/_array_synapses_4_N_1867624580", ios::binary | ios::out);
	if(outfile__array_synapses_4_N.is_open())
	{
		outfile__array_synapses_4_N.write(reinterpret_cast<char*>(_array_synapses_4_N), 1*sizeof(_array_synapses_4_N[0]));
		outfile__array_synapses_4_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_4_N." << endl;
	}
	ofstream outfile__array_synapses_5_N;
	outfile__array_synapses_5_N.open("results/_array_synapses_5_N_1855183539", ios::binary | ios::out);
	if(outfile__array_synapses_5_N.is_open())
	{
		outfile__array_synapses_5_N.write(reinterpret_cast<char*>(_array_synapses_5_N), 1*sizeof(_array_synapses_5_N[0]));
		outfile__array_synapses_5_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_5_N." << endl;
	}
	ofstream outfile__array_synapses_6_N;
	outfile__array_synapses_6_N.open("results/_array_synapses_6_N_1825924330", ios::binary | ios::out);
	if(outfile__array_synapses_6_N.is_open())
	{
		outfile__array_synapses_6_N.write(reinterpret_cast<char*>(_array_synapses_6_N), 1*sizeof(_array_synapses_6_N[0]));
		outfile__array_synapses_6_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_6_N." << endl;
	}
	ofstream outfile__array_synapses_7_N;
	outfile__array_synapses_7_N.open("results/_array_synapses_7_N_1830227677", ios::binary | ios::out);
	if(outfile__array_synapses_7_N.is_open())
	{
		outfile__array_synapses_7_N.write(reinterpret_cast<char*>(_array_synapses_7_N), 1*sizeof(_array_synapses_7_N[0]));
		outfile__array_synapses_7_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_7_N." << endl;
	}

	ofstream outfile__dynamic_array_spikemonitor_2_i;
	outfile__dynamic_array_spikemonitor_2_i.open("results/_dynamic_array_spikemonitor_2_i_2642822512", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_2_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_2_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_2_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_2_i[0]), _dynamic_array_spikemonitor_2_i.size()*sizeof(_dynamic_array_spikemonitor_2_i[0]));
		    outfile__dynamic_array_spikemonitor_2_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_2_t;
	outfile__dynamic_array_spikemonitor_2_t.open("results/_dynamic_array_spikemonitor_2_t_4269812137", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_2_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_2_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_2_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_2_t[0]), _dynamic_array_spikemonitor_2_t.size()*sizeof(_dynamic_array_spikemonitor_2_t[0]));
		    outfile__dynamic_array_spikemonitor_2_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_2_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_3_i;
	outfile__dynamic_array_spikemonitor_3_i.open("results/_dynamic_array_spikemonitor_3_i_2621714247", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_3_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_3_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_3_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_3_i[0]), _dynamic_array_spikemonitor_3_i.size()*sizeof(_dynamic_array_spikemonitor_3_i[0]));
		    outfile__dynamic_array_spikemonitor_3_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_3_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_3_t;
	outfile__dynamic_array_spikemonitor_3_t.open("results/_dynamic_array_spikemonitor_3_t_4282532766", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_3_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_3_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_3_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_3_t[0]), _dynamic_array_spikemonitor_3_t.size()*sizeof(_dynamic_array_spikemonitor_3_t[0]));
		    outfile__dynamic_array_spikemonitor_3_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_3_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4__synaptic_post;
	outfile__dynamic_array_synapses_4__synaptic_post.open("results/_dynamic_array_synapses_4__synaptic_post_225617685", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_4__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_4__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4__synaptic_post[0]), _dynamic_array_synapses_4__synaptic_post.size()*sizeof(_dynamic_array_synapses_4__synaptic_post[0]));
		    outfile__dynamic_array_synapses_4__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4__synaptic_pre;
	outfile__dynamic_array_synapses_4__synaptic_pre.open("results/_dynamic_array_synapses_4__synaptic_pre_455049877", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_4__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_4__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4__synaptic_pre[0]), _dynamic_array_synapses_4__synaptic_pre.size()*sizeof(_dynamic_array_synapses_4__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_4__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_A_SE;
	outfile__dynamic_array_synapses_4_A_SE.open("results/_dynamic_array_synapses_4_A_SE_4039909204", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_A_SE.is_open())
	{
        if (! _dynamic_array_synapses_4_A_SE.empty() )
        {
			outfile__dynamic_array_synapses_4_A_SE.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_A_SE[0]), _dynamic_array_synapses_4_A_SE.size()*sizeof(_dynamic_array_synapses_4_A_SE[0]));
		    outfile__dynamic_array_synapses_4_A_SE.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_A_SE." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_delay;
	outfile__dynamic_array_synapses_4_delay.open("results/_dynamic_array_synapses_4_delay_3745875037", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_delay.is_open())
	{
        if (! _dynamic_array_synapses_4_delay.empty() )
        {
			outfile__dynamic_array_synapses_4_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_delay[0]), _dynamic_array_synapses_4_delay.size()*sizeof(_dynamic_array_synapses_4_delay[0]));
		    outfile__dynamic_array_synapses_4_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_N_incoming;
	outfile__dynamic_array_synapses_4_N_incoming.open("results/_dynamic_array_synapses_4_N_incoming_1450066154", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_4_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_4_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_N_incoming[0]), _dynamic_array_synapses_4_N_incoming.size()*sizeof(_dynamic_array_synapses_4_N_incoming[0]));
		    outfile__dynamic_array_synapses_4_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_N_outgoing;
	outfile__dynamic_array_synapses_4_N_outgoing.open("results/_dynamic_array_synapses_4_N_outgoing_1903308848", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_4_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_4_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_N_outgoing[0]), _dynamic_array_synapses_4_N_outgoing.size()*sizeof(_dynamic_array_synapses_4_N_outgoing[0]));
		    outfile__dynamic_array_synapses_4_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_tau_inact;
	outfile__dynamic_array_synapses_4_tau_inact.open("results/_dynamic_array_synapses_4_tau_inact_1786054549", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_tau_inact.is_open())
	{
        if (! _dynamic_array_synapses_4_tau_inact.empty() )
        {
			outfile__dynamic_array_synapses_4_tau_inact.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_tau_inact[0]), _dynamic_array_synapses_4_tau_inact.size()*sizeof(_dynamic_array_synapses_4_tau_inact[0]));
		    outfile__dynamic_array_synapses_4_tau_inact.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_tau_inact." << endl;
	}
	ofstream outfile__dynamic_array_synapses_4_y;
	outfile__dynamic_array_synapses_4_y.open("results/_dynamic_array_synapses_4_y_2411725866", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_4_y.is_open())
	{
        if (! _dynamic_array_synapses_4_y.empty() )
        {
			outfile__dynamic_array_synapses_4_y.write(reinterpret_cast<char*>(&_dynamic_array_synapses_4_y[0]), _dynamic_array_synapses_4_y.size()*sizeof(_dynamic_array_synapses_4_y[0]));
		    outfile__dynamic_array_synapses_4_y.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_4_y." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5__synaptic_post;
	outfile__dynamic_array_synapses_5__synaptic_post.open("results/_dynamic_array_synapses_5__synaptic_post_2736404100", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_5__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_5__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5__synaptic_post[0]), _dynamic_array_synapses_5__synaptic_post.size()*sizeof(_dynamic_array_synapses_5__synaptic_post[0]));
		    outfile__dynamic_array_synapses_5__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5__synaptic_pre;
	outfile__dynamic_array_synapses_5__synaptic_pre.open("results/_dynamic_array_synapses_5__synaptic_pre_2732874109", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_5__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_5__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5__synaptic_pre[0]), _dynamic_array_synapses_5__synaptic_pre.size()*sizeof(_dynamic_array_synapses_5__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_5__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_A_SE;
	outfile__dynamic_array_synapses_5_A_SE.open("results/_dynamic_array_synapses_5_A_SE_999345393", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_A_SE.is_open())
	{
        if (! _dynamic_array_synapses_5_A_SE.empty() )
        {
			outfile__dynamic_array_synapses_5_A_SE.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_A_SE[0]), _dynamic_array_synapses_5_A_SE.size()*sizeof(_dynamic_array_synapses_5_A_SE[0]));
		    outfile__dynamic_array_synapses_5_A_SE.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_A_SE." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_delay;
	outfile__dynamic_array_synapses_5_delay.open("results/_dynamic_array_synapses_5_delay_2033356777", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_delay.is_open())
	{
        if (! _dynamic_array_synapses_5_delay.empty() )
        {
			outfile__dynamic_array_synapses_5_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_delay[0]), _dynamic_array_synapses_5_delay.size()*sizeof(_dynamic_array_synapses_5_delay[0]));
		    outfile__dynamic_array_synapses_5_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_N_incoming;
	outfile__dynamic_array_synapses_5_N_incoming.open("results/_dynamic_array_synapses_5_N_incoming_3452636293", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_5_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_5_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_N_incoming[0]), _dynamic_array_synapses_5_N_incoming.size()*sizeof(_dynamic_array_synapses_5_N_incoming[0]));
		    outfile__dynamic_array_synapses_5_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_N_outgoing;
	outfile__dynamic_array_synapses_5_N_outgoing.open("results/_dynamic_array_synapses_5_N_outgoing_3939990623", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_5_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_5_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_N_outgoing[0]), _dynamic_array_synapses_5_N_outgoing.size()*sizeof(_dynamic_array_synapses_5_N_outgoing[0]));
		    outfile__dynamic_array_synapses_5_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_tau_inact;
	outfile__dynamic_array_synapses_5_tau_inact.open("results/_dynamic_array_synapses_5_tau_inact_2885408853", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_tau_inact.is_open())
	{
        if (! _dynamic_array_synapses_5_tau_inact.empty() )
        {
			outfile__dynamic_array_synapses_5_tau_inact.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_tau_inact[0]), _dynamic_array_synapses_5_tau_inact.size()*sizeof(_dynamic_array_synapses_5_tau_inact[0]));
		    outfile__dynamic_array_synapses_5_tau_inact.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_tau_inact." << endl;
	}
	ofstream outfile__dynamic_array_synapses_5_y;
	outfile__dynamic_array_synapses_5_y.open("results/_dynamic_array_synapses_5_y_2382523933", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_5_y.is_open())
	{
        if (! _dynamic_array_synapses_5_y.empty() )
        {
			outfile__dynamic_array_synapses_5_y.write(reinterpret_cast<char*>(&_dynamic_array_synapses_5_y[0]), _dynamic_array_synapses_5_y.size()*sizeof(_dynamic_array_synapses_5_y[0]));
		    outfile__dynamic_array_synapses_5_y.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_5_y." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6__synaptic_post;
	outfile__dynamic_array_synapses_6__synaptic_post.open("results/_dynamic_array_synapses_6__synaptic_post_2329051766", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_6__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_6__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6__synaptic_post[0]), _dynamic_array_synapses_6__synaptic_post.size()*sizeof(_dynamic_array_synapses_6__synaptic_post[0]));
		    outfile__dynamic_array_synapses_6__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6__synaptic_pre;
	outfile__dynamic_array_synapses_6__synaptic_pre.open("results/_dynamic_array_synapses_6__synaptic_pre_3013161732", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_6__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_6__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6__synaptic_pre[0]), _dynamic_array_synapses_6__synaptic_pre.size()*sizeof(_dynamic_array_synapses_6__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_6__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_A_SE;
	outfile__dynamic_array_synapses_6_A_SE.open("results/_dynamic_array_synapses_6_A_SE_3171204703", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_A_SE.is_open())
	{
        if (! _dynamic_array_synapses_6_A_SE.empty() )
        {
			outfile__dynamic_array_synapses_6_A_SE.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_A_SE[0]), _dynamic_array_synapses_6_A_SE.size()*sizeof(_dynamic_array_synapses_6_A_SE[0]));
		    outfile__dynamic_array_synapses_6_A_SE.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_A_SE." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_delay;
	outfile__dynamic_array_synapses_6_delay.open("results/_dynamic_array_synapses_6_delay_1222284660", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_delay.is_open())
	{
        if (! _dynamic_array_synapses_6_delay.empty() )
        {
			outfile__dynamic_array_synapses_6_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_delay[0]), _dynamic_array_synapses_6_delay.size()*sizeof(_dynamic_array_synapses_6_delay[0]));
		    outfile__dynamic_array_synapses_6_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_N_incoming;
	outfile__dynamic_array_synapses_6_N_incoming.open("results/_dynamic_array_synapses_6_N_incoming_3126189685", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_6_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_6_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_N_incoming[0]), _dynamic_array_synapses_6_N_incoming.size()*sizeof(_dynamic_array_synapses_6_N_incoming[0]));
		    outfile__dynamic_array_synapses_6_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_N_outgoing;
	outfile__dynamic_array_synapses_6_N_outgoing.open("results/_dynamic_array_synapses_6_N_outgoing_2638851759", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_6_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_6_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_N_outgoing[0]), _dynamic_array_synapses_6_N_outgoing.size()*sizeof(_dynamic_array_synapses_6_N_outgoing[0]));
		    outfile__dynamic_array_synapses_6_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_tau_inact;
	outfile__dynamic_array_synapses_6_tau_inact.open("results/_dynamic_array_synapses_6_tau_inact_840547924", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_tau_inact.is_open())
	{
        if (! _dynamic_array_synapses_6_tau_inact.empty() )
        {
			outfile__dynamic_array_synapses_6_tau_inact.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_tau_inact[0]), _dynamic_array_synapses_6_tau_inact.size()*sizeof(_dynamic_array_synapses_6_tau_inact[0]));
		    outfile__dynamic_array_synapses_6_tau_inact.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_tau_inact." << endl;
	}
	ofstream outfile__dynamic_array_synapses_6_y;
	outfile__dynamic_array_synapses_6_y.open("results/_dynamic_array_synapses_6_y_2353320004", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_6_y.is_open())
	{
        if (! _dynamic_array_synapses_6_y.empty() )
        {
			outfile__dynamic_array_synapses_6_y.write(reinterpret_cast<char*>(&_dynamic_array_synapses_6_y[0]), _dynamic_array_synapses_6_y.size()*sizeof(_dynamic_array_synapses_6_y[0]));
		    outfile__dynamic_array_synapses_6_y.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_6_y." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7__synaptic_post;
	outfile__dynamic_array_synapses_7__synaptic_post.open("results/_dynamic_array_synapses_7__synaptic_post_616174567", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_7__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_7__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7__synaptic_post[0]), _dynamic_array_synapses_7__synaptic_post.size()*sizeof(_dynamic_array_synapses_7__synaptic_post[0]));
		    outfile__dynamic_array_synapses_7__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7__synaptic_pre;
	outfile__dynamic_array_synapses_7__synaptic_pre.open("results/_dynamic_array_synapses_7__synaptic_pre_174254316", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_7__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_7__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7__synaptic_pre[0]), _dynamic_array_synapses_7__synaptic_pre.size()*sizeof(_dynamic_array_synapses_7__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_7__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_A_SE;
	outfile__dynamic_array_synapses_7_A_SE.open("results/_dynamic_array_synapses_7_A_SE_1985506810", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_A_SE.is_open())
	{
        if (! _dynamic_array_synapses_7_A_SE.empty() )
        {
			outfile__dynamic_array_synapses_7_A_SE.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_A_SE[0]), _dynamic_array_synapses_7_A_SE.size()*sizeof(_dynamic_array_synapses_7_A_SE[0]));
		    outfile__dynamic_array_synapses_7_A_SE.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_A_SE." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_delay;
	outfile__dynamic_array_synapses_7_delay.open("results/_dynamic_array_synapses_7_delay_4004355776", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_delay.is_open())
	{
        if (! _dynamic_array_synapses_7_delay.empty() )
        {
			outfile__dynamic_array_synapses_7_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_delay[0]), _dynamic_array_synapses_7_delay.size()*sizeof(_dynamic_array_synapses_7_delay[0]));
		    outfile__dynamic_array_synapses_7_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_N_incoming;
	outfile__dynamic_array_synapses_7_N_incoming.open("results/_dynamic_array_synapses_7_N_incoming_569414170", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_7_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_7_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_N_incoming[0]), _dynamic_array_synapses_7_N_incoming.size()*sizeof(_dynamic_array_synapses_7_N_incoming[0]));
		    outfile__dynamic_array_synapses_7_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_N_outgoing;
	outfile__dynamic_array_synapses_7_N_outgoing.open("results/_dynamic_array_synapses_7_N_outgoing_116187840", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_7_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_7_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_N_outgoing[0]), _dynamic_array_synapses_7_N_outgoing.size()*sizeof(_dynamic_array_synapses_7_N_outgoing[0]));
		    outfile__dynamic_array_synapses_7_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_tau_inact;
	outfile__dynamic_array_synapses_7_tau_inact.open("results/_dynamic_array_synapses_7_tau_inact_4086784404", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_tau_inact.is_open())
	{
        if (! _dynamic_array_synapses_7_tau_inact.empty() )
        {
			outfile__dynamic_array_synapses_7_tau_inact.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_tau_inact[0]), _dynamic_array_synapses_7_tau_inact.size()*sizeof(_dynamic_array_synapses_7_tau_inact[0]));
		    outfile__dynamic_array_synapses_7_tau_inact.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_tau_inact." << endl;
	}
	ofstream outfile__dynamic_array_synapses_7_y;
	outfile__dynamic_array_synapses_7_y.open("results/_dynamic_array_synapses_7_y_2374417011", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_7_y.is_open())
	{
        if (! _dynamic_array_synapses_7_y.empty() )
        {
			outfile__dynamic_array_synapses_7_y.write(reinterpret_cast<char*>(&_dynamic_array_synapses_7_y[0]), _dynamic_array_synapses_7_y.size()*sizeof(_dynamic_array_synapses_7_y[0]));
		    outfile__dynamic_array_synapses_7_y.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_7_y." << endl;
	}

	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
}

