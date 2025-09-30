

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread. We also create a distribution for
// each thread, even though this is not strictly necessary for the uniform distribution, as
// the distribution is stateless.
std::vector< RandomGenerator > _random_generators;

//////////////// networks /////////////////
Network magicnetwork;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    if (name == "neurongroup_1._spikespace") {
        var_size = 10001;
        data_size = 10001*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup_1__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.lastspike") {
        var_size = 10000;
        data_size = 10000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.not_refractory") {
        var_size = 10000;
        data_size = 10000*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_1_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.v") {
        var_size = 10000;
        data_size = 10000*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_v, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses_2.delay") {
        var_size = _dynamic_array_synapses_2_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_2_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_2_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 10001;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 10000;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 10000;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 10000;
int32_t * _array_neurongroup_1_subgroup_1__sub_idx;
const int _num__array_neurongroup_1_subgroup_1__sub_idx = 5000;
int32_t * _array_neurongroup_1_subgroup_2__sub_idx;
const int _num__array_neurongroup_1_subgroup_2__sub_idx = 25;
int32_t * _array_neurongroup_1_subgroup_3__sub_idx;
const int _num__array_neurongroup_1_subgroup_3__sub_idx = 25;
int32_t * _array_neurongroup_1_subgroup_4__sub_idx;
const int _num__array_neurongroup_1_subgroup_4__sub_idx = 25;
int32_t * _array_neurongroup_1_subgroup_5__sub_idx;
const int _num__array_neurongroup_1_subgroup_5__sub_idx = 25;
int32_t * _array_neurongroup_1_subgroup__sub_idx;
const int _num__array_neurongroup_1_subgroup__sub_idx = 5000;
double * _array_neurongroup_1_v;
const int _num__array_neurongroup_1_v = 10000;
int32_t * _array_ratemonitor_N;
const int _num__array_ratemonitor_N = 1;
int32_t * _array_spikemonitor_1__source_idx;
const int _num__array_spikemonitor_1__source_idx = 25;
int32_t * _array_spikemonitor_1_count;
const int _num__array_spikemonitor_1_count = 25;
int32_t * _array_spikemonitor_1_N;
const int _num__array_spikemonitor_1_N = 1;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 25;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 25;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_statemonitor_1__indices;
const int _num__array_statemonitor_1__indices = 25;
double * _array_statemonitor_1_clock_1_dt;
const int _num__array_statemonitor_1_clock_1_dt = 1;
double * _array_statemonitor_1_clock_1_t;
const int _num__array_statemonitor_1_clock_1_t = 1;
int64_t * _array_statemonitor_1_clock_1_timestep;
const int _num__array_statemonitor_1_clock_1_timestep = 1;
int32_t * _array_statemonitor_1_N;
const int _num__array_statemonitor_1_N = 1;
double * _array_statemonitor_1_v;
const int _num__array_statemonitor_1_v = (0, 25);
int32_t * _array_statemonitor__indices;
const int _num__array_statemonitor__indices = 25;
double * _array_statemonitor_clock_1_dt;
const int _num__array_statemonitor_clock_1_dt = 1;
double * _array_statemonitor_clock_1_t;
const int _num__array_statemonitor_clock_1_t = 1;
int64_t * _array_statemonitor_clock_1_timestep;
const int _num__array_statemonitor_clock_1_timestep = 1;
int32_t * _array_statemonitor_N;
const int _num__array_statemonitor_N = 1;
double * _array_statemonitor_v;
const int _num__array_statemonitor_v = (0, 25);
int32_t * _array_synapses_2_N;
const int _num__array_synapses_2_N = 1;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<double> _dynamic_array_ratemonitor_rate;
std::vector<double> _dynamic_array_ratemonitor_t;
std::vector<int32_t> _dynamic_array_spikemonitor_1_i;
std::vector<double> _dynamic_array_spikemonitor_1_t;
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<double> _dynamic_array_statemonitor_1_t;
std::vector<double> _dynamic_array_statemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_2__synaptic_pre;
std::vector<double> _dynamic_array_synapses_2_delay;
std::vector<int32_t> _dynamic_array_synapses_2_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_2_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_statemonitor_1_v;
DynamicArray2D<double> _dynamic_array_statemonitor_v;

/////////////// static arrays /////////////
int32_t * _static_array__array_statemonitor_1__indices;
const int _num__static_array__array_statemonitor_1__indices = 25;
int32_t * _static_array__array_statemonitor__indices;
const int _num__static_array__array_statemonitor__indices = 25;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_pre(
    _dynamic_array_synapses__synaptic_pre,
    5000, 10000);
// synapses_2
SynapticPathway synapses_2_pre(
    _dynamic_array_synapses_2__synaptic_pre,
    0, 5000);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp
Clock statemonitor_1_clock_1;  // attributes will be set in run.cpp
Clock statemonitor_clock_1;  // attributes will be set in run.cpp

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

    _array_neurongroup_1__spikespace = new int32_t[10001];
    
    for(int i=0; i<10001; i++) _array_neurongroup_1__spikespace[i] = 0;

    _array_neurongroup_1_i = new int32_t[10000];
    
    for(int i=0; i<10000; i++) _array_neurongroup_1_i[i] = 0;

    _array_neurongroup_1_lastspike = new double[10000];
    
    for(int i=0; i<10000; i++) _array_neurongroup_1_lastspike[i] = 0;

    _array_neurongroup_1_not_refractory = new char[10000];
    
    for(int i=0; i<10000; i++) _array_neurongroup_1_not_refractory[i] = 0;

    _array_neurongroup_1_subgroup_1__sub_idx = new int32_t[5000];
    
    for(int i=0; i<5000; i++) _array_neurongroup_1_subgroup_1__sub_idx[i] = 0;

    _array_neurongroup_1_subgroup_2__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_2__sub_idx[i] = 0;

    _array_neurongroup_1_subgroup_3__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_3__sub_idx[i] = 0;

    _array_neurongroup_1_subgroup_4__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_4__sub_idx[i] = 0;

    _array_neurongroup_1_subgroup_5__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_5__sub_idx[i] = 0;

    _array_neurongroup_1_subgroup__sub_idx = new int32_t[5000];
    
    for(int i=0; i<5000; i++) _array_neurongroup_1_subgroup__sub_idx[i] = 0;

    _array_neurongroup_1_v = new double[10000];
    
    for(int i=0; i<10000; i++) _array_neurongroup_1_v[i] = 0;

    _array_ratemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_ratemonitor_N[i] = 0;

    _array_spikemonitor_1__source_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor_1__source_idx[i] = 0;

    _array_spikemonitor_1_count = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor_1_count[i] = 0;

    _array_spikemonitor_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikemonitor_1_N[i] = 0;

    _array_spikemonitor__source_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor__source_idx[i] = 0;

    _array_spikemonitor_count = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor_count[i] = 0;

    _array_spikemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

    _array_statemonitor_1__indices = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_statemonitor_1__indices[i] = 0;

    _array_statemonitor_1_clock_1_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_1_clock_1_dt[i] = 0;

    _array_statemonitor_1_clock_1_t = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_1_clock_1_t[i] = 0;

    _array_statemonitor_1_clock_1_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_1_clock_1_timestep[i] = 0;

    _array_statemonitor_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_1_N[i] = 0;

    _array_statemonitor__indices = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_statemonitor__indices[i] = 0;

    _array_statemonitor_clock_1_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_1_dt[i] = 0;

    _array_statemonitor_clock_1_t = new double[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_1_t[i] = 0;

    _array_statemonitor_clock_1_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_clock_1_timestep[i] = 0;

    _array_statemonitor_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;

    _array_synapses_2_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_2_N[i] = 0;

    _array_synapses_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_N[i] = 0;

    _dynamic_array_synapses_2_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_2_delay[i] = 0;

    _dynamic_array_synapses_delay.resize(1);
    
    for(int i=0; i<1; i++) _dynamic_array_synapses_delay[i] = 0;


    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[10000];
    
    for(int i=0; i<10000; i++) _array_neurongroup_1_i[i] = 0 + i;

    _array_neurongroup_1_subgroup_1__sub_idx = new int32_t[5000];
    
    for(int i=0; i<5000; i++) _array_neurongroup_1_subgroup_1__sub_idx[i] = 5000 + i;

    _array_neurongroup_1_subgroup_2__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_2__sub_idx[i] = 0 + i;

    _array_neurongroup_1_subgroup_3__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_3__sub_idx[i] = 0 + i;

    _array_neurongroup_1_subgroup_4__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_4__sub_idx[i] = 5000 + i;

    _array_neurongroup_1_subgroup_5__sub_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_neurongroup_1_subgroup_5__sub_idx[i] = 5000 + i;

    _array_neurongroup_1_subgroup__sub_idx = new int32_t[5000];
    
    for(int i=0; i<5000; i++) _array_neurongroup_1_subgroup__sub_idx[i] = 0 + i;

    _array_spikemonitor_1__source_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor_1__source_idx[i] = 0 + i;

    _array_spikemonitor__source_idx = new int32_t[25];
    
    for(int i=0; i<25; i++) _array_spikemonitor__source_idx[i] = 0 + i;


    // static arrays
    _static_array__array_statemonitor_1__indices = new int32_t[25];
    _static_array__array_statemonitor__indices = new int32_t[25];

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<1; i++)
        _random_generators.push_back(RandomGenerator());
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_statemonitor_1__indices;
    f_static_array__array_statemonitor_1__indices.open("static_arrays/_static_array__array_statemonitor_1__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor_1__indices.is_open())
    {
        f_static_array__array_statemonitor_1__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor_1__indices), 25*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor_1__indices." << endl;
    }
    ifstream f_static_array__array_statemonitor__indices;
    f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
    if(f_static_array__array_statemonitor__indices.is_open())
    {
        f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 25*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
    }
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_neurongroup_1__spikespace;
    outfile__array_neurongroup_1__spikespace.open(results_dir + "_array_neurongroup_1__spikespace_3155027917", ios::binary | ios::out);
    if(outfile__array_neurongroup_1__spikespace.is_open())
    {
        outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 10001*sizeof(_array_neurongroup_1__spikespace[0]));
        outfile__array_neurongroup_1__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open(results_dir + "_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 10000*sizeof(_array_neurongroup_1_i[0]));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open(results_dir + "_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 10000*sizeof(_array_neurongroup_1_lastspike[0]));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open(results_dir + "_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 10000*sizeof(_array_neurongroup_1_not_refractory[0]));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup_1__sub_idx;
    outfile__array_neurongroup_1_subgroup_1__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup_1__sub_idx_1584208906", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup_1__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup_1__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_1__sub_idx), 5000*sizeof(_array_neurongroup_1_subgroup_1__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup_1__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup_1__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup_2__sub_idx;
    outfile__array_neurongroup_1_subgroup_2__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup_2__sub_idx_3042617097", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup_2__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup_2__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_2__sub_idx), 25*sizeof(_array_neurongroup_1_subgroup_2__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup_2__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup_2__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup_3__sub_idx;
    outfile__array_neurongroup_1_subgroup_3__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup_3__sub_idx_1519963191", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup_3__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup_3__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_3__sub_idx), 25*sizeof(_array_neurongroup_1_subgroup_3__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup_3__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup_3__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup_4__sub_idx;
    outfile__array_neurongroup_1_subgroup_4__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup_4__sub_idx_3091519310", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup_4__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup_4__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_4__sub_idx), 25*sizeof(_array_neurongroup_1_subgroup_4__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup_4__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup_4__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup_5__sub_idx;
    outfile__array_neurongroup_1_subgroup_5__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup_5__sub_idx_1468447856", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup_5__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup_5__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup_5__sub_idx), 25*sizeof(_array_neurongroup_1_subgroup_5__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup_5__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup_5__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_subgroup__sub_idx;
    outfile__array_neurongroup_1_subgroup__sub_idx.open(results_dir + "_array_neurongroup_1_subgroup__sub_idx_1166957185", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_subgroup__sub_idx.is_open())
    {
        outfile__array_neurongroup_1_subgroup__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_1_subgroup__sub_idx), 5000*sizeof(_array_neurongroup_1_subgroup__sub_idx[0]));
        outfile__array_neurongroup_1_subgroup__sub_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_subgroup__sub_idx." << endl;
    }
    ofstream outfile__array_neurongroup_1_v;
    outfile__array_neurongroup_1_v.open(results_dir + "_array_neurongroup_1_v_1443512128", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_v.is_open())
    {
        outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 10000*sizeof(_array_neurongroup_1_v[0]));
        outfile__array_neurongroup_1_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
    }
    ofstream outfile__array_ratemonitor_N;
    outfile__array_ratemonitor_N.open(results_dir + "_array_ratemonitor_N_611090289", ios::binary | ios::out);
    if(outfile__array_ratemonitor_N.is_open())
    {
        outfile__array_ratemonitor_N.write(reinterpret_cast<char*>(_array_ratemonitor_N), 1*sizeof(_array_ratemonitor_N[0]));
        outfile__array_ratemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_ratemonitor_N." << endl;
    }
    ofstream outfile__array_spikemonitor_1__source_idx;
    outfile__array_spikemonitor_1__source_idx.open(results_dir + "_array_spikemonitor_1__source_idx_3609292218", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1__source_idx.is_open())
    {
        outfile__array_spikemonitor_1__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor_1__source_idx), 25*sizeof(_array_spikemonitor_1__source_idx[0]));
        outfile__array_spikemonitor_1__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1__source_idx." << endl;
    }
    ofstream outfile__array_spikemonitor_1_count;
    outfile__array_spikemonitor_1_count.open(results_dir + "_array_spikemonitor_1_count_3862916462", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_count.is_open())
    {
        outfile__array_spikemonitor_1_count.write(reinterpret_cast<char*>(_array_spikemonitor_1_count), 25*sizeof(_array_spikemonitor_1_count[0]));
        outfile__array_spikemonitor_1_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_count." << endl;
    }
    ofstream outfile__array_spikemonitor_1_N;
    outfile__array_spikemonitor_1_N.open(results_dir + "_array_spikemonitor_1_N_2390248205", ios::binary | ios::out);
    if(outfile__array_spikemonitor_1_N.is_open())
    {
        outfile__array_spikemonitor_1_N.write(reinterpret_cast<char*>(_array_spikemonitor_1_N), 1*sizeof(_array_spikemonitor_1_N[0]));
        outfile__array_spikemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_1_N." << endl;
    }
    ofstream outfile__array_spikemonitor__source_idx;
    outfile__array_spikemonitor__source_idx.open(results_dir + "_array_spikemonitor__source_idx_1477951789", ios::binary | ios::out);
    if(outfile__array_spikemonitor__source_idx.is_open())
    {
        outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 25*sizeof(_array_spikemonitor__source_idx[0]));
        outfile__array_spikemonitor__source_idx.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
    }
    ofstream outfile__array_spikemonitor_count;
    outfile__array_spikemonitor_count.open(results_dir + "_array_spikemonitor_count_598337445", ios::binary | ios::out);
    if(outfile__array_spikemonitor_count.is_open())
    {
        outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 25*sizeof(_array_spikemonitor_count[0]));
        outfile__array_spikemonitor_count.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
    }
    ofstream outfile__array_spikemonitor_N;
    outfile__array_spikemonitor_N.open(results_dir + "_array_spikemonitor_N_225734567", ios::binary | ios::out);
    if(outfile__array_spikemonitor_N.is_open())
    {
        outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
        outfile__array_spikemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
    }
    ofstream outfile__array_statemonitor_1__indices;
    outfile__array_statemonitor_1__indices.open(results_dir + "_array_statemonitor_1__indices_2504039125", ios::binary | ios::out);
    if(outfile__array_statemonitor_1__indices.is_open())
    {
        outfile__array_statemonitor_1__indices.write(reinterpret_cast<char*>(_array_statemonitor_1__indices), 25*sizeof(_array_statemonitor_1__indices[0]));
        outfile__array_statemonitor_1__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1__indices." << endl;
    }
    ofstream outfile__array_statemonitor_1_clock_1_dt;
    outfile__array_statemonitor_1_clock_1_dt.open(results_dir + "_array_statemonitor_1_clock_1_dt_2529238317", ios::binary | ios::out);
    if(outfile__array_statemonitor_1_clock_1_dt.is_open())
    {
        outfile__array_statemonitor_1_clock_1_dt.write(reinterpret_cast<char*>(_array_statemonitor_1_clock_1_dt), 1*sizeof(_array_statemonitor_1_clock_1_dt[0]));
        outfile__array_statemonitor_1_clock_1_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1_clock_1_dt." << endl;
    }
    ofstream outfile__array_statemonitor_1_clock_1_t;
    outfile__array_statemonitor_1_clock_1_t.open(results_dir + "_array_statemonitor_1_clock_1_t_3664298876", ios::binary | ios::out);
    if(outfile__array_statemonitor_1_clock_1_t.is_open())
    {
        outfile__array_statemonitor_1_clock_1_t.write(reinterpret_cast<char*>(_array_statemonitor_1_clock_1_t), 1*sizeof(_array_statemonitor_1_clock_1_t[0]));
        outfile__array_statemonitor_1_clock_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1_clock_1_t." << endl;
    }
    ofstream outfile__array_statemonitor_1_clock_1_timestep;
    outfile__array_statemonitor_1_clock_1_timestep.open(results_dir + "_array_statemonitor_1_clock_1_timestep_1478009061", ios::binary | ios::out);
    if(outfile__array_statemonitor_1_clock_1_timestep.is_open())
    {
        outfile__array_statemonitor_1_clock_1_timestep.write(reinterpret_cast<char*>(_array_statemonitor_1_clock_1_timestep), 1*sizeof(_array_statemonitor_1_clock_1_timestep[0]));
        outfile__array_statemonitor_1_clock_1_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1_clock_1_timestep." << endl;
    }
    ofstream outfile__array_statemonitor_1_N;
    outfile__array_statemonitor_1_N.open(results_dir + "_array_statemonitor_1_N_2754233271", ios::binary | ios::out);
    if(outfile__array_statemonitor_1_N.is_open())
    {
        outfile__array_statemonitor_1_N.write(reinterpret_cast<char*>(_array_statemonitor_1_N), 1*sizeof(_array_statemonitor_1_N[0]));
        outfile__array_statemonitor_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_1_N." << endl;
    }
    ofstream outfile__array_statemonitor__indices;
    outfile__array_statemonitor__indices.open(results_dir + "_array_statemonitor__indices_2854283999", ios::binary | ios::out);
    if(outfile__array_statemonitor__indices.is_open())
    {
        outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 25*sizeof(_array_statemonitor__indices[0]));
        outfile__array_statemonitor__indices.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
    }
    ofstream outfile__array_statemonitor_clock_1_dt;
    outfile__array_statemonitor_clock_1_dt.open(results_dir + "_array_statemonitor_clock_1_dt_1009499131", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_1_dt.is_open())
    {
        outfile__array_statemonitor_clock_1_dt.write(reinterpret_cast<char*>(_array_statemonitor_clock_1_dt), 1*sizeof(_array_statemonitor_clock_1_dt[0]));
        outfile__array_statemonitor_clock_1_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_1_dt." << endl;
    }
    ofstream outfile__array_statemonitor_clock_1_t;
    outfile__array_statemonitor_clock_1_t.open(results_dir + "_array_statemonitor_clock_1_t_981617170", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_1_t.is_open())
    {
        outfile__array_statemonitor_clock_1_t.write(reinterpret_cast<char*>(_array_statemonitor_clock_1_t), 1*sizeof(_array_statemonitor_clock_1_t[0]));
        outfile__array_statemonitor_clock_1_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_1_t." << endl;
    }
    ofstream outfile__array_statemonitor_clock_1_timestep;
    outfile__array_statemonitor_clock_1_timestep.open(results_dir + "_array_statemonitor_clock_1_timestep_238373899", ios::binary | ios::out);
    if(outfile__array_statemonitor_clock_1_timestep.is_open())
    {
        outfile__array_statemonitor_clock_1_timestep.write(reinterpret_cast<char*>(_array_statemonitor_clock_1_timestep), 1*sizeof(_array_statemonitor_clock_1_timestep[0]));
        outfile__array_statemonitor_clock_1_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_clock_1_timestep." << endl;
    }
    ofstream outfile__array_statemonitor_N;
    outfile__array_statemonitor_N.open(results_dir + "_array_statemonitor_N_4140778434", ios::binary | ios::out);
    if(outfile__array_statemonitor_N.is_open())
    {
        outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
        outfile__array_statemonitor_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_statemonitor_N." << endl;
    }
    ofstream outfile__array_synapses_2_N;
    outfile__array_synapses_2_N.open(results_dir + "_array_synapses_2_N_1809632310", ios::binary | ios::out);
    if(outfile__array_synapses_2_N.is_open())
    {
        outfile__array_synapses_2_N.write(reinterpret_cast<char*>(_array_synapses_2_N), 1*sizeof(_array_synapses_2_N[0]));
        outfile__array_synapses_2_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_2_N." << endl;
    }
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    ofstream outfile__dynamic_array_ratemonitor_rate;
    outfile__dynamic_array_ratemonitor_rate.open(results_dir + "_dynamic_array_ratemonitor_rate_1996511615", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_rate.is_open())
    {
        if (! _dynamic_array_ratemonitor_rate.empty() )
        {
            outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_rate[0]), _dynamic_array_ratemonitor_rate.size()*sizeof(_dynamic_array_ratemonitor_rate[0]));
            outfile__dynamic_array_ratemonitor_rate.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
    }
    ofstream outfile__dynamic_array_ratemonitor_t;
    outfile__dynamic_array_ratemonitor_t.open(results_dir + "_dynamic_array_ratemonitor_t_1139349932", ios::binary | ios::out);
    if(outfile__dynamic_array_ratemonitor_t.is_open())
    {
        if (! _dynamic_array_ratemonitor_t.empty() )
        {
            outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_t[0]), _dynamic_array_ratemonitor_t.size()*sizeof(_dynamic_array_ratemonitor_t[0]));
            outfile__dynamic_array_ratemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_1_i;
    outfile__dynamic_array_spikemonitor_1_i.open(results_dir + "_dynamic_array_spikemonitor_1_i_2680224553", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_i.is_open())
    {
        if (! _dynamic_array_spikemonitor_1_i.empty() )
        {
            outfile__dynamic_array_spikemonitor_1_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_i[0]), _dynamic_array_spikemonitor_1_i.size()*sizeof(_dynamic_array_spikemonitor_1_i[0]));
            outfile__dynamic_array_spikemonitor_1_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_i." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_1_t;
    outfile__dynamic_array_spikemonitor_1_t.open(results_dir + "_dynamic_array_spikemonitor_1_t_4240873456", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_1_t.is_open())
    {
        if (! _dynamic_array_spikemonitor_1_t.empty() )
        {
            outfile__dynamic_array_spikemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_1_t[0]), _dynamic_array_spikemonitor_1_t.size()*sizeof(_dynamic_array_spikemonitor_1_t[0]));
            outfile__dynamic_array_spikemonitor_1_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_1_t." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_i;
    outfile__dynamic_array_spikemonitor_i.open(results_dir + "_dynamic_array_spikemonitor_i_1976709050", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_i.is_open())
    {
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
            outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
            outfile__dynamic_array_spikemonitor_i.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
    }
    ofstream outfile__dynamic_array_spikemonitor_t;
    outfile__dynamic_array_spikemonitor_t.open(results_dir + "_dynamic_array_spikemonitor_t_383009635", ios::binary | ios::out);
    if(outfile__dynamic_array_spikemonitor_t.is_open())
    {
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
            outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
            outfile__dynamic_array_spikemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_1_t;
    outfile__dynamic_array_statemonitor_1_t.open(results_dir + "_dynamic_array_statemonitor_1_t_3600064330", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_1_t.is_open())
    {
        if (! _dynamic_array_statemonitor_1_t.empty() )
        {
            outfile__dynamic_array_statemonitor_1_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_t[0]), _dynamic_array_statemonitor_1_t.size()*sizeof(_dynamic_array_statemonitor_1_t[0]));
            outfile__dynamic_array_statemonitor_1_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_1_t." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_t;
    outfile__dynamic_array_statemonitor_t.open(results_dir + "_dynamic_array_statemonitor_t_3983503110", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_t.is_open())
    {
        if (! _dynamic_array_statemonitor_t.empty() )
        {
            outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
            outfile__dynamic_array_statemonitor_t.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_post;
    outfile__dynamic_array_synapses_2__synaptic_post.open(results_dir + "_dynamic_array_synapses_2__synaptic_post_1591987953", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_2__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_2__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_post[0]), _dynamic_array_synapses_2__synaptic_post.size()*sizeof(_dynamic_array_synapses_2__synaptic_post[0]));
            outfile__dynamic_array_synapses_2__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2__synaptic_pre;
    outfile__dynamic_array_synapses_2__synaptic_pre.open(results_dir + "_dynamic_array_synapses_2__synaptic_pre_971331175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_2__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_2__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2__synaptic_pre[0]), _dynamic_array_synapses_2__synaptic_pre.size()*sizeof(_dynamic_array_synapses_2__synaptic_pre[0]));
            outfile__dynamic_array_synapses_2__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_delay;
    outfile__dynamic_array_synapses_2_delay.open(results_dir + "_dynamic_array_synapses_2_delay_3163926887", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_delay.is_open())
    {
        if (! _dynamic_array_synapses_2_delay.empty() )
        {
            outfile__dynamic_array_synapses_2_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_delay[0]), _dynamic_array_synapses_2_delay.size()*sizeof(_dynamic_array_synapses_2_delay[0]));
            outfile__dynamic_array_synapses_2_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_N_incoming;
    outfile__dynamic_array_synapses_2_N_incoming.open(results_dir + "_dynamic_array_synapses_2_N_incoming_3109283082", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_2_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_2_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_incoming[0]), _dynamic_array_synapses_2_N_incoming.size()*sizeof(_dynamic_array_synapses_2_N_incoming[0]));
            outfile__dynamic_array_synapses_2_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_2_N_outgoing;
    outfile__dynamic_array_synapses_2_N_outgoing.open(results_dir + "_dynamic_array_synapses_2_N_outgoing_2656015824", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_2_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_2_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_2_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_2_N_outgoing[0]), _dynamic_array_synapses_2_N_outgoing.size()*sizeof(_dynamic_array_synapses_2_N_outgoing[0]));
            outfile__dynamic_array_synapses_2_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_2_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
            outfile__dynamic_array_synapses__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
            outfile__dynamic_array_synapses__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        if (! _dynamic_array_synapses_delay.empty() )
        {
            outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
            outfile__dynamic_array_synapses_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
            outfile__dynamic_array_synapses_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
            outfile__dynamic_array_synapses_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }

    ofstream outfile__dynamic_array_statemonitor_1_v;
    outfile__dynamic_array_statemonitor_1_v.open(results_dir + "_dynamic_array_statemonitor_1_v_949681766", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_1_v.is_open())
    {
        for (int n=0; n<_dynamic_array_statemonitor_1_v.n; n++)
        {
            if (! _dynamic_array_statemonitor_1_v(n).empty())
            {
                outfile__dynamic_array_statemonitor_1_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_1_v(n, 0)), _dynamic_array_statemonitor_1_v.m*sizeof(_dynamic_array_statemonitor_1_v(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_1_v.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_1_v." << endl;
    }
    ofstream outfile__dynamic_array_statemonitor_v;
    outfile__dynamic_array_statemonitor_v.open(results_dir + "_dynamic_array_statemonitor_v_56692266", ios::binary | ios::out);
    if(outfile__dynamic_array_statemonitor_v.is_open())
    {
        for (int n=0; n<_dynamic_array_statemonitor_v.n; n++)
        {
            if (! _dynamic_array_statemonitor_v(n).empty())
            {
                outfile__dynamic_array_statemonitor_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_v(n, 0)), _dynamic_array_statemonitor_v.m*sizeof(_dynamic_array_statemonitor_v(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_v.close();
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_statemonitor_v." << endl;
    }
    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
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
    if(_static_array__array_statemonitor_1__indices!=0)
    {
        delete [] _static_array__array_statemonitor_1__indices;
        _static_array__array_statemonitor_1__indices = 0;
    }
    if(_static_array__array_statemonitor__indices!=0)
    {
        delete [] _static_array__array_statemonitor__indices;
        _static_array__array_statemonitor__indices = 0;
    }
}

