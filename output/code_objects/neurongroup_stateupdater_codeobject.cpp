#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    static inline int64_t _timestep(double t, double dt)
    {
        return (int64_t)((t + 1e-3*dt)/dt); 
    }
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<int,long double> { typedef long double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long,long double> { typedef long double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<long long,long double> { typedef long double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int> { typedef long double type; };
    template < > struct _higher_type<long double,long> { typedef long double type; };
    template < > struct _higher_type<long double,long long> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const size_t _numI_syn_ee = 5000;
const size_t _numI_syn_ei = 5000;
const size_t _numI_syn_ie = 5000;
const size_t _numI_syn_ii = 5000;
const int64_t N = 5000;
const size_t _numR = 5000;
const size_t _numdt = 1;
const size_t _numlastspike = 5000;
const size_t _numnot_refractory = 5000;
const size_t _numt = 1;
const size_t _numtau_mem = 5000;
const size_t _numv = 5000;
    ///// POINTERS ////////////
        
    double* __restrict  _ptr_array_neurongroup_I_syn_ee = _array_neurongroup_I_syn_ee;
    double* __restrict  _ptr_array_neurongroup_I_syn_ei = _array_neurongroup_I_syn_ei;
    double* __restrict  _ptr_array_neurongroup_I_syn_ie = _array_neurongroup_I_syn_ie;
    double* __restrict  _ptr_array_neurongroup_I_syn_ii = _array_neurongroup_I_syn_ii;
    double* __restrict  _ptr_array_neurongroup_R = _array_neurongroup_R;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
    char* __restrict  _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
    double*   _ptr_array_defaultclock_t = _array_defaultclock_t;
    double* __restrict  _ptr_array_neurongroup_tau_mem = _array_neurongroup_tau_mem;
    double* __restrict  _ptr_array_neurongroup_v = _array_neurongroup_v;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double t = _ptr_array_defaultclock_t[0];
    const int64_t _lio_1 = _timestep(0.002, dt);
    const double _lio_2 = - dt;


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        const double I_syn_ee = _ptr_array_neurongroup_I_syn_ee[_idx];
        const double I_syn_ei = _ptr_array_neurongroup_I_syn_ei[_idx];
        const double I_syn_ie = _ptr_array_neurongroup_I_syn_ie[_idx];
        const double I_syn_ii = _ptr_array_neurongroup_I_syn_ii[_idx];
        const double R = _ptr_array_neurongroup_R[_idx];
        const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
        char not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
        const double tau_mem = _ptr_array_neurongroup_tau_mem[_idx];
        double v = _ptr_array_neurongroup_v[_idx];
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        double _v;
        if(!not_refractory)
            _v = (((((I_syn_ee * R) + (I_syn_ei * R)) + (I_syn_ie * R)) + (I_syn_ii * R)) + v) + (((- I_syn_ee) * R) - (((I_syn_ei * R) + (I_syn_ie * R)) + (I_syn_ii * R)));
        else 
            _v = (((((I_syn_ee * R) + (I_syn_ei * R)) + (I_syn_ie * R)) + (I_syn_ii * R)) + (v * exp(1.0f*_lio_2/tau_mem))) + ((((- I_syn_ee) * R) - (((I_syn_ei * R) + (I_syn_ie * R)) + (I_syn_ii * R))) * exp(1.0f*_lio_2/tau_mem));
        if(not_refractory)
            v = _v;
        _ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
        _ptr_array_neurongroup_v[_idx] = v;

    }

}


