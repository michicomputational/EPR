#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/poissoninput_1_codeobject.h"
#include "code_objects/spikemonitor_2_codeobject.h"
#include "code_objects/spikemonitor_3_codeobject.h"
#include "code_objects/synapses_4_pre_codeobject.h"
#include "code_objects/synapses_4_pre_push_spikes.h"
#include "code_objects/before_run_synapses_4_pre_push_spikes.h"
#include "code_objects/synapses_4_stateupdater_codeobject.h"
#include "code_objects/synapses_4_summed_variable_I_syn_ee_post_codeobject.h"
#include "code_objects/synapses_4_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_5_pre_codeobject.h"
#include "code_objects/synapses_5_pre_push_spikes.h"
#include "code_objects/before_run_synapses_5_pre_push_spikes.h"
#include "code_objects/synapses_5_stateupdater_codeobject.h"
#include "code_objects/synapses_5_summed_variable_I_syn_ei_post_codeobject.h"
#include "code_objects/synapses_5_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_6_pre_codeobject.h"
#include "code_objects/synapses_6_pre_push_spikes.h"
#include "code_objects/before_run_synapses_6_pre_push_spikes.h"
#include "code_objects/synapses_6_stateupdater_codeobject.h"
#include "code_objects/synapses_6_summed_variable_I_syn_ie_post_codeobject.h"
#include "code_objects/synapses_6_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_7_pre_codeobject.h"
#include "code_objects/synapses_7_pre_push_spikes.h"
#include "code_objects/before_run_synapses_7_pre_push_spikes.h"
#include "code_objects/synapses_7_stateupdater_codeobject.h"
#include "code_objects/synapses_7_summed_variable_I_syn_ii_post_codeobject.h"
#include "code_objects/synapses_7_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>


        std::string _format_time(float time_in_s)
        {
            float divisors[] = {24*60*60, 60*60, 60, 1};
            char letters[] = {'d', 'h', 'm', 's'};
            float remaining = time_in_s;
            std::string text = "";
            int time_to_represent;
            for (int i =0; i < sizeof(divisors)/sizeof(float); i++)
            {
                time_to_represent = int(remaining / divisors[i]);
                remaining -= time_to_represent * divisors[i];
                if (time_to_represent > 0 || text.length())
                {
                    if(text.length() > 0)
                    {
                        text += " ";
                    }
                    text += (std::to_string(time_to_represent)+letters[i]);
                }
            }
            //less than one second
            if(text.length() == 0) 
            {
                text = "< 1s";
            }
            return text;
        }
        void report_progress(const double elapsed, const double completed, const double start, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << _format_time(elapsed);
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << _format_time(remaining) << " remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_lastspike; i++)
                        {
                            _array_neurongroup_1_lastspike[i] = - 10000.0;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_not_refractory; i++)
                        {
                            _array_neurongroup_1_not_refractory[i] = true;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_tau_mem; i++)
                        {
                            _array_neurongroup_1_tau_mem[i] = 0.02;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_1_R; i++)
                        {
                            _array_neurongroup_1_R[i] = 80000000.0;
                        }
                        
        _dynamic_array_synapses_4_delay.resize(1);
        _dynamic_array_synapses_4_delay.resize(1);
        _dynamic_array_synapses_4_delay[0] = 0.001;
        _run_synapses_4_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_4_y.size(); i++)
                        {
                            _dynamic_array_synapses_4_y[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_4_tau_inact.size(); i++)
                        {
                            _dynamic_array_synapses_4_tau_inact[i] = 0.0015;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_4_A_SE.size(); i++)
                        {
                            _dynamic_array_synapses_4_A_SE[i] = 1e-11;
                        }
                        
        _dynamic_array_synapses_5_delay.resize(1);
        _dynamic_array_synapses_5_delay.resize(1);
        _dynamic_array_synapses_5_delay[0] = 0.001;
        _run_synapses_5_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_5_y.size(); i++)
                        {
                            _dynamic_array_synapses_5_y[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_5_tau_inact.size(); i++)
                        {
                            _dynamic_array_synapses_5_tau_inact[i] = 0.0015;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_5_A_SE.size(); i++)
                        {
                            _dynamic_array_synapses_5_A_SE[i] = 1e-11;
                        }
                        
        _dynamic_array_synapses_6_delay.resize(1);
        _dynamic_array_synapses_6_delay.resize(1);
        _dynamic_array_synapses_6_delay[0] = 0.001;
        _run_synapses_6_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_6_y.size(); i++)
                        {
                            _dynamic_array_synapses_6_y[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_6_tau_inact.size(); i++)
                        {
                            _dynamic_array_synapses_6_tau_inact[i] = 0.0015;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_6_A_SE.size(); i++)
                        {
                            _dynamic_array_synapses_6_A_SE[i] = - 4.9999999999999995e-11;
                        }
                        
        _dynamic_array_synapses_7_delay.resize(1);
        _dynamic_array_synapses_7_delay.resize(1);
        _dynamic_array_synapses_7_delay[0] = 0.001;
        _run_synapses_7_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_7_y.size(); i++)
                        {
                            _dynamic_array_synapses_7_y[i] = 0.0;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_7_tau_inact.size(); i++)
                        {
                            _dynamic_array_synapses_7_tau_inact[i] = 0.0015;
                        }
                        
        
                        
                        for(int i=0; i<_dynamic_array_synapses_7_A_SE.size(); i++)
                        {
                            _dynamic_array_synapses_7_A_SE[i] = - 4.9999999999999995e-11;
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_synapses_4_pre_push_spikes();
        _before_run_synapses_5_pre_push_spikes();
        _before_run_synapses_6_pre_push_spikes();
        _before_run_synapses_7_pre_push_spikes();
        network_1.clear();
        network_1.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        network_1.add(&defaultclock, _run_synapses_4_stateupdater_codeobject);
        network_1.add(&defaultclock, _run_synapses_4_summed_variable_I_syn_ee_post_codeobject);
        network_1.add(&defaultclock, _run_synapses_5_stateupdater_codeobject);
        network_1.add(&defaultclock, _run_synapses_5_summed_variable_I_syn_ei_post_codeobject);
        network_1.add(&defaultclock, _run_synapses_6_stateupdater_codeobject);
        network_1.add(&defaultclock, _run_synapses_6_summed_variable_I_syn_ie_post_codeobject);
        network_1.add(&defaultclock, _run_synapses_7_stateupdater_codeobject);
        network_1.add(&defaultclock, _run_synapses_7_summed_variable_I_syn_ii_post_codeobject);
        network_1.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        network_1.add(&defaultclock, _run_spikemonitor_2_codeobject);
        network_1.add(&defaultclock, _run_spikemonitor_3_codeobject);
        network_1.add(&defaultclock, _run_synapses_4_pre_push_spikes);
        network_1.add(&defaultclock, _run_synapses_4_pre_codeobject);
        network_1.add(&defaultclock, _run_synapses_5_pre_push_spikes);
        network_1.add(&defaultclock, _run_synapses_5_pre_codeobject);
        network_1.add(&defaultclock, _run_synapses_6_pre_push_spikes);
        network_1.add(&defaultclock, _run_synapses_6_pre_codeobject);
        network_1.add(&defaultclock, _run_synapses_7_pre_push_spikes);
        network_1.add(&defaultclock, _run_synapses_7_pre_codeobject);
        network_1.add(&defaultclock, _run_poissoninput_1_codeobject);
        network_1.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject);
        network_1.run(10.0, report_progress, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _array_defaultclock_dt[0] = 0.0001;
        #ifdef DEBUG
        _debugmsg_spikemonitor_2_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_spikemonitor_3_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_4_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_5_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_6_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_7_pre_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}