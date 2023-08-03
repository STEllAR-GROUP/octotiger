#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

/// Singleton util class to easily store the times per timestep and print them during shutdown
class timestep_util {
  private: 
    std::vector<double> times_per_timestep;
    /// Init with some space already reserved
    timestep_util(void) {times_per_timestep.reserve(200);}

    /// print the entire list of timesteps together with min,max,median and average
    /// Called automatically in destructor
    void print(void) {
      if (!times_per_timestep.empty()) {
        auto times = times_per_timestep;
        std::size_t n = times.size() / 2;
        std::nth_element(times.begin(), times.begin() + n, times.end());
        double median = times[n];
        if (times.size() % 2 == 0)
            std::nth_element(times.begin(), times.begin() + n - 1, times.end());
        median = 0.5 * (median + times[n - 1]);
        std::cout << std::endl;
        std::cout << "Time per timestep report (in seconds)" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Min time-per-timestep: "
                  << *(std::min_element(std::begin(times), std::end(times))) << std::endl;
        std::cout << "Median time-per-timestep: " << median << std::endl;
        std::cout << "Average time-per-timestep: "
                  << std::reduce(std::begin(times), std::end(times)) /
                  times.size() << std::endl;
        std::cout << "Max time-per-timestep: "
                  << *(std::max_element(std::begin(times), std::end(times))) << std::endl;
        std::cout << "List of times-per-timestep:";
        for (const auto time : times_per_timestep)
            std::cout << " " << time;
        std::cout << std::endl;
      }
    }

    /// Singleton access
    static timestep_util& instance(void) {
      static timestep_util inst{};
      return inst;
    }
  public:
    
    /// Add a timestep
    static void add_time_per_timestep(const double runtime) {
      instance().times_per_timestep.push_back(runtime);
    }

    /// Print when destroyed (and not empty)
    ~timestep_util(void) {print();}

};
