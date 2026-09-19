#pragma once
#include "octotiger/radiation/grey_opacity.hpp"
#include <boost/program_options.hpp>

namespace radiation {
inline void addGreyOpacityOptions(boost::program_options::options_description& description, GreyOpacity& opacity) {
    namespace po = boost::program_options;
    description.add_options()
        ("radiation.opacity.model", po::value<std::string>(&opacity.model)->default_value("legacy"), "legacy|skinner_ostriker|grey; legacy preserves old opacity selection")
        ("radiation.opacity.units", po::value<std::string>(&opacity.units)->default_value("cm2/g"), "Grey coefficients: cm2/g or 1/cm; does not change legacy constant units")
        ("radiation.opacity.absorption", po::value<double>(&opacity.absorption)->default_value(0), "Grey Planck/energy absorption mean; excludes scattering")
        ("radiation.opacity.scattering", po::value<double>(&opacity.scattering)->default_value(0), "Grey coherent isotropic scattering")
        ("radiation.opacity.transport_absorption", po::value<double>(&opacity.transport_absorption)->default_value(-1), "Grey Rosseland absorption; -1 follows absorption; total adds scattering");
}
} // namespace radiation
