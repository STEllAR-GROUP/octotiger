#pragma once
#include "octotiger/radiation/grey_opacity.hpp"
#include <boost/program_options.hpp>

namespace radiation {
inline void addGreyOpacityOptions(boost::program_options::options_description& description, GreyOpacity& opacity) {
    namespace po = boost::program_options;
    description.add_options()
        ("radiation.opacity.model", po::value<std::string>(&opacity.model)->default_value("legacy"),
            "Select legacy, skinner_ostriker, or grey opacity behavior; legacy preserves the historical selection.")
        ("radiation.opacity.units", po::value<std::string>(&opacity.units)->default_value("cm2/g"),
            "Select cm2/g or 1/cm for the grey coefficients; this does not change radiation.opacity.constant units.")
        ("radiation.opacity.absorption", po::value<double>(&opacity.absorption)->default_value(0),
            "Set the grey Planck-mean energy-absorption coefficient, excluding scattering.")
        ("radiation.opacity.scattering", po::value<double>(&opacity.scattering)->default_value(0),
            "Set the grey coherent isotropic-scattering coefficient.")
        ("radiation.opacity.transport_absorption", po::value<double>(&opacity.transportAbsorption)->default_value(-1),
            "Set the grey Rosseland-mean absorption coefficient; -1 uses radiation.opacity.absorption.");
}
} // namespace radiation
