// Parameterized grey comoving opacities; no frequency integration or EOS dependency.
#pragma once
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace radiation {
struct GreyOpacity {
    std::string model = "legacy";
    std::string units = "cm2/g";
    double absorption = 0;            // Planck AND radiation-energy absorption mean
    double scattering = 0;            // coherent, isotropic transport scattering
    double transportAbsorption = -1; // -1 follows absorption; otherwise Rosseland absorption

    void validate(double legacyConstant) const {
        if (model != "legacy" && model != "skinner_ostriker" && model != "grey")
            throw std::runtime_error("radiation.opacity.model must be legacy|skinner_ostriker|grey");
        if (units != "cm2/g" && units != "1/cm")
            throw std::runtime_error("radiation.opacity.units must be cm2/g|1/cm");
        if (!std::isfinite(absorption) || absorption < 0 ||
            !std::isfinite(scattering) || scattering < 0 ||
            !std::isfinite(transportAbsorption) ||
            (transportAbsorption < 0 && transportAbsorption != -1) ||
            !std::isfinite(legacyConstant))
            throw std::runtime_error("Invalid radiation.opacity coefficient: require finite nonnegative values (transport_absorption permits -1)");
        if (model == "skinner_ostriker" && legacyConstant < 0)
            throw std::runtime_error("skinner_ostriker requires radiation.opacity.constant >= 0 (legacy code area/mass units)");
        if (model == "grey" && legacyConstant >= 0)
            throw std::runtime_error("grey opacity conflicts with radiation.opacity.constant/rad_opacity; remove the constant setting or set it to -1");
        if (model != "grey" && (absorption != 0 || scattering != 0 || transportAbsorption != -1 || units != "cm2/g"))
            throw std::runtime_error("Separate absorption/scattering/transport units require radiation.opacity.model=grey");
    }
    template<class Archive> void serialize(Archive& arc, unsigned) {
        arc & model & units & absorption & scattering & transportAbsorption;
    }
};

// Restore only explicit config/CLI fields after loading checkpoint defaults.
template<class Supplied>
void restoreGreyOpacityOverrides(GreyOpacity& loaded, GreyOpacity const& explicitValues, Supplied supplied) {
    if (supplied("radiation.opacity.model")) loaded.model=explicitValues.model;
    if (supplied("radiation.opacity.units")) loaded.units=explicitValues.units;
    if (supplied("radiation.opacity.absorption")) loaded.absorption=explicitValues.absorption;
    if (supplied("radiation.opacity.scattering")) loaded.scattering=explicitValues.scattering;
    if (supplied("radiation.opacity.transport_absorption")) loaded.transportAbsorption=explicitValues.transportAbsorption;
}

struct GreyCoefficients {
    double absorption; // inverse CODE length; controls emission and thermal exchange
    double scattering;
    double transport;  // inverse CODE length; controls flux damping and force
};

inline GreyCoefficients greyCoefficients(GreyOpacity const& options,
    double rho, double codeToG, double codeToCm) {
    if (!(std::isfinite(rho) && rho >= 0 && std::isfinite(codeToG) && codeToG > 0 &&
          std::isfinite(codeToCm) && codeToCm > 0))
        throw std::runtime_error("Invalid density or unit conversion for grey opacity");
    // rho_cgs = rho_code * codeToG / codeToCm^3; chi_code = chi_cgs * codeToCm.
    double const factor = options.units == "1/cm" ? codeToCm :
        rho * (codeToG / codeToCm) / codeToCm;
    double const transportAbsorption = options.transportAbsorption < 0 ?
        options.absorption : options.transportAbsorption;
    GreyCoefficients const result{factor * options.absorption, factor * options.scattering,
        factor * (transportAbsorption + options.scattering)};
    if (!(std::isfinite(result.absorption) && result.absorption >= 0 &&
          std::isfinite(result.scattering) && result.scattering >= 0 &&
          std::isfinite(result.transport) && result.transport >= 0))
        throw std::runtime_error("Nonfinite or negative grey extinction coefficient");
    return result;
}

// Versioned, additive Silo metadata. Callbacks keep the exact production codec testable
// without Silo/HPX. Missing extension means a pre-Step-04 checkpoint (legacy defaults).
template<class WriteInt, class WriteReal>
void saveGreyOpacity(GreyOpacity const& o, WriteInt wi, WriteReal wr) {
    wi("rad_opacity_schema", 1);
    wi("rad_opacity_model", o.model == "legacy" ? 0 : o.model == "skinner_ostriker" ? 1 : 2);
    wi("rad_opacity_units", o.units == "cm2/g" ? 0 : 1);
    wr("rad_opacity_absorption", o.absorption);
    wr("rad_opacity_scattering", o.scattering);
    wr("rad_opacity_transport_absorption", o.transportAbsorption);
}
template<class Exists, class ReadInt, class ReadReal>
GreyOpacity loadGreyOpacity(Exists exists, ReadInt ri, ReadReal rr) {
    GreyOpacity o;
    if (!exists("rad_opacity_schema")) return o;
    if (ri("rad_opacity_schema") != 1)
        throw std::runtime_error("Unsupported radiation opacity checkpoint schema");
    for (auto key : {"rad_opacity_model", "rad_opacity_units", "rad_opacity_absorption",
                    "rad_opacity_scattering", "rad_opacity_transport_absorption"})
        if (!exists(key)) throw std::runtime_error("Incomplete radiation opacity checkpoint metadata");
    auto const model = ri("rad_opacity_model"), units = ri("rad_opacity_units");
    if (model < 0 || model > 2 || units < 0 || units > 1)
        throw std::runtime_error("Invalid radiation opacity checkpoint model/units");
    o.model = model == 0 ? "legacy" : model == 1 ? "skinner_ostriker" : "grey";
    o.units = units == 0 ? "cm2/g" : "1/cm";
    o.absorption = rr("rad_opacity_absorption");
    o.scattering = rr("rad_opacity_scattering");
    o.transportAbsorption = rr("rad_opacity_transport_absorption");
    return o; // Validate after explicit CLI/config overrides, alongside other options.
}
} // namespace radiation
