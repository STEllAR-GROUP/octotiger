// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/modularSilo.hpp"
#include <silo.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>

namespace octotiger {
void writeModularSilo(std::vector<TransportSnapshot> const& patches, std::string const& filename, int cycle, Real time) {
    if (patches.empty()) throw std::invalid_argument("Empty Silo patch directory");
    bool const hydroEnabled = patches.front().hydroEnabled;
    bool const radiationEnabled = patches.front().radiationEnabled;
    if ((!hydroEnabled && !radiationEnabled) || !std::isfinite(time))
        throw std::invalid_argument("Invalid Silo fields or physical time");
    for (auto const& patch : patches) {
        if (patch.hydroEnabled != hydroEnabled || patch.radiationEnabled != radiationEnabled ||
            (patch.hydroEnabled && patch.hydro.timeState().time != time) ||
            (patch.radiationEnabled && patch.radiation.timeState().time != time))
            throw std::invalid_argument("Silo snapshot field selection or physical time mismatch");
        if (patch.layout().dimensionCount() != patches.front().layout().dimensionCount() ||
            patch.location.dimensionCount != patch.layout().dimensionCount() ||
            (patch.hydroEnabled && patch.radiationEnabled &&
             (patch.hydro.layout().extents() != patch.radiation.layout().extents() ||
              patch.hydro.layout().interiorExtents() != patch.radiation.layout().interiorExtents() ||
              patch.hydro.layout().ghostWidth() != patch.radiation.layout().ghostWidth() ||
              patch.hydro.lower() != patch.radiation.lower() ||
              patch.hydro.cellWidth() != patch.radiation.cellWidth())))
            throw std::invalid_argument("Inconsistent Silo patch geometry");
        if ((patch.hydroEnabled && patch.hydro.values().size() != patch.hydro.layout().cellCount()) ||
            (patch.radiationEnabled && patch.radiation.values().size() != patch.radiation.layout().cellCount()))
            throw std::invalid_argument("Incomplete Silo field storage");
    }
    std::filesystem::path const path(filename);
    if (std::filesystem::exists(path)) throw std::runtime_error("Refusing to overwrite " + path.string());
    std::unique_ptr<DBfile, decltype(&DBClose)> fileOwner(
        DBCreate(path.string().c_str(), DB_CLOBBER, DB_LOCAL, "Modular transport", DB_HDF5), &DBClose);
    DBfile* file = fileOwner.get();
    if (!file) throw std::runtime_error("Unable to create modular Silo output");
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> optionsOwner(DBMakeOptlist(2), &DBFreeOptlist);
    DBoptlist* options = optionsOwner.get();
    if (!options) throw std::runtime_error("Unable to allocate Silo options");
    double outputTime = static_cast<double>(time);
    DBAddOption(options, DBOPT_DTIME, &outputTime);
    DBAddOption(options, DBOPT_CYCLE, &cycle);
    std::vector<std::string> meshNames;
    std::vector<std::string> variables;
    if (hydroEnabled) variables = {"density", "momentumX", "momentumY", "momentumZ", "energy"};
    if (radiationEnabled) variables.insert(variables.end(), {"radiationEnergy", "radiationFluxX", "radiationFluxY", "radiationFluxZ"});
    std::vector<std::vector<std::string>> variableNames(variables.size());
    int status = 0;
    for (std::size_t block = 0; block < patches.size(); ++block) {
        std::string const blockName = "block" + std::to_string(block);
        status |= DBMkDir(file, blockName.c_str());
        status |= DBSetDir(file, blockName.c_str());
        auto const& fields = patches[block];
        int const dimensions = fields.layout().dimensionCount();
        int meshDimensions[3]{1,1,1}, zoneDimensions[3]{1,1,1};
        std::array<std::vector<double>,3> coordinates;
        void* coordinatePointers[3]{};
        for (int axis = 0; axis < dimensions; ++axis) {
            zoneDimensions[axis] = fields.layout().interiorExtent(axis);
            meshDimensions[axis] = zoneDimensions[axis] + 1;
            for (int i = 0; i < meshDimensions[axis]; ++i)
                coordinates[axis].push_back(static_cast<double>(fields.lower()[axis] + i * fields.cellWidth()));
            coordinatePointers[axis] = coordinates[axis].data();
        }
        status |= DBPutQuadmesh(file, "mesh", nullptr, coordinatePointers, meshDimensions,
            dimensions, DB_DOUBLE, DB_COLLINEAR, options);
        meshNames.push_back(blockName + "/mesh");
        for (std::size_t field = 0; field < variables.size(); ++field) {
            std::vector<double> values;
            fields.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
                if (fields.hydroEnabled && field < 5) {
                    values.push_back(static_cast<double>(fields.hydro.values()[index][static_cast<int>(field)]));
                } else {
                    int const radiationField = static_cast<int>(field) - (fields.hydroEnabled ? 5 : 0);
                    auto const& state = fields.radiation.values()[index];
                    // Q is stored as F/c, not F/cHat. Always export physical F.
                    Real const value = radiationField == 0 ? state[0] : physicalLightSpeed * state[radiationField];
                    values.push_back(static_cast<double>(value));
                }
            });
            status |= DBPutQuadvar1(file, variables[field].c_str(), "mesh", values.data(), zoneDimensions,
                dimensions, nullptr, 0, DB_DOUBLE, DB_ZONECENT, options);
            variableNames[field].push_back(blockName + "/" + variables[field]);
        }
        status |= DBSetDir(file, "/");
    }
    std::vector<char const*> names;
    std::vector<int> types(patches.size(), DB_QUAD_RECT);
    for (auto const& meshName : meshNames) names.push_back(meshName.c_str());
    status |= DBPutMultimesh(file, "mesh", static_cast<int>(names.size()), names.data(), types.data(), options);
    std::fill(types.begin(), types.end(), DB_QUADVAR);
    for (std::size_t field = 0; field < variables.size(); ++field) {
        names.clear();
        for (auto const& variableName : variableNames[field]) names.push_back(variableName.c_str());
        status |= DBPutMultivar(file, variables[field].c_str(), static_cast<int>(names.size()), names.data(), types.data(), options);
    }
    optionsOwner.reset();
    status |= DBClose(fileOwner.release());
    if (status < 0) throw std::runtime_error("Writing modular Silo output failed");
}
} // namespace octotiger
