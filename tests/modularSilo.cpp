// Copyright (c) 2026 AUTHORS
// Distributed under the Boost Software License, Version 1.0.
#include "octotiger/subgrid/modularSilo.hpp"
#include <silo.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace octotiger;
void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) <=
        1e-12 * std::max(1.0, std::abs(expected)), "Silo value differs from source data");
}

std::vector<TransportSnapshot> initial(int dimensions, bool hydroEnabled, bool radiationEnabled) {
    mesh::BlockLocation const root{0,{0,0,0},dimensions};
    std::vector<TransportSnapshot> result;
    for (int slot = 0; slot < (1 << dimensions); ++slot) {
        TransportSnapshot snapshot;
        snapshot.location = root.child(slot);
        snapshot.hydroEnabled = hydroEnabled;
        snapshot.radiationEnabled = radiationEnabled;
        mesh::PhysicalCoordinates lower{};
        for (int axis = 0; axis < dimensions; ++axis) lower[axis] = snapshot.location.coordinates[axis];
        mesh::MeshLayout layout(dimensions,4,2);
        if (hydroEnabled) {
            snapshot.hydro = hydro::Fields(layout,Real(0.25),lower);
            snapshot.hydro.timeState().time = Real(0.25);
        }
        if (radiationEnabled) {
            snapshot.radiation = radiation::Fields(layout,Real(0.25),lower);
            snapshot.radiation.timeState().time = Real(0.25);
        }
        layout.forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
            Real const energy = 1 + Real(0.01)*static_cast<Real>(index) + slot;
            if (hydroEnabled) {
                hydro::PrimitiveState primitive;
                primitive.density() = energy;
                primitive.pressure() = 1;
                primitive.velocity(0) = Real(0.2);
                snapshot.hydro.values()[index] = hydro::HydroSystem(Real(1.4)).conservedState(primitive);
            }
            if (radiationEnabled)
                snapshot.radiation.values()[index] = {energy,Real(0.1)*energy,Real(0.2)*energy,Real(0.3)*energy};
        });
        result.push_back(std::move(snapshot));
    }
    return result;
}

void roundtrip(std::filesystem::path const& directory, int dimensions, bool hydroEnabled, bool radiationEnabled) {
    auto const patches = initial(dimensions, hydroEnabled, radiationEnabled);
    auto const filename = directory / (std::to_string(dimensions) + "D" +
        (hydroEnabled ? "Hydro" : "") + (radiationEnabled ? "Radiation" : "") + ".silo");
    writeModularSilo(patches, filename.string(), 7, Real(0.25));
    std::unique_ptr<DBfile,decltype(&DBClose)> file(DBOpen(filename.string().c_str(), DB_UNKNOWN, DB_READ), DBClose);
    require(bool(file), "Silo file did not reopen");
    std::unique_ptr<DBmultimesh,decltype(&DBFreeMultimesh)> multi(DBGetMultimesh(file.get(),"mesh"),DBFreeMultimesh);
    require(bool(multi) && multi->nblocks == static_cast<int>(patches.size()), "Incorrect Silo multimesh");
    std::vector<std::string> names;
    if (hydroEnabled) names = {"density","momentumX","momentumY","momentumZ","energy"};
    if (radiationEnabled) names.insert(names.end(), {"radiationEnergy","radiationFluxX","radiationFluxY","radiationFluxZ"});
    for (auto const& name : names) {
        std::unique_ptr<DBmultivar,decltype(&DBFreeMultivar)> variable(DBGetMultivar(file.get(),name.c_str()),DBFreeMultivar);
        require(bool(variable) && variable->nvars == multi->nblocks, "Incorrect Silo multivar");
    }
    for (std::size_t block = 0; block < patches.size(); ++block) {
        auto const& patch = patches[block];
        std::string const prefix = "block" + std::to_string(block) + "/";
        std::unique_ptr<DBquadmesh,decltype(&DBFreeQuadmesh)> quadMesh(
            DBGetQuadmesh(file.get(),(prefix+"mesh").c_str()),DBFreeQuadmesh);
        require(bool(quadMesh) && quadMesh->ndims == dimensions && quadMesh->datatype == DB_DOUBLE,
            "Silo mesh has wrong dimensionality or datatype");
        require(quadMesh->cycle == 7, "Silo cycle is missing");
        near(quadMesh->dtime,0.25);
        for (int axis = 0; axis < dimensions; ++axis) {
            require(quadMesh->dims[axis] == 5, "Silo includes ghost coordinates or misses cell boundaries");
            auto const* coordinates = static_cast<double const*>(quadMesh->coords[axis]);
            for (int coordinate = 0; coordinate < 5; ++coordinate)
                near(coordinates[coordinate],patch.lower()[axis]+Real(0.25)*coordinate);
        }
        for (std::size_t field = 0; field < names.size(); ++field) {
            std::unique_ptr<DBquadvar,decltype(&DBFreeQuadvar)> variable(
                DBGetQuadvar(file.get(),(prefix+names[field]).c_str()),DBFreeQuadvar);
            require(bool(variable) && variable->ndims == dimensions && variable->datatype == DB_DOUBLE &&
                variable->centering == DB_ZONECENT && variable->nvals == 1 &&
                variable->nels == static_cast<int>(patch.layout().interiorCellCount()),
                "Silo variable geometry/centering is incorrect");
            std::size_t outputIndex = 0;
            auto const* values = static_cast<double const*>(variable->vals[0]);
            patch.layout().forEachInterior([&](mesh::Coordinates const&, std::size_t index) {
                Real expected;
                if (hydroEnabled && field < 5) expected = patch.hydro.values()[index][static_cast<int>(field)];
                else {
                    int const component = static_cast<int>(field) - (hydroEnabled ? 5 : 0);
                    expected = patch.radiation.values()[index][component] * (component == 0 ? 1 : physicalLightSpeed);
                }
                near(values[outputIndex++],expected);
            });
        }
    }
    bool rejected = false;
    try { writeModularSilo(patches,filename.string(),7,Real(0.25)); }
    catch (std::exception const&) { rejected = true; }
    require(rejected,"Silo writer must not overwrite an existing output frame");
}
}

int main() {
    try {
        auto const suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        auto const directory = std::filesystem::temp_directory_path() / ("octotigerSilo"+std::to_string(suffix));
        require(std::filesystem::create_directory(directory),"Cannot create unique Silo test directory");
        for (int dimensions = 1; dimensions <= 3; ++dimensions) {
            roundtrip(directory, dimensions, true, false);
            roundtrip(directory, dimensions, false, true);
            roundtrip(directory, dimensions, true, true);
        }
        std::cout << "Modular Silo roundtrip passed: " << directory << '\n';
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
