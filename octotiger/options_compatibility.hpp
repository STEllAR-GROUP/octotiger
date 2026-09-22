#pragma once

#include <boost/program_options.hpp>

#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace octotiger::optionsCompatibility {

using migration = std::pair<std::string, std::string>; // legacy, canonical

inline std::string canonicalOptionDescription(
    boost::program_options::options_description const& legacyOptions,
    char const* legacy) {
    auto const* option = legacyOptions.find_nothrow(legacy, false);
    std::string description = option == nullptr ? std::string{} : option->description();
    if (!description.empty()) {
        description += " ";
    }
    description += "(legacy spelling: --";
    description += legacy;
    description += ")";
    return description;
}

template <class T>
void addCanonicalOption(boost::program_options::options_description& options,
    boost::program_options::options_description const& legacyOptions,
    std::vector<migration>& migrations, char const* canonical, char const* legacy,
    T* value) {
    auto const description = canonicalOptionDescription(legacyOptions, legacy);
    options.add_options()(canonical, boost::program_options::value<T>(value),
        description.c_str());
    migrations.emplace_back(legacy, canonical);
}

template <class T>
void addCanonicalMultitokenOption(boost::program_options::options_description& options,
    boost::program_options::options_description const& legacyOptions,
    std::vector<migration>& migrations, char const* canonical, char const* legacy,
    T* value) {
    auto const description = canonicalOptionDescription(legacyOptions, legacy);
    options.add_options()(canonical,
        boost::program_options::value<T>(value)->multitoken(), description.c_str());
    migrations.emplace_back(legacy, canonical);
}

inline void rememberSupplied(std::set<std::string>& supplied,
    boost::program_options::parsed_options const& parsed) {
    for (auto const& option : parsed.options) {
        if (!option.string_key.empty() && !option.unregistered) {
            supplied.insert(option.string_key);
        }
    }
}

inline bool checkCompatibilitySpellings(std::set<std::string> const& supplied,
    std::vector<migration> const& migrations, std::ostream& error = std::cerr) {
    bool valid = true;
    for (auto const& entry : migrations) {
        auto const& legacy = entry.first;
        auto const& canonical = entry.second;
        if (supplied.count(legacy) != 0 && supplied.count(canonical) != 0) {
            error << "ERROR: both legacy option '" << legacy
                  << "' and canonical option '" << canonical
                  << "' were supplied for the same setting. Remove one spelling; prefer '"
                  << canonical << "'.\n";
            valid = false;
        }
    }
    return valid;
}

inline void warnLegacySpellings(std::set<std::string> const& supplied,
    std::vector<migration> const& migrations, std::ostream& warning = std::cerr) {
    std::vector<migration> used;
    std::set<std::string> seen;
    for (auto const& entry : migrations) {
        if (supplied.count(entry.first) != 0 && seen.insert(entry.first).second) {
            used.push_back(entry);
        }
    }
    if (used.empty()) {
        return;
    }
    warning << "WARNING: legacy options were used. These options have been upgraded "
               "to hierarchical names:\n";
    for (auto const& entry : used) {
        warning << "  --" << entry.first << " -> --" << entry.second << '\n';
    }
}

inline void reapplyCanonicalValues(boost::program_options::variables_map const& values,
    boost::program_options::options_description const& canonicalOptions,
    std::vector<migration> const& migrations) {
    // Legacy entries carry the historical default_value declarations. Since
    // Boost notifies a variables_map in key order, a legacy default can
    // otherwise overwrite an explicit dotted alias.
    for (auto const& entry : migrations) {
        auto const found = values.find(entry.second);
        if (found == values.end() || found->second.defaulted()) {
            continue;
        }
        auto const* description = canonicalOptions.find_nothrow(entry.second, false);
        if (description != nullptr && description->semantic() != nullptr) {
            description->semantic()->notify(found->second.value());
        }
    }
}

} // namespace octotiger::optionsCompatibility
