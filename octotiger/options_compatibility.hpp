#ifndef OCTOTIGER_OPTIONS_COMPATIBILITY_HPP
#define OCTOTIGER_OPTIONS_COMPATIBILITY_HPP

#include <boost/program_options.hpp>

#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace octotiger::options_compatibility {

using migration = std::pair<std::string, std::string>; // legacy, canonical

template <class T>
void add_canonical_option(boost::program_options::options_description& options,
    std::vector<migration>& migrations, char const* canonical, char const* legacy,
    T* value) {
    std::string description = std::string("canonical spelling; legacy: --") + legacy;
    options.add_options()(canonical, boost::program_options::value<T>(value),
        description.c_str());
    migrations.emplace_back(legacy, canonical);
}

template <class T>
void add_canonical_multitoken_option(boost::program_options::options_description& options,
    std::vector<migration>& migrations, char const* canonical, char const* legacy,
    T* value) {
    std::string description = std::string("canonical spelling; legacy: --") + legacy;
    options.add_options()(canonical,
        boost::program_options::value<T>(value)->multitoken(), description.c_str());
    migrations.emplace_back(legacy, canonical);
}

inline void remember_supplied(std::set<std::string>& supplied,
    boost::program_options::parsed_options const& parsed) {
    for (auto const& option : parsed.options) {
        if (!option.string_key.empty() && !option.unregistered) {
            supplied.insert(option.string_key);
        }
    }
}

inline bool check_compatibility_spellings(std::set<std::string> const& supplied,
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

inline void warn_legacy_spellings(std::set<std::string> const& supplied,
    std::vector<migration> const& migrations, std::ostream& warning = std::cerr) {
    std::set<std::string> warned;
    for (auto const& entry : migrations) {
        if (supplied.count(entry.first) != 0 && warned.insert(entry.first).second) {
            warning << "WARNING: option '" << entry.first
                    << "' is deprecated; use '" << entry.second << "'.\n";
        }
    }
}

inline void reapply_canonical_values(boost::program_options::variables_map const& values,
    boost::program_options::options_description const& canonical_options,
    std::vector<migration> const& migrations) {
    // Legacy entries carry the historical default_value declarations. Since
    // Boost notifies a variables_map in key order, a legacy default can
    // otherwise overwrite an explicit dotted alias.
    for (auto const& entry : migrations) {
        auto const found = values.find(entry.second);
        if (found == values.end() || found->second.defaulted()) {
            continue;
        }
        auto const* description = canonical_options.find_nothrow(entry.second, false);
        if (description != nullptr && description->semantic() != nullptr) {
            description->semantic()->notify(found->second.value());
        }
    }
}

} // namespace octotiger::options_compatibility

#endif
