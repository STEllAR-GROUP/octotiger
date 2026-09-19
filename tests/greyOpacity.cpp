#include "octotiger/radiation/grey_opacity.hpp"
#include <any>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

void check(bool value, char const* what) { if(!value)throw std::runtime_error(what); }
template<class F> void rejects(F f) {
    bool threw=false;try { f(); } catch(std::runtime_error const&) {threw=true;}
    check(threw,"expected invalid opacity to fail");
}
struct Archive {
    std::vector<std::any> values;
    std::size_t cursor=0;
    bool reading=false;
    template<class T> Archive& operator&(T& value) {
        if(reading)value=std::any_cast<T>(values.at(cursor++));else values.emplace_back(value);
        return *this;
    }
};
int main() {
    using radiation::GreyOpacity;
    try {
        GreyOpacity defaults;defaults.validate(-1);
        check(defaults.model=="legacy","historical default model");
        auto o=defaults;o.model="skinner_ostriker";o.validate(.7);
        rejects([&]{o.validate(-1);});
        o.model="grey";o.validate(-1);
        rejects([&]{o.validate(0);});
        for(double bad:{-2.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            auto b=o;b.absorption=bad;rejects([&]{b.validate(-1);});
            b=o;b.scattering=bad;rejects([&]{b.validate(-1);});
            b=o;b.transport_absorption=bad;rejects([&]{b.validate(-1);});
        }
        o.transport_absorption=-.5;rejects([&]{o.validate(-1);});o.transport_absorption=-1;
        o.units="code";rejects([&]{o.validate(-1);});o.units="cm2/g";
        o.model="ensman";rejects([&]{o.validate(-1);});o.model="grey";
        o.absorption=2;o.scattering=3;o.transport_absorption=.5;
        rejects([&]{radiation::greyCoefficients(o,-1,1,1);});
        rejects([&]{radiation::greyCoefficients(o,1,1,0);});
        auto z=radiation::greyCoefficients(o,0,1,1);
        check(z.absorption==0 && z.transport==0,"vacuum mass extinction");
        // Production HPX member serialization, with a type-preserving test archive.
        Archive arc;o.serialize(arc,0);arc.reading=true;
        GreyOpacity restored;restored.serialize(arc,0);
        check(restored.model==o.model && restored.units==o.units && restored.absorption==2 &&
              restored.scattering==3 && restored.transport_absorption==.5,"HPX field round trip");
        // Exact production checkpoint codec, including pre-Step-04 and corrupt files.
        std::map<std::string,double> fields;
        auto wi=[&](char const* k,int v){fields[k]=v;};
        auto wr=[&](char const* k,double v){fields[k]=v;};
        auto load=[&]{return radiation::loadGreyOpacity(
            [&](char const* k){return fields.count(k);},
            [&](char const* k){return int(fields.at(k));},
            [&](char const* k){return fields.at(k);});};
        check(load().model=="legacy","old checkpoint retains historical model");
        for(auto model:{"legacy","skinner_ostriker","grey"})for(auto units:{"cm2/g","1/cm"}) {
            o.model=model;o.units=units;radiation::saveGreyOpacity(o,wi,wr);
            auto r=load();
            check(r.model==model && r.units==units && r.absorption==2 && r.scattering==3 &&
                  r.transport_absorption==.5,"checkpoint opacity round trip");
        }
        auto loaded=load();auto explicitValues=GreyOpacity{};explicitValues.scattering=9;
        radiation::restoreGreyOpacityOverrides(loaded,explicitValues,
            [](std::string key){return key=="radiation.opacity.scattering";});
        check(loaded.model=="grey" && loaded.scattering==9 && loaded.absorption==2,
            "explicit options override only supplied checkpoint fields");
        auto old=GreyOpacity{};
        radiation::restoreGreyOpacityOverrides(old,o,[](char const*){return true;});
        check(old.model==o.model && old.units==o.units && old.absorption==o.absorption,
            "explicit grey config survives old checkpoint load");
        fields["rad_opacity_schema"]=2;rejects(load);
        fields["rad_opacity_schema"]=1;fields["rad_opacity_model"]=99;rejects(load);
        fields["rad_opacity_model"]=2;fields.erase("rad_opacity_scattering");rejects(load);
        std::cout<<"GreyOpacity: validation, vacuum, serialization and checkpoint compatibility passed\n";
    } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
