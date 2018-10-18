/*
 * options_enum.hpp
 *
 *  Created on: Oct 10, 2018
 *      Author: dmarce1
 */

#ifndef SRC_OPTIONS_ENUM_HPP_
#define SRC_OPTIONS_ENUM_HPP_


#define COMMAND_LINE_ENUM( enum_name, args... )               \
	enum enum_name : std::uint32_t {                                       \
		args                                                               \
	};                                                                     \
	static inline std::istream& operator>>(std::istream& in, enum_name& e) \
	{                                                                      \
		std::vector<std::string> strings;                                  \
		boost::split(strings, #args, boost::is_any_of(" ,\t"));            \
		static enum_name enums[] = {args};                                 \
		bool success = false;                                              \
		std::string token;                                                 \
	    in >> token;                                                       \
		for( std::size_t i = 0; i < strings.size(); i++ ) {                \
			if( boost::iequals(strings[i], token)) {                       \
				success = true;                                            \
				e = enums[i];                                              \
			}                                                              \
		}                                                                  \
	   if( !success ) {                                                    \
	        in.setstate(std::ios_base::failbit);                           \
		}                                                                  \
	    return in;                                                         \
	}                                                                      \
	static inline std::string to_string(enum_name e) {                     \
		std::vector<std::string> strings;                                  \
		boost::split(strings, #args, boost::is_any_of(" ,\t"));            \
		static enum_name enums[] = {args};                                 \
		static int sz1 = std::vector<enum_name>(std::begin(enums),std::end(enums)).size(); \
		if(  sz1 != strings.size() ) {\
			printf( "Different sizes %i %i\n", int(sz1), int(strings.size())); \
		}\
		std::string rc;                                                    \
	 	for( std::size_t i = 0; i < strings.size(); i++ ) {                \
			if( e == enums[i])  {                                          \
				rc = strings[i];                                           \
				break;                                                     \
			}                                                              \
		}                                                                  \
		if( rc.empty() ) {                                                 \
			std::cout << "invalid enum value for " << #enum_name;          \
			std::cout << "\n";                                             \
			throw;                                                         \
		}                                                                  \
	    return rc;                                                         \
	}                                                                      \





#endif /* SRC_OPTIONS_ENUM_HPP_ */
