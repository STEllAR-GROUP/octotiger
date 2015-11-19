/*
 * hpx.hpp
 *
 *  Created on: Sep 17, 2015
 *      Author: dmarce1
 */

#ifndef HPX_HPP_
#define HPX_HPP_


#include <mpi.h>

#include <fenv.h>
#include <qthread.h>
#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <list>
#include <set>
#include <vector>

#include <cmath>

#include <fenv.h>
#include <unistd.h>

//#define HPX_ALWAYS_REMOTE
#define HPX_STACK_SIZE (std::size_t(1)<<15)

#define MAX_MESSAGE_SIZE (1 << 25)

#define HPX_DEFINE_COMPONENT_ACTION(class_name, method, action_name ) \
		class action_name : public hpx::detail::component_action< decltype(& class_name :: method), & class_name :: method > {\
			using base_type = hpx::detail::component_action< decltype(& class_name :: method), & class_name :: method >; \
		public: \
			action_name() : base_type( #action_name ) {} \
			static void register_me() { \
				hpx::detail::register_action_name( #action_name,  & action_name ::invoke); \
			} \
		};

#define HPX_REGISTER_ACTION( action_type)  \
		__attribute__((constructor)) \
		static void register_##action_type() { \
			action_type :: register_me(); \
		}
#define HPX_REGISTER_ACTION_DECLARATION( action_type)
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(base_type, class_name) \
	__attribute__((constructor)) \
	static void register_constructor_##class_name () { \
		base_type::register_( #class_name , &hpx::detail::create_component< class_name > );\
	}

#define HPX_DEFINE_PLAIN_ACTION(func, action_name)                                                \
		struct action_name : public hpx::detail::action_base<decltype(func)>  {                   \
			static void invoke( hpx::detail::ibuffer_type&, hpx::detail::obuffer_type);      \
			typedef hpx::detail::action_base<decltype(func)> base_type;                           \
			action_name() :                                                                       \
					base_type( #action_name ) {        \
			   fptr = &func;                                                                      \
			}                                                                                     \
		}                                                                                         \


#define HPX_REGISTER_PLAIN_ACTION( action_name )                                                         \
	    __attribute__((constructor))  \
	    static void register_##action_name() { \
			hpx::detail::register_action_name( #action_name,  & action_name ::invoke); \
		} \
		void action_name ::invoke(hpx::detail::ibuffer_type& ibuf,  hpx::detail::obuffer_type buffer) {  \
			action_name act;                                                                             \
			hpx::detail::invoke(act.fptr, ibuf, buffer );                                    \
        }

#define HPX_PLAIN_ACTION(func, name)           \
		HPX_DEFINE_PLAIN_ACTION(func, name );  \
		HPX_REGISTER_PLAIN_ACTION( name )

#define HPX_SERIALIZATION_SPLIT_MEMBER() \
		void serialize(hpx::detail::ibuffer_type& arc, const unsigned v) {                           \
			save(arc, v);                                                                      \
		}                                                                                      \
		void serialize(hpx::detail::obuffer_type& arc, const unsigned v) {                           \
			typedef typename std::remove_const<decltype(this)>::type this_type;                \
			const_cast<this_type>(this)->load(const_cast<hpx::detail::obuffer_type&>(arc), v); \
		}

#define HPX_ACTION_USES_HUGE_STACK(a)
#define HPX_ACTION_USES_LARGE_STACK(a)
#define HPX_ACTION_USES_MEDIUM_STACK(a)
#define HPX_ACTION_USES_SMALL_STACK(a)

#define QCHECK(call)                                         \
	if( (call) != 0 ) {                                        \
		printf( "Qthread returned error\n");                 \
		printf( "File: %s, Line: %i\n", __FILE__, __LINE__); \
		abort();                                             \
	}


#include "hpx_decl.hpp"

HPX_DEFINE_PLAIN_ACTION(hpx::detail::add_ref_count, action_add_ref_count);

#include "hpx_impl.hpp"

namespace hpx {
const static id_type invalid_id;
}

#endif /* HPX_HPP_ */
