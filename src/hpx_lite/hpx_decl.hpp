/*
 * hpx_decl.hpp
 *
 *  Created on: Oct 1, 2015
 *      Author: dmarce1
 */

#ifndef HPX_DECL_HPP_
#define HPX_DECL_HPP_

/************FORWARD DECLARATIONS******************/

namespace hpx {

class id_type;

template<class T>
class promise;
template<class T>
class future;
class mutex;
class thread;

namespace lcos {
namespace local {

using spinlock = hpx::mutex;

using mutex =hpx::mutex;

}}

hpx::future<void> make_ready_future();
namespace detail {

template<class >
class shared_state;

typedef char invokation_type;
#define NAME  1
#define ADDRESS 2
#define PROMISE 3

struct agas_entry_t;
}

namespace detail {

int& mpi_comm_rank();

int& mpi_comm_size();
template<int Index, class ...Args>
struct variadic_type;

template<class T>
struct action_base;

template<class T>
struct component_action_base;

template<class MethodType, MethodType MethodPtr>
struct component_action;

}

namespace serialization {
class archive;
class oarchive;
class iarchive;
}

/****************************************************typedefs***************************/
typedef std::int8_t byte_t;
typedef id_type gid_type;
namespace detail {
typedef void (*deleter_type)(void*);
typedef hpx::serialization::oarchive obuffer_type;
typedef hpx::serialization::iarchive ibuffer_type;
typedef void (*naked_action_type)(ibuffer_type&, obuffer_type);
typedef void (*promise_action_type)(obuffer_type);
}

/****************************************************Function decs***************************/

int get_locality_id();
std::vector<hpx::id_type> find_all_localities();
int finalize();
hpx::id_type find_here();

namespace this_thread {
void yield();
}

namespace naming {
int get_locality_id_from_gid(const gid_type&);
}

namespace detail {

void register_action_name(const std::string& strname, naked_action_type fptr);
naked_action_type lookup_action(const std::string& strname);

void add_ref_count(std::uintptr_t, int change);
void pack_args(ibuffer_type& buffer);
int server(int, char*[]);
void remote_action(int target, obuffer_type&& buffer );
void set_exit_signal();
std::uintptr_t promise_table_generate_future(hpx::future<void>&);
void promise_table_satisfy_void_promise(obuffer_type data);
}

template<class Type, class ...Args>
hpx::future<id_type> new_(const id_type& id, Args&&...args);

template<class Function, class ... Args>
typename std::enable_if<!std::is_void<typename std::result_of<Function(Args...)>::type>::value,
		future<typename std::result_of<Function(Args...)>::type>>::type async(Function&& f, Args&&... args );

template<class Function, class ... Args>
typename std::enable_if<std::is_void<typename std::result_of<Function(Args...)>::type>::value, future<void>>::type async(
		Function&& f, Args&&... args );

template<class Function, class ... Args>
future<typename Function::return_type> async(const id_type& id, Args&& ... args);

namespace detail {

template<class Type>
std::uintptr_t promise_table_generate_future(hpx::future<Type>& future);

template<class Type>
void promise_table_satisfy_promise(obuffer_type data);

template<class T, class ...Args>
void pack_args(ibuffer_type& buffer, T&& arg1, Args&&...rest);

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		void (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<!std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& ibuffer, const obuffer_type& buffer, ArgsDone&&...args_done);

template<class Type, class ...Args>
void create_component(hpx::detail::ibuffer_type& return_buffer, hpx::detail::obuffer_type buffer);

template<class Type, class ...Args>
id_type create_component_direct(Args ... args);
}

namespace serialization {

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		oarchive& arc, Type& data_out, const unsigned);

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		iarchive& arc, Type& data_in, const unsigned);

template<class Arc, class Type>
void serialize(Arc& arc, std::vector<Type>& vec, const unsigned);

template<class Arc, class Type, std::size_t Size>
void serialize(Arc& arc, std::array<Type, Size>& a, const unsigned);

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::set<Type>& s, const unsigned v);

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::set<Type>& s, const unsigned v);

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::list<Type>& s, const unsigned v);

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::list<Type>& s, const unsigned v);

template<class Arc, class Type1, class Type2>
void serialize(Arc& arc, std::pair<Type1, Type2>& a, const unsigned);

template<class Arc>
void serialize(Arc& arc, std::string& vec, const unsigned);

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const long v) -> decltype(serialize(arc, data, v), void());

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const unsigned v) -> decltype(data.serialize(arc, v), void());

template<class Arc, class T>
Arc& operator<<(Arc& arc, const T& data);

template<class Arc, class T>
const Arc& operator>>(const Arc& arc, T& data);

}

namespace this_thread {
void yield();
}

class thread {
private:

	template<class F, class...Args>
	using tuple_type = std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>;

	template<class F, class ...Args>
	static aligned_t wrapper(void* arg);
public:

	thread();
	template<class F, class ...Args>
	thread(F&& f, Args&&...args);
	thread(const thread&) = delete;
	thread(thread&&) = default;
	~thread() = default;
	thread& operator=(const thread&) = delete;
	thread& operator=(thread&&) = default;
	void detach();
};

namespace components {
template<class Base>
struct managed_component_base {
	static std::string class_name;
	static hpx::detail::naked_action_type fptr;
	static void register_(std::string _class_name, hpx::detail::naked_action_type _fptr) {
		class_name = _class_name;
		fptr = _fptr;
		hpx::detail::register_action_name(class_name, fptr);
	}
};

template<class Base>
using managed_component = managed_component_base<Base>;

template<class Base>
std::string managed_component_base<Base>::class_name;

template<class Base>
hpx::detail::naked_action_type managed_component_base<Base>::fptr;

}

namespace detail {

template<int...>
struct int_seq {
};

template<int N, int ...S>
struct iota_seq: iota_seq<N - 1, N - 1, S...> {
};

template<int ...S>
struct iota_seq<1, S...> {
	typedef int_seq<S...> type;
};

template<class R, class F, class ...Args, int ...S>
R invoke_function(std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>& tup, int_seq<S...>);

template<class R, class ...Args>
R invoke_function(std::tuple<typename std::decay<Args>::type...>& tup);

}

template<class T>
class future {
private:
	std::shared_ptr<detail::shared_state<T>> state;
public:
	future() = default;
	future( const future&) = delete;
	future( future&&) = default;
	~future() = default;
	future& operator=( const future&) = delete;
	future& operator=( future&&) = default;
	void wait();
	bool valid() const;
	T get();
	future(hpx::future<hpx::future<T>>&& other );
	future& operator=(hpx::future<hpx::future<T>>&& other );
	template<class Function>
	future<typename std::result_of<Function(hpx::future<T>&&)>::type>
	then(Function&& function);
	template<class Archive>
	void load(const Archive& arc, const unsigned );
	template<class Archive>
	void save(Archive& arc, const unsigned ) const;
	HPX_SERIALIZATION_SPLIT_MEMBER();
	friend class
	promise<T>;
};

template<>
class future<void> {
private:
	std::shared_ptr<detail::shared_state<void>> state;
public:
	future() = default;
	future( const future&) = delete;
	future( future&&) = default;
	~future() = default;
	future& operator=( const future&) = delete;
	future& operator=( future&&) = default;
	void wait();
	void get();
	bool valid() const;
	future(hpx::future<hpx::future<void>>&& other );
	future& operator=(hpx::future<hpx::future<void>>&& other );
	template<class Function>
	future<typename std::result_of<Function(hpx::future<void>&&)>::type>
	then(Function&& function);
	template<class Archive>
	void load(const Archive& arc, const unsigned );
	template<class Archive>
	void save(Archive& arc, const unsigned ) const;
	HPX_SERIALIZATION_SPLIT_MEMBER();
	friend class
	promise<void>;
};

class id_type {
	int rank;
	std::shared_ptr<detail::agas_entry_t*> address;

	void set_rank(int);
	void set_address(std::uintptr_t);
	std::uintptr_t get_address() const;
	int get_rank() const;

public:
	id_type();
	id_type(const id_type& other);
	id_type(id_type&& other);
	~id_type() = default;
	id_type& operator=(const id_type& other);
	id_type& operator=(id_type&& other );
	bool operator==(const id_type&) const;
	bool operator!=(const id_type&) const;
	gid_type get_gid() const;

	friend int finalize();
	friend std::vector<hpx::id_type> find_all_localities();
	friend hpx::detail::ibuffer_type& operator<<(hpx::detail::ibuffer_type& buffer, const id_type& id);
	friend const hpx::detail::obuffer_type& operator>>(const hpx::detail::obuffer_type& buffer, id_type&);
	friend int hpx::naming::get_locality_id_from_gid(const gid_type& gid);
	friend id_type find_here();

	template<class>
	friend struct hpx::detail::action_base;

	template<class>
	friend struct hpx::detail::component_action_base;

	template<class Type, class ...Args>
	friend hpx::future<id_type> new_(const id_type& id, Args&&...args);

	template<class Type, class ...Args>
	friend void hpx::detail::create_component(hpx::detail::ibuffer_type&, hpx::detail::obuffer_type);

	template<class Type, class ...Args>
	friend id_type hpx::detail::create_component_direct(Args...);

	template<class Function, class ... Args>
	friend typename std::enable_if<!std::is_void<typename std::result_of<Function(Args...)>::type>::value,
	future<typename std::result_of<Function(Args...)>::type>>::type async(Function&& f, Args&&... args );

	template<class Function, class ... Args>
	friend typename std::enable_if<std::is_void<typename std::result_of<Function(Args...)>::type>::value, future<void>>::type async(
			Function&& f, Args&&... args );

	template<class Function, class ... Args>
	friend future<typename Function::return_type> async(const id_type& id, Args&& ... args);
};

class mutex {
private:
	std::atomic<int> locked;
	std::atomic<int> waiting;
public:
	constexpr mutex() :
			locked(0), waiting(false) {
	}
	mutex(const mutex&) = delete;
	mutex(mutex&&) = delete;
	~mutex() = default;
	mutex& operator=(const mutex&) = delete;
	mutex& operator=(mutex&&) = delete;
	void lock();
	void unlock();
};

namespace detail {

struct agas_entry_t {
	std::uintptr_t pointer;
	hpx::mutex mtx;
	int reference_count;
	deleter_type deleter;

	agas_entry_t(std::uintptr_t, deleter_type);
}
;

template<class ReturnType, class ...Args>
struct component_action_base<ReturnType(std::uintptr_t, Args...)> : public action_base<
		ReturnType(std::uintptr_t, Args...)> {
	component_action_base(const std::string& str);

	template<class ...Args2>
	ReturnType operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<ReturnType> get_action_future(const id_type& id, Args2&& ...args) const;
};

template<class ...Args>
struct component_action_base<void(std::uintptr_t, Args...)> : public action_base<void(std::uintptr_t, Args...)> {
	component_action_base(const std::string& str);

	template<class ...Args2>
	void operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<void> get_action_future(const id_type& id, Args2&& ...args) const;
}
;

template<class ReturnType, class ...Args>
struct action_base<ReturnType(Args...)> {
	typedef ReturnType return_type;
	const std::string name;
	ReturnType (*fptr)(Args...);

	action_base(const std::string&);

	template<class ...Args2>
	ReturnType operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<ReturnType> get_action_future(const id_type& id, Args2&& ...args) const;
};

template<class ...Args>
struct action_base<void(Args...)> {
	typedef void return_type;

	const std::string name;

	void (*fptr)(Args...);

	action_base(const std::string& str);

	template<class ...Args2>
	void operator()(const id_type& id, Args2&& ...args) const;

	template<class...Args2>
	hpx::future<void> get_action_future(const id_type& id, Args2&& ...args) const;
};

template<int Index, class NextType, class ...Args>
struct variadic_type<Index, NextType, Args...> {
	typedef typename variadic_type<Index - 1, Args...>::type type;
};

template<int Index>
struct variadic_type<Index> {
	typedef void type;
};

template<class NextType, class ...Args>
struct variadic_type<0, NextType, Args...> {
	typedef NextType type;
};

template<class Type, class ...Args>
struct creator {
	static std::uintptr_t call(Args&&...args);
};

}

namespace serialization {

class archive: public std::vector<byte_t> {
protected:
	typedef std::vector<byte_t> buffer_type;
	static const std::size_t initial_size;
	mutable std::size_t start;
public:
	archive(std::size_t = 0);
	~archive() = default;
	archive( const archive& ) = default;
	archive( archive&& ) = default;
	archive& operator=( const archive& ) = default;
	archive& operator=( archive&& ) = default;
	std::size_t current_position() const;
	void seek_to(std::size_t);
};

class iarchive: public archive {
public:
	iarchive(std::size_t = 0);
	~iarchive() = default;
	iarchive( const iarchive& ) = default;
	iarchive( iarchive&& ) = default;
	iarchive& operator=( const iarchive& ) = default;
	iarchive& operator=( iarchive&& ) = default;
	iarchive( const oarchive& );
	iarchive( oarchive&& );
	iarchive& operator=( const oarchive& );
	iarchive& operator=( oarchive&& );

	template<class Type>
	friend typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type
	serialize(iarchive& arc,Type& data_in, const unsigned);

	template<class T>
	iarchive& operator&(const T& data);
};

class oarchive: public archive {
public:
	oarchive(std::size_t = 0);
	oarchive(const oarchive&) = default;
	oarchive( oarchive&& ) = default;
	oarchive( const iarchive& );
	oarchive( iarchive&& );
	~oarchive() = default;
	oarchive& operator=( const oarchive& ) = default;
	oarchive& operator=( oarchive&& ) = default;
	oarchive& operator=( const iarchive& );
	oarchive& operator=( iarchive&& );

	template<class T>
	const oarchive& operator&(T& data) const;

	template<class Type>
	friend typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value >::type
	serialize(oarchive&, Type& data_out, const unsigned );

};

}

namespace detail {
template<class T>
class shared_state {
private:
	T data;
	bool ready;
public:
	shared_state() :
			ready(false) {
	}
	shared_state(const shared_state&) = delete;
	shared_state(shared_state&&) = delete;
	~shared_state() = default;
	shared_state& operator=( const shared_state&) = delete;
	shared_state& operator=(shared_state&&) = delete;
	template<class V>
	void set_value(V&& value);
	template<class V>
	void set_value(const V& value);
	void wait() const;
	T get();
};

template<>
class shared_state<void> {
private:
	bool ready;
public:
	shared_state() :
			ready(false) {
	}
	shared_state(const shared_state&) = delete;
	shared_state(shared_state&&) = delete;
	~shared_state() = default;
	shared_state& operator=( const shared_state&) = delete;
	shared_state& operator=(shared_state&&) = delete;
	void set_value();
	void wait() const;
	void get();
};}

template<class T>
class promise {
private:
	std::shared_ptr<detail::shared_state<T>> state;
public:
	promise();
	promise(const promise&) = delete;
	promise( promise&&) = default;
	~promise() = default;
	promise& operator=( const promise&) = delete;
	promise& operator=( promise&&) = default;
	template<class V>
	void set_value(V&& v);
	template<class V>
	void set_value(const V& v);
	future<T> get_future() const;
};

template<>
class promise<void> {
private:
	std::shared_ptr<detail::shared_state<void>> state;
public:
	promise();
	promise(const promise&) = delete;
	promise( promise&&) = default;
	~promise() = default;
	promise& operator=( const promise&) = delete;
	promise& operator=( promise&&) = default;
	void set_value();
	future<void> get_future() const;
};

template<class T>
promise<T>::promise() :
		state(std::make_shared<detail::shared_state<T>>()) {
}

template<class T>
template<class V>
void promise<T>::set_value(V&& v) {
	state->set_value(std::forward<V>(v));
}

template<class T>
template<class V>
void promise<T>::set_value(const V& v) {
	state->set_value(v);
}

template<class T>
future<T> promise<T>::get_future() const {
	future<T> fut;
	fut.state = state;
	return std::move(fut);
}

}

#endif /* HPX_DECL_HPP_ */
