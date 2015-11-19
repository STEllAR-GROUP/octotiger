/*
 * hpx_impl.hpp
 *
 *  Created on: Oct 1, 2015
 *      Author: dmarce1
 */

#ifndef HPX_IMPL_HPP_
#define HPX_IMPL_HPP_

namespace hpx {

template<class F, class ...Args>
aligned_t thread::wrapper(void* arg) {
	auto tup_ptr = reinterpret_cast<tuple_type<F, Args...>*>(arg);
	typedef typename std::result_of<F(Args...)>::type return_type;
	detail::invoke_function<return_type, F, Args...>(*tup_ptr);
	delete tup_ptr;
	return 0;
}

template<class F, class ...Args>
thread::thread(F&& f, Args&&...args) {

	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	static std::atomic<unsigned> current_shep(0);
	static unsigned num_sheps = qthread_num_shepherds();
	unsigned next_shep = ++current_shep;
	next_shep = next_shep % num_sheps;
	auto tup_ptr = new tuple_type<F,Args...>(std::make_tuple(std::forward<F>(f),std::forward<Args>(args)...));
	auto void_ptr = reinterpret_cast<void*>(tup_ptr);
	QCHECK(qthread_fork_to(&wrapper<F,Args...>, void_ptr,0, next_shep));
}

template<class Type>
hpx::future<typename std::decay<Type>::type> make_ready_future(Type&& arg) {
	typedef typename std::decay<Type>::type data_type;
	hpx::promise<data_type> promise;
	promise.set_value(std::forward<Type>(arg));
	return promise.get_future();
}

template<class Type, class ...Args>
hpx::future<id_type> new_(const id_type& id, Args&&...args) {
#ifndef HPX_ALWAYS_REMOTE
		if( id == find_here()) {
			return hpx::make_ready_future<id_type>(hpx::detail::create_component_direct<Type,Args...>(std::forward<Args>(args)...));
	} else {
#endif
//		auto fptr = &hpx::detail::create_component<Type, typename std::decay<Args>::type...>;
		//	auto name = reinterpret_cast<std::uintptr_t>(fptr);
		hpx::detail::ibuffer_type send_buffer;
		hpx::future < id_type > future;
		//	hpx::detail::naked_action_type name = Type::get_creator();
		const std::uintptr_t index = hpx::detail::promise_table_generate_future<id_type>(future);
		send_buffer << hpx::detail::invokation_type(NAME);
		send_buffer << Type::class_name;
		send_buffer << index;
		std::uintptr_t promise_function = reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_promise<id_type>);
		send_buffer << promise_function;
		Type tmp(std::forward<Args>(args)...);
		send_buffer << tmp;
//		hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
		hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
		return std::move(future);
#ifndef HPX_ALWAYS_REMOTE
	}
#endif
}

template<class Function, class ... Args>
typename std::enable_if<!std::is_void<typename std::result_of<Function(Args...)>::type>::value,
		future<typename std::result_of<Function(Args...)>::type>>::type async(Function&& f, Args&&... args ) {
	typedef typename std::result_of<Function(Args...)>::type value_type;
	auto promise_ptr = std::make_shared<hpx::promise<value_type>>();
	hpx::future<value_type> future = promise_ptr->get_future();
	hpx::thread([=](typename std::decay<Args>::type...these_args) {
				promise_ptr->set_value(f(std::move(these_args)...));
			},std::forward<Args>(args)...).detach();
	return std::move(future);

}

template<class Function, class ... Args>
typename std::enable_if<std::is_void<typename std::result_of<Function(Args...)>::type>::value, future<void>>::type async(
		Function&& f, Args&&... args ) {
			auto promise_ptr = std::make_shared<hpx::promise<void>>();
			hpx::future<void> future = promise_ptr->get_future();
			hpx::thread([=](typename std::decay<Args>::type...these_args) {
						f(std::move(these_args)...);
						promise_ptr->set_value();
					},std::forward<Args>(args)...).detach();

			return std::move(future);
		}

template<class Function, class ... Args>
future<typename Function::return_type> async(const id_type& id, Args&& ... args) {
	Function function;
	future<typename Function::return_type> future;
	if( id.get_rank() == find_here().get_rank() ) {
		future = hpx::async(function, id, std::forward<Args>(args)...);
	} else {
		future = function.get_action_future(id, std::forward<Args>(args)...);
	}
	return future;
}

namespace detail {

template<class R, class F, class ...Args, int ...S>
R invoke_function(std::tuple<typename std::decay<F>::type, typename std::decay<Args>::type...>& tup, int_seq<S...>) {
	return std::get < 0 > (tup)(std::forward<Args>(std::get<S>(tup))...);
}

template<class R, class ...Args>
R invoke_function(std::tuple<typename std::decay<Args>::type...>& tup) {
	return invoke_function<R, Args...>(tup, typename iota_seq<sizeof...(Args)>::type());
}

template<class Type>
std::uintptr_t promise_table_generate_future(hpx::future<Type>& future) {
	auto pointer = new hpx::promise<Type>();
	future = pointer->get_future();
	return reinterpret_cast<std::uintptr_t>(pointer);
}

template<class Type>
void promise_table_satisfy_promise(obuffer_type data) {
	std::uintptr_t index;
	data >> index;
	auto pointer = reinterpret_cast<hpx::promise<Type>*>(index);
	typename std::remove_reference<typename std::remove_const<Type>::type>::type argument;
	data >> argument;
	pointer->set_value(std::move(argument));
	delete pointer;
}

template<class T, class ...Args>
void pack_args(ibuffer_type& buffer, T&& arg1, Args&&...rest) {
	buffer << arg1;
	pack_args(buffer, std::forward<Args>(rest)...);
}

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
			assert(fptr);
			obuffer << (*fptr)(std::forward<ArgsDone>(args_done)...);
		}

template<class ...Args, class ...ArgsDone>
typename std::enable_if<std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		void (*fptr)(Args...), ibuffer_type& obuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
			assert(fptr);
			(*fptr)(std::forward<ArgsDone>(args_done)...);
		}

template<class ReturnType, class ...Args, class ...ArgsDone>
typename std::enable_if<!std::is_void<typename variadic_type<sizeof...(ArgsDone), Args...>::type>::value>::type invoke(
		ReturnType (*fptr)(Args...), ibuffer_type& ibuffer, const obuffer_type& buffer, ArgsDone&&...args_done) {
			assert(fptr);
			typedef typename variadic_type<sizeof...(ArgsDone), Args...>::type next_type;
			typename std::remove_const<typename std::remove_reference<next_type>::type>::type next_data;
			buffer >> next_data;
			invoke(fptr, ibuffer, buffer, std::forward<ArgsDone>(args_done)..., std::forward<next_type>(next_data));
		}

template<class Type, class ...Args>
void create_component(hpx::detail::ibuffer_type& return_buffer, hpx::detail::obuffer_type buffer) {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	Type* ptr;
	ptr = new Type();
	buffer >> *ptr;
	auto entry = new detail::agas_entry_t(reinterpret_cast<std::uintptr_t>(ptr), [](void* ptr) {
		auto _ptr = reinterpret_cast<Type*>(ptr);
		delete _ptr;
	});
	std::uintptr_t del_index = reinterpret_cast<std::uintptr_t>(entry);
	auto ptrptr = new (detail::agas_entry_t*)(entry);
	id.address = std::shared_ptr<detail::agas_entry_t*>(ptrptr, [=](detail::agas_entry_t** ptrptr) {
		hpx::detail::add_ref_count(del_index, -1);
		delete ptrptr;
	});
	id.rank = rank;
	return_buffer << id;
}

template<class Type, class ...Args>
id_type create_component_direct(Args ... args) {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	Type* ptr;
	ptr = new Type(std::move(args)...);
	auto entry = new detail::agas_entry_t(reinterpret_cast<std::uintptr_t>(ptr), [](void* ptr) {
		auto _ptr = reinterpret_cast<Type*>(ptr);
		delete _ptr;
	});
	std::uintptr_t del_index = reinterpret_cast<std::uintptr_t>(entry);
	auto ptrptr = new (detail::agas_entry_t*)(entry);
	id.address = std::shared_ptr<detail::agas_entry_t*>(ptrptr, [=](detail::agas_entry_t** ptrptr) {
		hpx::detail::add_ref_count(del_index, -1);
		delete ptrptr;
	});
	id.rank = rank;
	return id;
}

}

namespace serialization {

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		oarchive& arc, Type& data_out, const unsigned) {
	assert(arc.start + sizeof(Type) <= arc.size());
	memcpy(&data_out, arc.data() + arc.start, sizeof(Type));
	arc.start += sizeof(Type);
}

template<class Type>
typename std::enable_if<std::is_fundamental<typename std::remove_reference<Type>::type>::value>::type serialize(
		iarchive& arc, Type& data_in, const unsigned) {
	auto new_size = arc.size() + sizeof(Type);
	if (arc.capacity() < new_size) {
		arc.reserve(std::max(2 * arc.capacity(), arc.initial_size));
	}
	void* pointer = arc.data() + arc.size();
	arc.resize(new_size);
	memcpy(pointer, &data_in, sizeof(Type));
}

template<class Arc, class Type>
void serialize(Arc& arc, std::vector<Type>& vec, const unsigned) {
	auto size = vec.size();
	auto capacity = vec.capacity();
	arc & size;
	arc & capacity;
	if (capacity != vec.capacity()) {
		vec.reserve(capacity);
	}
	if (size != vec.size()) {
		vec.resize(size);
	}
	for (auto i = vec.begin(); i != vec.end(); ++i) {
		arc & (*i);
	}
}

template<class Arc, class Type, std::size_t Size>
void serialize(Arc& arc, std::array<Type, Size>& a, const unsigned) {
	for (auto i = a.begin(); i != a.end(); ++i) {
		arc & (*i);
	}
}

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::set<Type>& s, const unsigned v) {
	const std::size_t sz = s.size();
	arc << sz;
	for (auto i = s.begin(); i != s.end(); ++i) {
		arc << (*i);
	}
}

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::set<Type>& s, const unsigned v) {
	std::size_t sz;
	arc >> sz;
	s.clear();
	Type element;
	for (std::size_t i = 0; i != sz; ++i) {
		arc >> element;
		s.insert(element);
	}
}

template<class Type>
void serialize(hpx::detail::ibuffer_type& arc, std::list<Type>& list, const unsigned v) {
	const std::size_t sz = list.size();
	arc << sz;
	for (auto& ele : list) {
		arc << ele;
	}
}

template<class Type>
void serialize(hpx::detail::obuffer_type& arc, std::list<Type>& list, const unsigned v) {
	std::size_t sz;
	arc >> sz;
	list.clear();
	Type ele;
	for (std::size_t i = 0; i != sz; ++i) {
		arc >> ele;
		list.push_back(ele);
	}
}

template<class Arc, class Type1, class Type2>
void serialize(Arc& arc, std::pair<Type1, Type2>& a, const unsigned) {
	arc & a.first;
	arc & a.second;
}

template<class Arc>
void serialize(Arc& arc, std::string& vec, const unsigned) {
	auto size = vec.size();
	auto capacity = vec.capacity();
	arc & size;
	arc & capacity;
	if (capacity != vec.capacity()) {
		vec.reserve(capacity);
	}
	if (size != vec.size()) {
		vec.resize(capacity);
	}
	for (auto i = vec.begin(); i != vec.end(); ++i) {
		arc & (*i);
	}
}

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const long v) -> decltype(serialize(arc, data, v), void()) {
	serialize(arc, data, v);
}

template<class Archive, class Type>
auto serialize_imp(Archive& arc, Type& data, const unsigned v) -> decltype(data.serialize(arc, v), void()) {
	data.serialize(arc, v);
}

template<class Arc, class T>
Arc& operator<<(Arc& arc, const T& data) {
	serialize_imp(arc, const_cast<T&>(data), unsigned(0));
	return arc;
}

template<class Arc, class T>
const Arc& operator>>(const Arc& arc, T& data) {
	serialize_imp(const_cast<Arc&>(arc), data, unsigned(0));
	return arc;
}

}

template<class T>
bool future<T>::valid() const {
	return state != nullptr;
}

template<class T>
bool future<T>::is_ready() const {
	return state->is_ready();
}

template<class T>
void future<T>::wait() {
	state->wait();
}

template<class T>
T future<T>::get() {
	assert(state != nullptr);
	return state->get();
}

template<class Function>
future<typename std::result_of<Function(hpx::future<void>&&)>::type>
future<void>::then(Function&& function) {
	typedef typename std::result_of<Function(hpx::future<void>&&)>::type then_type;
	auto promise_ptr = std::make_shared<hpx::promise<then_type>>();
	auto future_ptr = std::make_shared<hpx::future<void>>();
	*future_ptr = std::move(*this);
	auto function_ptr = std::make_shared<Function>(std::forward<Function>(function));
	auto return_future = promise_ptr->get_future();
	hpx::thread([=]() {
				future_ptr->wait();
				(*function_ptr)(std::move(*future_ptr));
				promise_ptr->set_value();
			}).detach();

	return std::move(return_future);
}

template<class Archive>
void future<void>::load(const Archive& arc, const unsigned) {
	bool is_valid;
	arc >> is_valid;
	if (is_valid) {
		*this = hpx::make_ready_future();
	} else {
		*this = hpx::future<void>();
	}
}

template<class Archive>
void future<void>::save(Archive& arc, const unsigned) const {
	bool is_valid;
	is_valid = this->valid();
	arc << is_valid;
	if (is_valid) {
		const_cast<future<void>&>(*this).get();
	}
}

template<class T>
future<T>::future(hpx::future<hpx::future<T>>&& other ) {
	*this = std::move(other);
}

template<class T>
future<T>& future<T>::operator=(hpx::future<hpx::future<T>>&& other ) {
	auto promise_ptr = std::make_shared<hpx::promise<T>>();
	auto future_ptr = std::make_shared<hpx::future<hpx::future<T>>>(std::move(other));
	*this = promise_ptr->get_future();
	hpx::thread([=]() {
				hpx::future<T> fut = future_ptr->get();
				promise_ptr->set_value(fut.get());
			}).detach();

	return *this;
}

template<class T>
template<class Function>
future<typename std::result_of<Function(hpx::future<T>&&)>::type>
future<T>::then(Function&& function) {
	typedef typename std::result_of<Function(hpx::future<T>&&)>::type then_type;
	auto promise_ptr = std::make_shared<hpx::promise<then_type>>();
	auto future_ptr = std::make_shared<hpx::future<T>>();
	*future_ptr = std::move(*this);
	auto function_ptr = std::make_shared<Function>(std::forward<Function>(function));
	auto return_future = promise_ptr->get_future();
	hpx::thread([=]() {
				future_ptr->wait();
				promise_ptr->set_value((*function_ptr)(std::move(*future_ptr)));
			}).detach();

	return std::move(return_future);
}

template<class T>
template<class Archive>
void future<T>::load(const Archive& arc, const unsigned) {
	T data;
	bool is_valid;
	arc >> is_valid;
	if (is_valid) {
		arc >> data;
		*this = hpx::make_ready_future<T>(std::move(data));
	} else {
		*this = hpx::future<T>();
	}
}

template<class T>
template<class Archive>
void future<T>::save(Archive& arc, const unsigned) const {
	bool is_valid;
	is_valid = this->valid();
	arc << is_valid;
	if (is_valid) {
		arc << const_cast<future<T>&>(*this).get();
	}
}

namespace detail {

template<class ReturnType, class ...Args>
action_base<ReturnType(Args...)>::action_base(const std::string& str) :
		name(str) {
}

template<class ReturnType, class ...Args2>
template<class ...Args>
ReturnType action_base<ReturnType(Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	ReturnType result;
#ifndef HPX_ALWAYS_REMOTE
		if (find_here() != id) {
#endif
		auto future = get_action_future(id, std::forward<Args>(args)...);
		result = future.get();
#ifndef HPX_ALWAYS_REMOTE
	} else {
		result = (*fptr)(std::forward<Args>(args)...);
	}
#endif
		return result;
	}

template<class ReturnType, class ...Args2>
template<class ...Args>
hpx::future<ReturnType> action_base<ReturnType(Args2...)>::get_action_future(const id_type& id, Args&& ...args) const {
	hpx::detail::ibuffer_type send_buffer;
	hpx::future<ReturnType> future;
	const std::uintptr_t index = promise_table_generate_future<ReturnType>(future);
	send_buffer << invokation_type(NAME);
	send_buffer << name;
	send_buffer << index;
	std::uintptr_t promise_function =
	reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_promise<ReturnType>);
	send_buffer << promise_function;
	hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
	hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
	return future;
}

template<class ...Args>
action_base<void(Args...)>::action_base(const std::string& str) :
		name(str) {
}

template<class ...Args2>
template<class ...Args>
hpx::future<void> action_base<void(Args2...)>::get_action_future(const id_type& id, Args&& ...args) const {
	hpx::detail::ibuffer_type send_buffer;
	hpx::future<void> future;
	const std::uintptr_t index = promise_table_generate_future(future);
	send_buffer << invokation_type(NAME);
	send_buffer << name;
	send_buffer << index;
	std::uintptr_t promise_function =
	reinterpret_cast<std::uintptr_t>(&hpx::detail::promise_table_satisfy_void_promise);
	send_buffer << promise_function;
	hpx::detail::pack_args(send_buffer, std::forward<Args>(args)...);
	hpx::detail::remote_action(id.get_rank(), std::move(send_buffer));
	return future;
}

template<class ...Args2>
template<class ...Args>
void action_base<void(Args2...)>::operator()(const id_type& id, Args&& ...args) const {
#ifndef HPX_ALWAYS_REMOTE
		if (find_here() != id) {
#endif
		auto future = get_action_future(id,std::forward<Args>(args)...);
		future.get();
#ifndef HPX_ALWAYS_REMOTE
	} else {
		(*fptr)(std::forward<Args>(args)...);
	}
#endif
	}

template<class ReturnType, class ...Args>
component_action_base<ReturnType(std::uintptr_t, Args...)>::component_action_base(const std::string& str) :
		action_base<ReturnType(std::uintptr_t, Args...)>::action_base(str) {
}

template<class ReturnType, class ...Args2>
template<class ... Args>
ReturnType component_action_base<ReturnType(std::uintptr_t, Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	return action_base<ReturnType(std::uintptr_t, Args2...)>::operator()(remote_id, agas_ptr, std::forward<Args>(args)...);
}

template<class ReturnType, class ...Args2>
template<class ...Args>
future<ReturnType> component_action_base<ReturnType(std::uintptr_t, Args2...)>::get_action_future(const id_type& id,
		Args&& ...args) const {
			id_type remote_id;
			remote_id.set_rank(id.get_rank());
			std::uintptr_t agas_ptr = id.get_address();
			return action_base<ReturnType(std::uintptr_t, Args2...)>::get_action_future(remote_id, std::move(agas_ptr), std::forward<Args>(args)...);
		}

template<class ...Args2>
template<class ...Args>
future<void> component_action_base<void(std::uintptr_t, Args2...)>::get_action_future(const id_type& id, Args&& ...args)const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	return action_base<void(std::uintptr_t, Args2...)>::get_action_future(remote_id, std::move(agas_ptr), std::forward<Args>(args)...);
}

template<class ...Args>
component_action_base<void(std::uintptr_t, Args...)>::component_action_base(const std::string& str) :
		action_base<void(std::uintptr_t, Args...)>(str) {
}

template<class ...Args2>
template<class ...Args>
void component_action_base<void(std::uintptr_t, Args2...)>::operator()(const id_type& id, Args&& ...args) const {
	id_type remote_id;
	remote_id.set_rank(id.get_rank());
	std::uintptr_t agas_ptr = id.get_address();
	action_base<void(std::uintptr_t, Args2...)>::operator()(remote_id, agas_ptr, std::forward<Args>(args)...);
}

template<class Type, class ...Args>
std::uintptr_t creator<Type, Args...>::call(Args&&...args) {
	return reinterpret_cast<std::uintptr_t>(new Type(std::forward<Args>(args)...));
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
struct component_action<ReturnType (ClassType::*)(Args...), Pointer> : public hpx::detail::component_action_base<
		ReturnType(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>;
	typedef ReturnType return_type;
	static ReturnType call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
struct component_action<ReturnType (ClassType::*)(Args...) const, Pointer> : public hpx::detail::component_action_base<
		ReturnType(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>;
	typedef ReturnType return_type;
	static ReturnType call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
struct component_action<void (ClassType::*)(Args...), Pointer> : public hpx::detail::component_action_base<
		void(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<void(std::uintptr_t, Args...)>;
	typedef void return_type;
	static void call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);

};

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
struct component_action<void (ClassType::*)(Args...) const, Pointer> : public hpx::detail::component_action_base<
		void(std::uintptr_t, Args...)> {
	using base_type = hpx::detail::component_action_base<void(std::uintptr_t, Args...)>;
	typedef void return_type;
	static void call(std::uintptr_t _this, Args ...args);
	static void invoke(hpx::detail::ibuffer_type& ibuf, hpx::detail::obuffer_type buffer);
	component_action(const std::string&);
};

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
ReturnType component_action<ReturnType (ClassType::*)(Args...), Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	return ((*__this).*ptr)(args...);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
void component_action<ReturnType (ClassType::*)(Args...), Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...)>
component_action<ReturnType (ClassType::*)(Args...), Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>::fptr = call;
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
ReturnType component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::call(std::uintptr_t _this,
		Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<const ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	return ((*__this).*ptr)(args...);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
void component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ReturnType, class ClassType, class ...Args, ReturnType (ClassType::*Pointer)(Args...) const>
component_action<ReturnType (ClassType::*)(Args...) const, Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<ReturnType(std::uintptr_t, Args...)>::fptr = call;
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
void component_action<void (ClassType::*)(Args...), Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	((*__this).*ptr)(std::forward<Args>(args)...);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
void component_action<void (ClassType::*)(Args...), Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...)>
component_action<void (ClassType::*)(Args...), Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<void(std::uintptr_t, Args...)>::fptr = call;
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
void component_action<void (ClassType::*)(Args...) const, Pointer>::call(std::uintptr_t _this, Args ...args) {
	auto ptr = Pointer;
	auto __this = reinterpret_cast<const ClassType*>(reinterpret_cast<hpx::detail::agas_entry_t*>(_this)->pointer);
	((*__this).*ptr)(args...);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
void component_action<void (ClassType::*)(Args...) const, Pointer>::invoke(hpx::detail::ibuffer_type& ibuf,
		hpx::detail::obuffer_type buffer) {
	hpx::detail::invoke(call, ibuf, buffer);
}

template<class ClassType, class ...Args, void (ClassType::*Pointer)(Args...) const>
component_action<void (ClassType::*)(Args...) const, Pointer>::component_action(const std::string& str) :
		base_type(str) {
	hpx::detail::component_action_base<void(std::uintptr_t, Args...)>::fptr = call;
}

}

namespace serialization {

template<class T>
iarchive& iarchive::operator&(const T& data) {
	*this << data;
	return *this;
}

template<class T>
const oarchive& oarchive::operator&(T& data) const {
	*this >> data;
	return *this;
}

}

namespace detail {
template<class T>
template<class V>
void shared_state<T>::set_value(V&& value) {
	data = std::forward<V>(value);
	ready = true;
}

template<class T>
template<class V>
void shared_state<T>::set_value(const V& value) {
	V tmp = value;
	set_value(std::move(tmp));
}

template<class T>
bool shared_state<T>::is_ready() const {
	return ready;
}

template<class T>
void shared_state<T>::wait() const {
	while (!ready) {
		hpx::this_thread::yield();
	}
}

template<class T>
T shared_state<T>::get() {
	wait();
	return std::move(data);
}
}

}

#endif /* HPX_IMPL_HPP_ */
