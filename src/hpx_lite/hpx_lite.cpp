/*
 * hpx.cpp
 *
 *  Created on: Sep 22, 2015
 *      Author: dmarce1
 */

#include "hpx_lite.hpp"
#include <list>
#include <unistd.h>
#include <unordered_map>

#define CHECK_MPI_RETURN_CODE(code) check_mpi_return_code((code), __FILE__, __LINE__)

static bool exit_signal = false;

namespace hpx {

namespace this_thread {
void yield() {
	qthread_yield();
}
}

thread::thread() {
}

void thread::detach() {
}

namespace detail {

typedef std::pair<std::string, naked_action_type> action_dir_entry_type;
typedef std::unordered_map<std::string, naked_action_type> action_dir_type;

static action_dir_type& action_dir() {
	static action_dir_type a;
	return a;
}

static std::mutex& action_dir_mutex() {
	static std::mutex a;
	return a;
}

void register_action_name(const std::string& strname, naked_action_type fptr) {
	std::lock_guard < std::mutex > lock(action_dir_mutex());
	action_dir().insert(std::make_pair(strname, fptr));
}

naked_action_type lookup_action(const std::string& strname) {
	return action_dir()[strname];
}

int& mpi_comm_rank() {
	static int r;
	return r;
}
int& mpi_comm_size() {
	static int r;
	return r;
}
}
}

extern int hpx_main(int argc, char* argv[]);

namespace hpx {
namespace detail {
struct message_type {
	int target_rank;
	obuffer_type buffer;
	std::vector<MPI_Request> mpi_requests;
	int message_size;
	bool sent;
};

typedef std::list<std::shared_ptr<message_type>> message_queue_type;

std::uintptr_t promise_table_generate_future(hpx::future<void>& future) {
	auto pointer = new hpx::promise<void>();
	future = pointer->get_future();
	return reinterpret_cast<std::uintptr_t>(pointer);
}

void promise_table_satisfy_void_promise(obuffer_type data) {
	std::uintptr_t index;
	data >> index;
	auto pointer = reinterpret_cast<hpx::promise<void>*>(index);
	pointer->set_value();
	delete pointer;
}

message_queue_type message_queue;
hpx::lcos::local::mutex message_queue_mutex;

static void handle_incoming_message(int, obuffer_type);
static int handle_outgoing_message(std::shared_ptr<message_type>);
static void check_mpi_return_code(int, const char*, int);

}
}

static void local_finalize();

HPX_PLAIN_ACTION(local_finalize, finalize_action);

std::atomic<bool> main_exit_signal(false);

int main(int argc, char* argv[]) {
	char string[16];
	sprintf(string, "%i", int(HPX_STACK_SIZE));
	setenv("QTHREAD_STACK_SIZE", string, 1);
	setenv("QTHREAD_INFO", "1", 1);
	sprintf(string, "%i", int(std::thread::hardware_concurrency()));
#ifdef __MIC__
	setenv("QTHREAD_HWPAR", "61", 1);
//	setenv("QTHREAD_NUM_WORKERS_PER_SHEPHERD", "244", 1);
//	setenv("QTHREAD_NUM_SHEPHERDS", "1", 1);
#else
	setenv("QTHREAD_SHEPHERD_BOUNDARY", "socket", 1);
	setenv("QTHREAD_HWPAR", string, 1);
#endif
	QCHECK(qthread_initialize());
	int rank, provided, rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED) {
		printf("MPI threading insufficient\n");
		rc = EXIT_FAILURE;
	} else {
		MPI_Comm_rank(MPI_COMM_WORLD, &hpx::detail::mpi_comm_rank());
		MPI_Comm_size(MPI_COMM_WORLD, &hpx::detail::mpi_comm_size());
		rank = hpx::detail::mpi_comm_rank();
		if (rank == 0) {
			hpx::thread([](int a, char* b[]) {
				hpx_main(a, b);
				main_exit_signal = true;
			}, argc, argv).detach();

		}
		hpx::detail::server(argc, argv);
		rc = EXIT_SUCCESS;
	}
	MPI_Finalize();
	qthread_finalize();
	return rc;
}

static void local_finalize() {
	hpx::thread([]() {
		sleep(1);
		hpx::detail::set_exit_signal();
	}).detach();

}

namespace hpx {

gid_type id_type::get_gid() const {
	return *this;
}

int finalize() {
	hpx::thread([]() {
		while(!main_exit_signal) {
			hpx::this_thread::yield();
		}
		int size;
		size = hpx::detail::mpi_comm_size();
		for (int i = size - 1; i > 0; i--) {
			hpx::id_type id;
			id.set_rank(i);
			hpx::async<finalize_action>(id).get();
		}
		hpx::detail::set_exit_signal();
	}).detach();

	return EXIT_SUCCESS;
}

hpx::detail::ibuffer_type& operator<<(hpx::detail::ibuffer_type& buffer, const id_type& id) {
	buffer << id.rank;
	if (id.address != nullptr) {
		auto ptr = reinterpret_cast<std::uintptr_t>(*(id.address));
		buffer << ptr;
		action_add_ref_count act;
		id_type remote;
		remote.set_rank(id.rank);
		act(remote, ptr, +1);
	} else {
		buffer << std::uintptr_t(0);
	}
	return buffer;
}

const hpx::detail::obuffer_type& operator>>(const hpx::detail::obuffer_type& buffer, id_type& id) {
	std::uintptr_t ptr;
	buffer >> id.rank;
	buffer >> ptr;
	if (ptr) {
		action_add_ref_count act;
		auto ptrptr = new (detail::agas_entry_t*)(reinterpret_cast<detail::agas_entry_t*>(ptr));
		int remote_rank = id.rank;
		id.address = std::shared_ptr<detail::agas_entry_t*>(ptrptr, [=](detail::agas_entry_t** ptrptr) {
			id_type remote;
			remote.set_rank(remote_rank);
			act(remote, reinterpret_cast<std::uintptr_t>(*ptrptr), -1);
			delete ptrptr;
		});
	} else {
		id.address = nullptr;
	}
	return buffer;
}

int get_locality_id() {
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	return rank;
}

std::vector<hpx::id_type> find_all_localities() {
	int size;
	size = hpx::detail::mpi_comm_size();
	std::vector<hpx::id_type> localities(size);
	for (int i = 0; i < size; ++i) {
		localities[i].set_rank(i);
	}
	return localities;
}

std::uintptr_t id_type::get_address() const {
	if (address != nullptr) {
		return reinterpret_cast<std::uintptr_t>(*address);
	} else {
		return 0;
	}
}

void id_type::set_rank(int r) {
	rank = r;
}

int id_type::get_rank() const {
	return rank;
}

id_type::id_type() :
		rank(-1), address(nullptr) {
}

id_type& id_type::operator=(const id_type& other) {
	rank = other.rank;
	address = other.address;
	return *this;
}

id_type& id_type::operator=(id_type&& other ) {
	rank = std::move(other.rank);
	address = std::move(other.address);
	other.rank = -1;
	other.address = nullptr;
	return *this;
}

id_type::id_type(const id_type& other) {
	*this = other;
}

id_type::id_type(id_type&& other) {
	*this = std::move(other);
}

future<void>::future(hpx::future<hpx::future<void>>&& other ) {
	auto promise_ptr = std::make_shared<hpx::promise<void>>();
	auto future_ptr = std::make_shared<hpx::future<hpx::future<void>>>(std::move(other));
	*this = promise_ptr->get_future();
	hpx::thread([=]() {
				hpx::future<void> fut = future_ptr->get();
				fut.get();
				promise_ptr->set_value();
			}).detach();

}

void future<void>::wait() {
	state->wait();
}

void future<void>::get() {
	state->get();
}

promise<void>::promise() :
		state(std::make_shared<detail::shared_state<void>>()) {
}

void promise<void>::set_value() {
	state->set_value();
}

future<void> promise<void>::get_future() const {
	future<void> fut;
	fut.state = state;
	return std::move(fut);
}

namespace detail {

void add_ref_count(std::uintptr_t id, int change) {
	auto ptr = reinterpret_cast<agas_entry_t*>(id);
	std::unique_lock<hpx::mutex> lock(ptr->mtx);
	ptr->reference_count += change;
	assert(ptr->reference_count >= 0);
	if (ptr->reference_count == 0) {
		auto del_ptr = ptr->deleter;
		(*del_ptr)(reinterpret_cast<void*>(ptr->pointer));
		lock.unlock();
		delete ptr;
	}
}

agas_entry_t::agas_entry_t(std::uintptr_t id, deleter_type _deleter) {
	reference_count = 1;
	pointer = id;
	deleter = _deleter;
}

}

namespace detail {

void set_exit_signal() {
	exit_signal = true;
}

void remote_action(int target, obuffer_type&& buffer ) {
	auto message = std::make_shared<message_type>();
	message->target_rank = target;
	message->sent = false;
	message->buffer = std::move(buffer);
	std::lock_guard<hpx::lcos::local::mutex> lock(message_queue_mutex);
	message_queue.push_back(message);
}

int server(int argc, char* argv[]) {
	obuffer_type buffer;
	bool found_stuff;
	do {
		found_stuff = false;
		int rc;
		MPI_Status stat;
		int flag;
		do {
			flag = 0;
			int count, src;
			rc = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &stat);
			CHECK_MPI_RETURN_CODE(rc);
			if (flag) {
				found_stuff = true;
				src = stat.MPI_SOURCE;
				if (stat.MPI_TAG != 0) {
					count = stat.MPI_TAG;
					buffer = obuffer_type(count);
					rc = MPI_Recv(buffer.data(), count, MPI_BYTE, src, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					CHECK_MPI_RETURN_CODE(rc);
				} else {
					int total_count;
					rc = MPI_Recv(&total_count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					CHECK_MPI_RETURN_CODE(rc);
					buffer = obuffer_type(total_count);
					const int message_count = (total_count - 1) / MAX_MESSAGE_SIZE + 1;
					int remaining_count = total_count;
					for (int i = 0; i != message_count; ++i) {
						const int this_count = std::min(MAX_MESSAGE_SIZE, remaining_count);
						void* ptr = buffer.data() + i * MAX_MESSAGE_SIZE;
						rc = MPI_Recv(ptr, this_count, MPI_BYTE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						CHECK_MPI_RETURN_CODE(rc);
						remaining_count -= MAX_MESSAGE_SIZE;
					}
				}
				auto buffer_ptr = std::make_shared < obuffer_type > (std::move(buffer));
				hpx::thread([=]() {
					handle_incoming_message(src, *buffer_ptr);
				}).detach();
			}
		} while (flag && !exit_signal);
		std::unique_lock < hpx::lcos::local::mutex > lock(message_queue_mutex);
		auto i = message_queue.begin();
		while (i != message_queue.end() && !exit_signal) {
			found_stuff = true;
			lock.unlock();
			int rc = handle_outgoing_message(*i);
			if (rc == +1) {
				lock.lock();
				i = message_queue.erase(i);
			} else {
				auto tmp = *i;
				++i;
				if (rc == -1) {
					found_stuff = true;
				}
				lock.lock();
			}
		}
		if (!found_stuff) {
			hpx::this_thread::yield();
		}
	} while (found_stuff || !exit_signal);
	return EXIT_SUCCESS;
}

static void handle_incoming_message(int return_rank, obuffer_type message) {
	ibuffer_type buffer;
	invokation_type itype;
	message >> itype;
	if (itype != PROMISE) {
		naked_action_type func_ptr;
		if (itype == NAME) {
			std::string function_name;
			message >> function_name;
			func_ptr = lookup_action(function_name);
		} else if (itype == ADDRESS) {
			std::uintptr_t function_name;
			message >> function_name;
			func_ptr = reinterpret_cast<naked_action_type>(function_name);
		} else {
			assert(false);
			func_ptr = nullptr;
		}
		std::uintptr_t index;
		std::uintptr_t promise_function;
		message >> index;
		message >> promise_function;
		buffer << hpx::detail::invokation_type(PROMISE);
		buffer << promise_function;
		buffer << index;
		(*func_ptr)(buffer, std::move(message));
		hpx::detail::remote_action(return_rank, std::move(buffer));
	} else {
		std::uintptr_t ptrint;
		message >> ptrint;
		auto func_ptr = reinterpret_cast<promise_action_type>(ptrint);
		(*func_ptr)(std::move(message));
	}
}

void pack_args(ibuffer_type& buffer) {
}

static int handle_outgoing_message(std::shared_ptr<message_type> message) {
	int rc, mpi_rc;
	int flag;
	if (!message->sent) {
		int count = message->buffer.size();
		int dest = message->target_rank;
		if (count < MAX_MESSAGE_SIZE) {
			void* buffer = message->buffer.data();
			message->mpi_requests.resize(1);
			MPI_Request* request = message->mpi_requests.data();
			mpi_rc = MPI_Isend(buffer, count, MPI_BYTE, dest, count, MPI_COMM_WORLD, request);
			CHECK_MPI_RETURN_CODE(mpi_rc);
		} else {
			const int message_count = (count - 1) / MAX_MESSAGE_SIZE + 1;
			message->mpi_requests.resize(message_count + 1);
			message->message_size = count;
			MPI_Request* request = &(message->mpi_requests[0]);
			mpi_rc = MPI_Isend(&(message->message_size), 1, MPI_INT, dest, 0, MPI_COMM_WORLD, request);
			CHECK_MPI_RETURN_CODE(mpi_rc);
			int remaining_count = count;
			for (int i = 0; i < message_count; ++i) {
				request = &(message->mpi_requests[i + 1]);
				void* buffer = message->buffer.data() + i * MAX_MESSAGE_SIZE;
				const int this_count = std::min(MAX_MESSAGE_SIZE, remaining_count);
				mpi_rc = MPI_Isend(buffer, this_count, MPI_BYTE, dest, 0, MPI_COMM_WORLD, request);
				CHECK_MPI_RETURN_CODE(mpi_rc);
				remaining_count -= MAX_MESSAGE_SIZE;
			}
		}
		message->sent = true;
		rc = -1;
	} else {
		const int count = message->mpi_requests.size();
		MPI_Request* requests = message->mpi_requests.data();
		mpi_rc = MPI_Testall(count, requests, &flag, MPI_STATUSES_IGNORE);
		CHECK_MPI_RETURN_CODE(mpi_rc);
		if (!flag) {
			rc = 0;
		} else {
			rc = +1;
		}
	}
	return rc;
}

static void check_mpi_return_code(int code, const char* file, int line) {
	if (code != MPI_SUCCESS) {
		printf("Internal MPI error - ");
		switch (code) {
		case MPI_ERR_COMM:
			printf("MPI_ERR_COMM");
			break;
		case MPI_ERR_RANK:
			printf("MPI_ERR_RANK");
			break;
		case MPI_ERR_TAG:
			printf("MPI_ERR_TAG");
			break;
		case MPI_ERR_PENDING:
			printf("MPI_ERR_PENDING");
			break;
		case MPI_ERR_IN_STATUS:
			printf("MPI_ERR_IN_STATUS");
			break;
		default:
			printf("unknown code %i ", code);
		}
		printf(" - from call in %s on line %i\n", file, line);
		exit (EXIT_FAILURE);
	}
}

}

namespace naming {

int get_locality_id_from_gid(const gid_type& gid) {
	return gid.get_rank();
}

}

namespace serialization {

const std::size_t archive::initial_size = 1024;

oarchive::oarchive(const iarchive& other) :
		archive(other) {
}

oarchive::oarchive(iarchive&& other ) : archive( std::move(other)) {

}

oarchive& oarchive::operator=(const iarchive& other) {
	archive::operator=(*this);
	return *this;
}

oarchive& oarchive::operator=(iarchive&& other ) {
	archive::operator=(std::move(other));
	return *this;
}

iarchive::iarchive(const oarchive& other) :
		archive(other) {
}

iarchive::iarchive(oarchive&& other ) : archive( std::move(other)) {

}

iarchive& iarchive::operator=(const oarchive& other) {
	archive::operator=(other);
	return *this;
}

iarchive& iarchive::operator=(oarchive&& other ) {
	archive::operator=(std::move(other));
	return *this;
}

archive::archive(std::size_t size) :
		buffer_type(size), start(0) {

}

iarchive::iarchive(std::size_t size) :
		archive(size) {

}

oarchive::oarchive(std::size_t size) :
		archive(size) {

}

std::size_t archive::current_position() const {
	return start;
}

void archive::seek_to(std::size_t i) {
	start = i;
}

}

}

namespace hpx {

bool id_type::operator==(const id_type& other) const {
	return !(id_type::operator!=(other));
}

bool id_type::operator!=(const id_type& other) const {
	bool rc;
	if (rank != other.rank) {
		rc = true;
	} else if (address == nullptr) {
		rc = other.address != nullptr;
	} else if (other.address == nullptr) {
		rc = address != nullptr;
	} else if (*address != *(other.address)) {
		rc = true;
	} else {
		rc = false;
	}
	return rc;

}

bool future<void>::valid() const {
	return state != nullptr;
}

hpx::future<void> make_ready_future() {
	hpx::promise<void> promise;
	promise.set_value();
	return promise.get_future();
}

id_type find_here() {
	id_type id;
	int rank;
	rank = hpx::detail::mpi_comm_rank();
	id.set_rank(rank);
	return id;

}

void mutex::lock() {
	++waiting;
	while (locked++ != 0) {
		hpx::this_thread::yield();
	}
	--waiting;
}

void mutex::unlock() {
	locked = 0;
	if (waiting) {
		hpx::this_thread::yield();
	}
}

namespace detail {
void shared_state<void>::set_value() {
	ready = true;
}

void shared_state<void>::wait() const {
	while (!ready) {
		hpx::this_thread::yield();
	}
}

void shared_state<void>::get() {
	wait();
}
}
}

HPX_REGISTER_PLAIN_ACTION(action_add_ref_count);

