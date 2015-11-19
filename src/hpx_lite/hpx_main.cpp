/*
 * hpx_main.cpp
 *
 *  Created on: Sep 18, 2015
 *      Author: dmarce1
 */

#include <unistd.h>


#include "hpx.hpp"

int test(int i, char c) {
	printf("Testing! %i %c\n", i, c);
	return 42;
}

struct test_class {
	int j;

	template<class Arc>
	void load(const Arc& arc, unsigned) {
		arc & j;
	}

	template<class Arc>
	void save(Arc& arc, unsigned) const {
		arc & j;
	}

	HPX_SERIALIZATION_SPLIT_MEMBER()
	;

	test_class(int i) {
		j = i;
		printf("Constructor called %i\n", i);
	}
	~test_class() {
		printf("Destructor called\n");
	}
	int test_method(char c) {
		printf("test method called\n");
		printf("%i %c\n", j, c);
		return j;
	}
	HPX_DEFINE_COMPONENT_ACTION(test_class, test_method, test_action);
};

typedef test_class::test_action test_action_type;
HPX_REGISTER_ACTION(test_action_type);

HPX_PLAIN_ACTION(test, action_test);

int hpx_main(int argc, char* argv[]) {
	test_class test_serial(1);
	hpx::detail::ibuffer_type ibuf;
//	hpx::detail::obuffer_type obuf;
	ibuf << test_serial;
	//obuf >> test_serial;
	hpx::id_type locality;
	locality.set_rank(0);
	hpx::future<hpx::id_type> fut = hpx::new_<test_class>(locality, 1);
	hpx::id_type id = fut.get();
	auto fut2 = hpx::async<typename test_class::test_action>(id, 'f');
	printf("%i\n", fut2.get());
	action_test act;
	char c = 'c';
	auto rc = act(locality, int(0), c);
	printf("%i\n", rc);
	rc = act(locality, int(4), 'd');
	printf("%i\n", rc);
	hpx::future<int> fut1;
	hpx::detail::ibuffer_type buf;
	buf << fut1;
	hpx::detail::obuffer_type bufo = buf;
	bufo >> fut1;

	std::array<int, 5> test_array;
	bufo >> test_array;
	buf << test_array;
	return hpx::finalize();
}


