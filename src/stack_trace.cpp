//  Copyright (c) 2019 AUTHORS
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/* MODIFIED FROM THIS ORIGINAL SOURCE : http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes */

#include "octotiger/print.hpp"

#if !defined(_MSC_VER)
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void handler(int sig) {
	char hostname[256];
	gethostname(hostname, sizeof(hostname));
	static char command[1024];
	auto pid = getpid();
	sprintf(command, "echo \"SIGNAL %i\n\" > gdb.%s.%i.txt", sig, hostname, pid);
	sprintf(command, "ssh %s 'gdb --batch --quiet -ex \"thread apply all bt\" -ex \"quit\" -p %i' >> gdb.%s.%i.txt\n",
			hostname, pid, hostname, pid);
	if (system(command) != 0) {
		goto UNABLE;
	}
	goto ABLE;
	UNABLE: print("UNABLE TO PRINT STACK FROM GDB!\n");
	ABLE:
	exit(0);
}

__attribute__((constructor))
void install_stack_trace() {
	//signal(SIGABRT, handler);
	//signal(SIGINT, handler);
//	signal(SIGSEGV, handler);
//	signal(SIGFPE, handler);
//	signal(SIGILL, handler);
//	signal(SIGTERM, handler);
//	signal(SIGHUP, handler);
}
#endif
