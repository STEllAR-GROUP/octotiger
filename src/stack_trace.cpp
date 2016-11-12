/*
 * stack_trace.cpp
 *

 *  Created on: Sep 23, 2015
 *      Author: dmarce1
 */

/*
 * stack_trace.cpp
 *
 *  Created on: Jul 28, 2015
 *      Author: dmarce1
 */

/* MODIFIED FROM THIS ORIGINAL SOURCE : http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes */

#if !defined(_MSC_VER)
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
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
	UNABLE: printf("UNABLE TO PRINT STACK FROM GDB!\n");
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
