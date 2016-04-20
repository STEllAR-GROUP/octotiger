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

/* SOURCE : http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes */

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

void handler(int sig) {
	char hostname[256];
	printf("SIGNAL %i\n", sig);
	if (sig != SIGINT && sig != SIGABRT) {
		gethostname(hostname, sizeof(hostname));
		static char command[1024];
		auto pid = getpid();
		sprintf(command,
				"ssh %s 'gdb --batch --quiet -ex \"thread apply all bt\" -ex \"quit\" -p %i' > gdb.%s.%i.txt\n",
				hostname, pid, hostname, pid);
		printf(command);
		if (system(command) != 0) {
			goto UNABLE;
		}
		goto ABLE;
		UNABLE: printf("UNABLE TO PRINT STACK FROM GDB!\n");
	}
	ABLE: exit(sig);

}

__attribute__((constructor))
void install_stack_trace() {
	signal(SIGSEGV, handler);   // install our handler
	signal(SIGABRT, handler);   // install our handler
	signal(SIGFPE, handler);   // install our handler
	signal(SIGILL, handler);   // install our handler
}
