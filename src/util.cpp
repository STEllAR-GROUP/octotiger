/*
 * util.cpp
 *
 *  Created on: Apr 20, 2016
 *      Author: dmarce1
 */

#include <stdio.h>


int file_copy(const char* fin, const char* fout) {
	constexpr size_t chunk_size = 1024;
	char buffer[chunk_size];
	FILE* fp_in = fopen(fin, "rb");
	FILE* fp_out = fopen(fout, "wb");
	if (fp_in == NULL) {
		return 1;
	}
	if (fp_out == NULL) {
		return 2;
	}
	size_t bytes_read;
	while ((bytes_read = fread(buffer, sizeof(char), chunk_size, fp_in)) != 0) {
		fwrite(buffer, sizeof(char), bytes_read, fp_out);
	}
	fclose(fp_in);
	fclose(fp_out);
}
