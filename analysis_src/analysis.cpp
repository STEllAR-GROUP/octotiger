#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <valarray>
#include <algorithm>
#include <utility>

#define MAX_BUFFER (64*1024)

template<class ...Args>
void double_print(FILE* fp, const char* str, Args ...args) {
	if (fp) {
		fprintf(fp, str, args...);
	}
	printf(str, args...);
}

inline double sinc(double x) {
	if (x != 0.0) {
		return sin(x) / x;
	} else {
		return 1.0;
	}
}

int main(int argc, char* argv[]) {
	if (argc != 3) {
		printf("average <file> <period>\n");
		abort();
	}

	double period = atof(argv[2]);
	if (period <= 0.0) {
		printf("Period must be positive\n");
		abort();
	}
	static char buffer[MAX_BUFFER];
	double entry;

	std::vector<std::valarray<double>> data;
	std::vector<std::valarray<double>> avg;
	std::vector<std::valarray<double>> dif;

	FILE* fp = fopen(argv[1], "rt");

	if (fp == NULL) {
		printf("Unable to open %s\n", argv[1]);
		abort();
	}
	int ncol = 0;
	while (!feof(fp)) {
		fgets(buffer, MAX_BUFFER, fp);
		char* ptr = buffer;
		while (*ptr == ' ' || *ptr == '\t') {
			ptr++;
		}
		std::vector<double> row;
		while (*ptr != '\n') {
			entry = atof(ptr);
			row.push_back(entry);
			while (!(*ptr == ' ' || *ptr == '\t')) {
				ptr++;
			}
			while (*ptr == ' ' || *ptr == '\t') {
				ptr++;
			}
		}
		ncol = std::max(ncol, int(row.size()));
		const int sz = data.size();
		const int szm1 = sz - 1;
		if ((sz == 0) || (row[0] > data[szm1][0])) {
			data.push_back(std::valarray<double>(row.data(), row.size()));
		}
	}
	fclose(fp);
	period *= 2.0 * M_PI / data[0][2];
	int head = 2, tail = 1;
	for (int i = 0; i < int(data.size()); i++) {
		if (data[i][0] - data[0][0] <= period) {
			head++;
		} else {
			break;
		}
	}
	int j = 0;
	for (head = head; head < int(data.size()) - 1; head++) {
		while (data[head][0] - data[tail][0] > period) {
			tail++;
		}
		printf("%i %i %e %e\n", head, tail, data[head][0] - data[tail][0], period);
		std::valarray<double> this_avg(0.0, ncol);
		std::valarray<double> this_dif(0.0, ncol);
		double weight_sum = 0.0;
		double t0 = (data[head][0] + data[tail][0]) * 0.5;
		for (int i = tail; i <= head; i++) {
			const double dt = 0.5 * (data[i + 1][0] - data[i - 1][0]);
			const double t = data[i][0];
			const double kernel = sinc( M_PI * (t - t0) / period);
			const double weight = kernel * dt;
			weight_sum += weight;
			this_avg += data[i] * weight;
			this_dif += (data[i + 1] - data[i - 1]) * 0.5 / dt * weight;
		}
		this_avg /= weight_sum;
		this_dif /= weight_sum;
		this_dif[0] = this_avg[0] = t0;
		j++;
		avg.push_back(std::move(this_avg));
		dif.push_back(std::move(this_dif));
	}

	using array_type = decltype(data);
	const auto print_array = [&](const array_type& a, const char* filename ) {
		fp = fopen(filename, "wt");
		if( !fp ) {
			printf( "Unable to write to %s\n", filename);
		}
		for (int j = 0; j < int(a.size()); j++) {
			const auto& this_a = a[j];
			for (int i = 0; i < int(this_a.size()); i++) {
				double_print(fp, "%e ", this_a[i]);
			}
			double_print(fp, "\n");
		}
		fclose(fp);
	};
	print_array(avg, "avg.dat");
	print_array(dif, "dif.dat");
	return 0;

}
