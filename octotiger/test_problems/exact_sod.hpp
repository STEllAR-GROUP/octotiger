#ifndef EXACT_SOD__C
#define EXACT_SOD__C


typedef struct {
	double rhol, rhor, pl, pr, gamma;
} sod_init_t;

typedef struct {
	double rho, v, p;
} sod_state_t;

void exact_sod(sod_state_t* out, const sod_init_t* in, double x, double t);


constexpr sod_init_t sod_init = { 1.0, 0.125, 1.0, 0.1, 7.0 / 5.0 };


#endif
