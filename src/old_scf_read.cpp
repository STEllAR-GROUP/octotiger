#define N_PHI 256
#define N_RAD 130
#define N_VER 64

#include <array>
#include <cmath>

#define DPHI ((2.0*M_PI)/N_PHI)
#define DX (1.0 / (N_RAD-3))

static bool file_exists = false;

typedef double real;
typedef std::array<std::array<std::array<real, N_RAD>, N_VER>, N_PHI> array_type;

static array_type rho;
static array_type pre;

static real interp_scf(array_type& a, real x, real y, real z) {
	if (!file_exists) {
		printf("density.bin no founde!\n");
	}
	const real dr = DX;
	const real dz = DX;
	const real dphi = 2.0 * M_PI / N_PHI;
	const real r = std::sqrt(x * x + y * y);
	real phi = atan2(y, x);
	if (phi < 0.0) {
		phi += 2.0 * M_PI;
	}
	const int vm = std::abs(z + 0.5 * dz) / dz;
	const int rm = (r + 0.5 * dr) / dr;
	const int pm = (phi) / dphi;
	const int vp = vm + 1;
	const int rp = rm + 1;
	if (vm >= N_VER - 1 || rm >= N_RAD - 1) {
		return 0.0;
	}
	int pp = pm + 1;
	if (pp == 256) {
		pp = 0;
	}
	const real vw = std::abs(z + 0.5 * dz) / dz - vm;
	const real rw = (r + 0.5 * dr) / dr - rm;
	const real pw = (phi) / dphi - pm;
	real n = 0.0;
	n += pw * vw * rw * a[pp][vp][rp];
	n += pw * vw * (1.0 - rw) * a[pp][vp][rm];
	n += pw * (1.0 - vw) * rw * a[pp][vm][rp];
	n += pw * (1.0 - vw) * (1.0 - rw) * a[pp][vm][rm];
	n += (1.0 - pw) * vw * rw * a[pm][vp][rp];
	n += (1.0 - pw) * vw * (1.0 - rw) * a[pm][vp][rm];
	n += (1.0 - pw) * (1.0 - vw) * rw * a[pm][vm][rp];
	n += (1.0 - pw) * (1.0 - vw) * (1.0 - rw) * a[pm][vm][rm];
	return n;
}

real interp_scf_rho(real x, real y, real z) {
	return interp_scf(rho, x, y, z);
}

real interp_scf_pre(real x, real y, real z) {
	return interp_scf(pre, x, y, z);
}

__attribute__((constructor))
static void read_file() {
	FILE* fp = fopen("density.bin", "rb");
	if (fp != NULL) {
		printf("density.bin found\n");
		file_exists = true;
	}
	fseek(fp, 8, SEEK_CUR);
	std::size_t bytes_read = 0;
	for (int pi = 0; pi != N_PHI; ++pi) {
		for (int vi = 0; vi != N_VER; ++vi) {
			for (int ri = 0; ri != N_RAD; ++ri) {
				real& r = rho[pi][vi][ri];
				bytes_read += fread(&r, sizeof(real), 1, fp) * sizeof(real);
				std::uint32_t* r1 = reinterpret_cast<std::uint32_t*>(&r);
				std::uint32_t* r2 = r1 + 1;
				*r1 = __builtin_bswap32(*r1);
				*r2 = __builtin_bswap32(*r2);
				real* r3 = reinterpret_cast<real*>(r1);
				r = *r3;
			}
		}
	}
	fclose(fp);

	fp = fopen("pressure.bin", "rb");
	if (fp != NULL) {
		printf("pressure.bin found\n");
		file_exists = true;
	} else {
		file_exists = false;
	}
	fseek(fp, 8, SEEK_CUR);
	for (int pi = 0; pi != N_PHI; ++pi) {
		for (int vi = 0; vi != N_VER; ++vi) {
			for (int ri = 0; ri != N_RAD; ++ri) {
				real& r = pre[pi][vi][ri];
				bytes_read += fread(&r, sizeof(real), 1, fp) * sizeof(real);
				std::uint32_t* r1 = reinterpret_cast<std::uint32_t*>(&r);
				std::uint32_t* r2 = r1 + 1;
				*r1 = __builtin_bswap32(*r1);
				*r2 = __builtin_bswap32(*r2);
				real* r3 = reinterpret_cast<real*>(r1);
				r = *r3;
			}
		}
	}
	fclose(fp);

}
