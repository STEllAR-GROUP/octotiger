#include <string>
#include <functional>


std::function<double(double)> build_rho_of_h_from_mesa(
		const std::string& filename);
