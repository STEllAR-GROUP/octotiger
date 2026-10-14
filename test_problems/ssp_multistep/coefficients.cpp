#include <octotiger/ssp_multistep.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "SSP multistep test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

bool close(long double a, long double b, long double tolerance = 2e-12L) {
    return std::abs(a - b) <= tolerance * std::max(1.0L, std::max(std::abs(a), std::abs(b)));
}

void order_conditions(double a, const ssp_multistep::coefficients& c) {
    const double old_weight = c.c4 * c.c2 + c.c5;
    const double stage_time = c.c1 * c.v - a * c.c2;
    const double old_derivative = c.c5 * c.v5;
    require(close(c.c1 + c.c2, 1), "stage consistency");
    require(close(c.c3 + c.c4 + c.c5, 1), "output consistency");
    require(close(-a * old_weight + c.c4 * c.v * (c.c1 + 1) + old_derivative, 1),
        "first-order tree");
    require(close(a * a * old_weight / 2 + c.c4 * c.v * stage_time - a * old_derivative, 0.5),
        "second-order tree");
    require(close(-a * a * a * old_weight / 6 +
        c.c4 * c.v * c.c2 * a * a / 2 + old_derivative * a * a / 2, 1.0 / 6),
        "third-order chain tree");
    require(close(-a * a * a * old_weight / 6 +
        c.c4 * c.v * stage_time * stage_time / 2 + old_derivative * a * a / 2, 1.0 / 6),
        "third-order branching tree");
}

void paper_table() {
    // The manuscript's table deliberately includes non-SSP negative c3 values.
    const double rows[][8] = {
        {0.5, 0.5, 2.65406, 0.319075, 0.291921, 0.708079, -2.47029, 2.97029},
        {0.75, 0.5, 1.85691, 0.493013, 0.492624, 0.507376, -0.261017, 0.761017},
        {0.5, 0.75, 1.87391, 0.31097, 0.376885, 0.623115, -3.46077, 3.71077},
        {0.75, 0.75, 1.34595, 0.453131, 0.587622, 0.412378, -0.998627, 1.24863},
        {1, 0.75, 1.13633, 0.595645, 0.717075, 0.282925, -0.371179, 0.621179}
    };
    for (const auto& row : rows) {
        ssp_multistep::coefficients c;
        require(ssp_multistep::from_c4(row[0], row[1], c), "table coefficients exist");
        const long double actual[] = {c.v, c.v5, c.c1, c.c2, c.c3, c.c5};
        for (int i = 0; i < 6; ++i) {
            require(close(actual[i], row[i + 2], 6e-6), "paper table agreement");
        }
        order_conditions(row[0], c);
    }
}

// The exact solution y(t)=1/(1-t) exercises the nonlinear RHS y'=y^2.
double nonlinear_error(int steps, int pattern) {
    const double final_time = 0.25;
    const double base_dt = final_time / steps;
    double y = 1;
    double time = 0;
    double old_y = 0;
    double old_rhs = 0;
    double previous_dt = 0;
    int evaluations = 0;
    const auto rhs = [&evaluations](double value) {
        ++evaluations;
        return value * value;
    };
    for (int n = 0; n < steps; ++n) {
        // The six-step sequence reaches the 1.9 growth cap and includes a
        // factor-19 decrease. Refinement repeats the same pattern more often.
        constexpr double wide_weights[] = {1, 1.9, 0.1, 0.19, 0.361, 0.6859};
        constexpr double wide_average = (1 + 1.9 + 0.1 + 0.19 + 0.361 + 0.6859) / 6;
        double h = base_dt * (pattern == 0 ? 1 : pattern == 1 ?
            (n % 2 == 0 ? 0.9 : 1.1) : wide_weights[n % 6] / wide_average);
        if (n != 0 && static_cast<long double>(h) >
            static_cast<long double>(previous_dt) * ssp_multistep::maximum_growth) {
            const long double cap = static_cast<long double>(previous_dt) *
                ssp_multistep::maximum_growth;
            h = static_cast<double>(cap);
            if (static_cast<long double>(h) > cap) {
                h = std::nextafter(h, 0.0);
            }
        }
        const double f = rhs(y);
        double next;
        if (n == 0) {
            const double stage1 = y + h * f;
            const double stage2 = 0.75 * y + 0.25 * (stage1 + h * rhs(stage1));
            next = y / 3 + (2.0 / 3) * (stage2 + h * rhs(stage2));
        } else {
            ssp_multistep::coefficients c;
            require(ssp_multistep::make_step_coefficients(previous_dt, h, c), "convergence coefficients");
            const double stage = c.c1 * (y + c.v * h * f) + c.c2 * old_y;
            next = c.c3 * y + c.c4 * (stage + c.v * h * rhs(stage)) +
                c.c5 * (old_y + c.v5 * h * old_rhs);
        }
        old_y = y;
        old_rhs = f;
        previous_dt = h;
        time += h;
        y = next;
    }
    require(evaluations == 2 * steps + 1, "two RHS evaluations after startup");
    require(close(time, final_time), "variable sequences share the final time");
    return std::abs(y - 1 / (1 - time));
}

bool admissible(double previous_dt, double dt, double current_fe, double previous_fe) {
    ssp_multistep::coefficients c;
    return ssp_multistep::make_step_coefficients(previous_dt, dt, c) &&
        static_cast<long double>(dt) * c.v <= current_fe &&
        static_cast<long double>(dt) * c.v5 <= previous_fe;
}

void check_selection(double previous_dt, double requested, double current_fe, double previous_fe) {
    const double dt = ssp_multistep::select_step(previous_dt, requested, current_fe, previous_fe);
    require(dt > 0 && dt <= requested && admissible(previous_dt, dt, current_fe, previous_fe),
        "separate current and historical FE bounds");
    const double larger = std::nextafter(dt, std::numeric_limits<double>::infinity());
    require(larger > requested || !admissible(previous_dt, larger, current_fe, previous_fe),
        "largest representable supported timestep");
}

void step_selection() {
    for (int n = 1; n <= 200; ++n) {
        const double requested = 0.01 * n;
        const double current_fe = 0.02 + 0.007 * n;
        const double previous_fe = 0.03 + 0.005 * n;
        if (static_cast<long double>(previous_fe) <= 0.4L) {
            require(ssp_multistep::select_step(1, requested, current_fe, previous_fe) == 0,
                "historical Euler lower limit requires fallback");
        } else {
            check_selection(1, requested, current_fe, previous_fe);
        }
    }
    require(ssp_multistep::select_step(0, 1, 1, 1) == 0, "missing history fallback");
    require(ssp_multistep::select_step(1, 0.01, 1, 1) == 0.01, "large decrease retains history");
    require(ssp_multistep::select_step(1, 1, 1, 0) == 0, "invalid historical FE bound");
    require(ssp_multistep::select_step(1, 1, 1, 0.39) == 0, "no feasible historical Euler step");
    require(ssp_multistep::select_step(1, 1, std::numeric_limits<double>::quiet_NaN(), 1) == 0,
        "invalid current FE bound");
    check_selection(1, 1, 100, 0.401);
    check_selection(1, 1, 1e-200, 1);
    check_selection(1e300, 1e-300, 1e-300, 1e300);
    const double smallest = std::numeric_limits<double>::denorm_min();
    const double largest = std::numeric_limits<double>::max();
    check_selection(largest, smallest, smallest, largest);
    check_selection(smallest, largest, largest, largest);
    check_selection(std::numeric_limits<double>::min(), smallest, smallest, 1);
    for (int exponent = -300; exponent <= 300; exponent += 10) {
        for (int mantissa = 1; mantissa <= 9; ++mantissa) {
            const double previous_dt = mantissa * std::pow(10.0, exponent);
            check_selection(previous_dt, 2 * previous_dt, 100 * previous_dt, 100 * previous_dt);
            check_selection(previous_dt, previous_dt / 1024,
                100 * previous_dt, 100 * previous_dt);
        }
    }
}

} // namespace

int main() {
    paper_table();
    ssp_multistep::coefficients c;
    for (int i = 0; i <= 4000; ++i) {
        const double alpha = ssp_multistep::minimum_alpha + 0.001 * i;
        require(ssp_multistep::make_coefficients(alpha, c), "supported coefficient interval");
        require(c.c1 >= 0 && c.c2 >= 0 && c.c3 >= 0 && c.c4 >= 0 && c.c5 >= 0 &&
            c.v > 0 && c.v5 >= 0, "SSP nonnegative coefficients");
        // For F=0, consecutive differences satisfy delta[n+1]=-B[n]*delta[n]
        // with B=c4*c2+c5. A uniform B<1 controls arbitrary supported ratio
        // sequences, not merely the constant-step characteristic roots.
        // B decreases with alpha; its maximum at alpha=1/1.9 is 0.995110796.
        const double parasitic_factor = c.c4 * c.c2 + c.c5;
        require(parasitic_factor >= 0 && parasitic_factor < 0.996,
            "uniform variable-step zero-stability contraction");
        require(close(c.c3 + c.c4 * c.c1, 1 - parasitic_factor),
            "constant solution root for the two-step recurrence");
        order_conditions(alpha, c);
        require(close(ssp_multistep::stage_abscissa(c), c.c1 * c.v - alpha * c.c2),
            "stable stage abscissa matches order conditions");
    }
    require(!ssp_multistep::make_coefficients(0.5, c), "singular alpha fallback");
    require(!ssp_multistep::make_coefficients(0.52, c), "growth cap fallback");
    require(ssp_multistep::make_coefficients(4.01, c), "old decrease cap removed");
    for (const double alpha : {8.0, 16.0, 32.0}) {
        require(ssp_multistep::make_coefficients(alpha, c), "large ratio coefficients");
        order_conditions(alpha, c);
    }
    for (const double alpha : {1e8, 1e100, 1e200, std::numeric_limits<double>::max()}) {
        require(ssp_multistep::make_coefficients(alpha, c), "extreme finite ratio coefficients");
        require(c.c1 >= 0 && c.c2 >= 0 && c.c3 >= 0 && c.c4 >= 0 && c.c5 >= 0 &&
            c.v > 0 && std::isfinite(c.v5), "extreme ratios retain nonnegative finite coefficients");
        require(close(c.v, 0.5, 1e-8) && close(c.v5 / alpha, 0.4, 1e-8),
            "large ratio Euler duration limits");
        require(close(ssp_multistep::stage_abscissa(c), 0.5, 1e-8),
            "stable large ratio stage abscissa");
    }
    require(!ssp_multistep::make_coefficients(std::numeric_limits<double>::infinity(), c),
        "infinite alpha fallback");
    require(!ssp_multistep::make_coefficients(std::numeric_limits<double>::quiet_NaN(), c),
        "NaN alpha fallback");
    for (const int pattern : {0, 1, 2}) {
        const double e1 = nonlinear_error(48, pattern);
        const double e2 = nonlinear_error(96, pattern);
        const double e3 = nonlinear_error(192, pattern);
        require(e1 / e2 > 7.5 && e1 / e2 < 8.5 && e2 / e3 > 7.5 && e2 / e3 < 8.5,
            "third-order nonlinear convergence");
        std::printf("%s nonlinear refinement ratios: %.6f %.6f\n",
            pattern == 0 ? "constant-step" : pattern == 1 ? "variable-step" : "wide-ratio",
            e1 / e2, e2 / e3);
    }
    step_selection();
    std::puts("SSP multistep coefficients, nonlinear order, and timestep bounds passed");
}
