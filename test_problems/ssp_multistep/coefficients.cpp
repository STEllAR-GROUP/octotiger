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

bool close(double a, double b, double tolerance = 2e-12) {
    return std::abs(a - b) <= tolerance * std::max(1.0, std::max(std::abs(a), std::abs(b)));
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
        const double actual[] = {c.v, c.v5, c.c1, c.c2, c.c3, c.c5};
        for (int i = 0; i < 6; ++i) {
            require(close(actual[i], row[i + 2], 6e-6), "paper table agreement");
        }
        order_conditions(row[0], c);
    }
}

// The exact solution y(t)=1/(1-t) exercises the nonlinear RHS y'=y^2.
double nonlinear_error(int steps, bool variable_steps) {
    const double final_time = 0.25;
    const double base_dt = final_time / steps;
    double y = 1;
    double old_y = 0;
    double old_rhs = 0;
    double previous_dt = 0;
    int evaluations = 0;
    const auto rhs = [&evaluations](double value) {
        ++evaluations;
        return value * value;
    };
    for (int n = 0; n < steps; ++n) {
        // Even step counts integrate to the same endpoint; ratios alternate
        // between 9/11 and 11/9 within the supported interval.
        const double h = base_dt * (variable_steps ? (n % 2 == 0 ? 0.9 : 1.1) : 1);
        const double f = rhs(y);
        double next;
        if (n == 0) {
            const double stage1 = y + h * f;
            const double stage2 = 0.75 * y + 0.25 * (stage1 + h * rhs(stage1));
            next = y / 3 + (2.0 / 3) * (stage2 + h * rhs(stage2));
        } else {
            ssp_multistep::coefficients c;
            require(ssp_multistep::make_coefficients(previous_dt / h, c), "convergence coefficients");
            const double stage = c.c1 * (y + c.v * h * f) + c.c2 * old_y;
            next = c.c3 * y + c.c4 * (stage + c.v * h * rhs(stage)) +
                c.c5 * (old_y + c.v5 * h * old_rhs);
        }
        old_y = y;
        old_rhs = f;
        previous_dt = h;
        y = next;
    }
    require(evaluations == 2 * steps + 1, "two RHS evaluations after startup");
    return std::abs(y - 1 / (1 - final_time));
}

void step_selection() {
    for (int n = 1; n <= 200; ++n) {
        const double requested = 0.01 * n;
        const double current_fe = 0.02 + 0.007 * n;
        const double previous_fe = 0.03 + 0.005 * n;
        const double dt = ssp_multistep::select_step(1, requested, current_fe, previous_fe);
        if (dt == 0) {
            ssp_multistep::coefficients c;
            require(ssp_multistep::make_coefficients(4, c), "lower endpoint coefficients");
            require(requested < 0.25 || 0.25 * c.v > current_fe || 0.25 * c.v5 > previous_fe,
                "fallback has no supported admissible timestep");
        } else {
            ssp_multistep::coefficients c;
            require(ssp_multistep::make_coefficients(1 / dt, c), "selected coefficients");
            require(dt <= requested && dt * c.v <= current_fe && dt * c.v5 <= previous_fe,
                "separate current and historical FE bounds");
            const double larger = dt * (1 + 1e-10);
            const bool supported = ssp_multistep::make_coefficients(1 / larger, c);
            require(larger > requested || !supported || larger * c.v > current_fe ||
                larger * c.v5 > previous_fe, "maximal supported timestep");
        }
    }
    require(ssp_multistep::select_step(0, 1, 1, 1) == 0, "missing history fallback");
    require(ssp_multistep::select_step(1, 0.01, 1, 1) == 0, "large decrease fallback");
    require(ssp_multistep::select_step(1, 1, 1, 0) == 0, "invalid historical FE bound");
    require(ssp_multistep::select_step(1, 1, std::numeric_limits<double>::quiet_NaN(), 1) == 0,
        "invalid current FE bound");
    for (int exponent = -300; exponent <= 300; exponent += 10) {
        for (int mantissa = 1; mantissa <= 9; ++mantissa) {
            const double previous_dt = mantissa * std::pow(10.0, exponent);
            const double dt = ssp_multistep::select_step(previous_dt, 2 * previous_dt,
                100 * previous_dt, 100 * previous_dt);
            ssp_multistep::coefficients c;
            require(dt > 0 && ssp_multistep::make_coefficients(previous_dt / dt, c),
                "reconstructible upper endpoint at extreme scales");
            require(dt <= previous_dt / ssp_multistep::minimum_alpha,
                "growth cap at extreme scales");
            const double lower = ssp_multistep::select_step(previous_dt, previous_dt / 4,
                100 * previous_dt, 100 * previous_dt);
            require(lower > 0 && ssp_multistep::make_coefficients(previous_dt / lower, c),
                "reconstructible lower endpoint at extreme scales");
        }
    }
}

} // namespace

int main() {
    paper_table();
    ssp_multistep::coefficients c;
    for (int i = 0; i <= 3300; ++i) {
        const double alpha = 0.7 + 0.001 * i;
        require(ssp_multistep::make_coefficients(alpha, c), "supported coefficient interval");
        require(c.c1 >= 0 && c.c2 >= 0 && c.c3 >= 0 && c.c4 >= 0 && c.c5 >= 0 &&
            c.v > 0 && c.v5 >= 0, "SSP nonnegative coefficients");
        // For F=0, consecutive differences satisfy delta[n+1]=-B[n]*delta[n]
        // with B=c4*c2+c5. A uniform B<1 controls arbitrary supported ratio
        // sequences, not merely the constant-step characteristic roots.
        // B decreases with alpha; its maximum at alpha=0.7 is 0.857100237.
        const double parasitic_factor = c.c4 * c.c2 + c.c5;
        require(parasitic_factor >= 0 && parasitic_factor < 0.858,
            "uniform variable-step zero-stability contraction");
        require(close(c.c3 + c.c4 * c.c1, 1 - parasitic_factor),
            "constant solution root for the two-step recurrence");
        order_conditions(alpha, c);
    }
    require(!ssp_multistep::make_coefficients(0.5, c), "singular alpha fallback");
    require(!ssp_multistep::make_coefficients(0.69, c), "small alpha fallback");
    require(!ssp_multistep::make_coefficients(4.01, c), "large alpha fallback");
    require(!ssp_multistep::make_coefficients(std::numeric_limits<double>::infinity(), c),
        "infinite alpha fallback");
    require(!ssp_multistep::make_coefficients(std::numeric_limits<double>::quiet_NaN(), c),
        "NaN alpha fallback");
    for (const bool variable : {false, true}) {
        const double e1 = nonlinear_error(40, variable);
        const double e2 = nonlinear_error(80, variable);
        const double e3 = nonlinear_error(160, variable);
        require(e1 / e2 > 7.5 && e1 / e2 < 8.5 && e2 / e3 > 7.5 && e2 / e3 < 8.5,
            "third-order nonlinear convergence");
        std::printf("%s nonlinear refinement ratios: %.6f %.6f\n",
            variable ? "variable-step" : "constant-step", e1 / e2, e2 / e3);
    }
    step_selection();
    std::puts("SSP multistep coefficients, nonlinear order, and timestep bounds passed");
}
