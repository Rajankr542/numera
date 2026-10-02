#include <cmath>

#include "p14_window.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("p14: window functions follow NumPy's formulas") {
  NDArray h = p14::window(p14::Window::Hanning, 5);
  CHECK(h.shape() == (Shape{5}));
  CHECK_EQ(h.get_double(2), 1.0);
  CHECK_EQ(h.get_double(1), 0.5);
  CHECK(p14::window(p14::Window::Hamming, 0).shape() == (Shape{0}));
  CHECK(p14::window(p14::Window::Hamming, -3).shape() == (Shape{0}));
  CHECK_EQ(p14::window(p14::Window::Blackman, 1).get_double(0), 1.0);
  NDArray b = p14::window(p14::Window::Bartlett, 5.5);
  CHECK(b.shape() == (Shape{5}));
  CHECK_EQ(b.get_double(3), 1 - 1.5 / 4.5);
  CHECK_EQ(p14::window(p14::Window::Hamming, 4).get_double(0), 0.08000000000000002);
}

TEST_CASE("p14: kaiser and i0") {
  CHECK_EQ(p14::bessel_i0(0), 1.0);
  CHECK(std::fabs(p14::bessel_i0(10) - 2815.716628466254) < 1e-9);
  NDArray k = p14::kaiser(4, 5);
  CHECK_EQ(k.get_double(0), 0.036710892271286676);
  CHECK_EQ(p14::kaiser(1, 5).get_double(0), 1.0);
  CHECK(p14::kaiser(0, 5).shape() == (Shape{0}));
  CHECK(std::isnan(p14::kaiser(2.5, 1).get_double(2)));
}
