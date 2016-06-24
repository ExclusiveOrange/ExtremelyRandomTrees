// nrandom.hpp - 2016 - Atlee Brink
// randomization utility module

#pragma once

#include <limits>
#include <random>

namespace nrandom {
    using namespace std;

    random_device r;
    default_random_engine e1( r() );
    seed_seq seeds{ e1(), e1(), e1(), e1(), e1(), e1(), e1(), e1() };
    mt19937_64 twister( seeds );
    uniform_int_distribution< size_t > urand_sizet( 0, numeric_limits<size_t>::max() );
}
