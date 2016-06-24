// nutil.hpp - 2016 - Atlee Brink
// general utility module

#pragma once

#include <cmath>
#include <istream>
#include <string>

namespace nutil {
    using namespace std;

    string
    getline(
        istream &is
    ) {
        
        string line;
        std::getline( is, line );
        switch( line.back() ) {
            case '\n':
            case '\r':
                line.erase( line.size() - 1, 1 );
        }

        return line;
    }
}
