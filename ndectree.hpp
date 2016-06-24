// ndectree.hpp - 2016 - Atlee Brink
// Decision Tree module

#pragma once

#include "nutil.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ndectree {
    using namespace std;

    const string STR_LEAF = "\\";
    const string STR_BRANCH = "+";
    const string STR_DEPTH = "|";

    template< typename ATTR_T >
    struct dectree_t {
        bool isleaf; // else is branch
        union {
            vector< size_t > classfreqs; // if leaf
            struct { // if branch
                size_t attrindex;
                ATTR_T splitvalue;
                dectree_t *left, *right;
            };
        };

        dectree_t( // leaf constructor
            vector< size_t > &&classfreqs
        ) : isleaf( true ),
            classfreqs( classfreqs )
        {}

        dectree_t( // branch constructor
            size_t attrindex,
            ATTR_T splitvalue,
            dectree_t *left,
            dectree_t *right
        ) : isleaf( false ),
            attrindex( attrindex ),
            splitvalue( splitvalue ),
            left( left ),
            right( right )
        {}

        ~dectree_t() {
            if( isleaf ) classfreqs.~vector<size_t>();
            else {
                delete left;
                delete right;
            }
        }

        void
        storetostream(
            ostream &out
        ) const {
            if( isleaf ) {
                out << STR_LEAF << " ";
                for( size_t i = 0; i + 1 < classfreqs.size(); i++ ) {
                    out << classfreqs[i] << " ";
                }
                out << classfreqs.back() << "\n";
            }
            else {
                out << STR_BRANCH << " " << attrindex
                    << " " << splitvalue << "\n";
                left->storetostream( out );
                right->storetostream( out );
            }
        }

        static
        dectree_t*
        loadfromstream(
            istream &in,
            size_t numclasses
        ) {
            try {

                string str;

                // get type marker
                in >> str;

                if( str == STR_LEAF ) {

                    vector< size_t > classfreqs;
                    classfreqs.reserve( numclasses );

                    for( size_t c = 0; c < numclasses; c++ ) {
                        size_t freq;
                        in >> freq;
                        classfreqs.push_back( freq );
                    }

                    return new dectree_t( move( classfreqs ) );
                }
                else if( str == STR_BRANCH ) {

                    size_t attrindex;
                    in >> attrindex;

                    ATTR_T splitvalue;
                    in >> splitvalue;

                    dectree_t *left = loadfromstream( in, numclasses );
                    if( left == nullptr ) return nullptr;
                    dectree_t *right = loadfromstream( in, numclasses );
                    if( right == nullptr ) {
                        delete left;
                        return nullptr;
                    }

                    return new dectree_t( attrindex, splitvalue, left, right );
                }

            }
            catch( ... ) {
                cerr << "dectree_t::loadfromstream: error reading from stream" << endl;
            } 
            return nullptr;
        }
    };

    template< typename ATTR_T, typename LABEL_T >
    struct forest_t {
        vector< dectree_t< ATTR_T >* > trees;
        vector< LABEL_T > indextolabel;

        forest_t() {}

        forest_t( forest_t &&other ) {
            swap( trees, other.trees );
            swap( indextolabel, other.indextolabel );
        }

        forest_t(
            vector< dectree_t< ATTR_T >* > &&trees,
            vector< LABEL_T > &&indextolabel
        ) : trees( trees ),
            indextolabel( indextolabel )
        {}

        ~forest_t() {
            for( auto tree : trees ) delete tree;
        }

        forest_t&
        operator=( forest_t &&other ) {
            swap( trees, other.trees );
            swap( indextolabel, other.indextolabel );
            return *this;
        }

        LABEL_T
        classify(
            const vector< ATTR_T > featurevector
        ) const {

            vector< size_t > labelcounts( indextolabel.size(), 0 );

            for( dectree_t< ATTR_T > *pdectree : trees ) {

                auto pt = pdectree;
                while( !pt->isleaf ) {
                    ATTR_T a = featurevector[ pt->attrindex ];
                    pt = a < pt->splitvalue ? pt->left : pt->right;
                }

                size_t maxfreq = 0;
                size_t maxlabelindex = 0;
                size_t labelindex = 0;
                for( size_t freq : pt->classfreqs ) {
                    
                    if( freq > maxfreq ) {
                        maxfreq = freq;
                        maxlabelindex = labelindex;
                    }

                    labelindex++;
                }

                labelcounts[ maxlabelindex ]++;
            }

            size_t maxcount = 0;
            size_t maxcountindex = 0;
            size_t countindex = 0;
            for( size_t count : labelcounts ) {
                if( count > maxcount ) {
                    maxcount = count;
                    maxcountindex = countindex;
                }
                countindex++;
            }

            return indextolabel[ maxcountindex ];
        }
    };
}
