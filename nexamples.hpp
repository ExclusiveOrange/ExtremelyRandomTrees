// nexamples.hpp - 2016 - Atlee Brink
// Machine-Learning Example-Dataset module

#pragma once

#include "nrandom.hpp"
#include "nutil.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
//#include <random>
#include <set>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace nexamples {
    using namespace std;
    using namespace nrandom;

    typedef float feature_t; // pseudo-continuous or discrete features

    typedef int label_t; // discrete labels

    typedef vector< feature_t > featurevector_t;

    class cexampleset {
        public:
            vector< string > names; // should match size of featurevector_t
            vector< featurevector_t > featurevectors;
            string labelname; // may be empty
            vector< label_t > labels; // may be empty, else should be same size as examples
            set< label_t > labelset; // contains all unique labels
            vector< string > exnames; // excluded feature names in order of column
            vector< vector< string > > exfeaturevectors; // excluded feature values

            bool statsed = false, normalized = false;
            vector< double > featuremeans; // average of each feature
            vector< double > featurestddevs; // standard deviation of each feature

            void
            computefeaturestats() {
                // compute featuremeans and featurestddevs

                if( statsed ) return;

                featuremeans = vector< double >( names.size(), 0.0 );
                
                // sums
                for( const auto &X : featurevectors ) {

                    auto meansi = featuremeans.begin();
                    for( const auto &x : X ) *meansi++ += x;
                
                }

                // means
                double rnum = 1.0 / featurevectors.size();
                for( auto &mean : featuremeans ) mean *= rnum;

                featurestddevs = vector< double >( names.size(), 0.0 );

                // sum of squared deviations
                for( const auto &X : featurevectors ) {

                    auto meansi = featuremeans.cbegin();
                    auto stddevsi = featurestddevs.begin();
                    for( const auto &x : X ) {
                        auto dev = *meansi++ - x;
                        *stddevsi++ += dev * dev;
                    }

                }

                // standard deviations (square root of mean of squared deviations)
                for( auto &stddev : featurestddevs ) stddev = sqrt( stddev * rnum );

                // set flag that stats have been computed
                statsed = true;

            }

            bool
            loadfromfile(
                const string &filename,
                string &labelcolumnname, // if empty and islabeled, then will be filled on return
                bool &islabeled, // in: must find label column; out: whether it was found
                const set< string > &excludefeatures
            ) {

                // reset flags
                statsed = false;
                normalized = false;

                // check parameters a bit
                if( excludefeatures.find( labelcolumnname )
                        != excludefeatures.end() ) {
                    cerr << "was asked to exclude label column, but I can't do that" << endl;
                    return false;
                }

                // open file
                ifstream infile( filename );
                if( !infile ) {
                    cerr << "error opening file: " << filename << endl;
                    return false;
                }

                // prepare column indices
                bool labelfound = false;
                size_t numfilecolumns = 0;
                enum coltype : char { EXCLUDE, FEATURE, LABEL };
                vector< coltype > columnmap;

                { // read header line
                    stringstream linestream( nutil::getline( infile ) );
                    size_t index = 0;

                    // read column names: learn indices of label and excluded columns
                    for( string token; getline( linestream, token, ',' ); index++ ) {

                        if( token == labelcolumnname ) {
                            labelfound = true;
                            columnmap.push_back( LABEL );
                        }
                        else {
                            if( excludefeatures.find( token ) != excludefeatures.end() ) {
                                exnames.push_back( move(token) );
                                columnmap.push_back( EXCLUDE );
                            }
                            else {
                                names.push_back( move(token) );
                                columnmap.push_back( FEATURE );
                            }
                        }
                    }

                    numfilecolumns = index;

                    // decide whether there should be a label column present,
                    //   and which one it is (if needed):
                    if( labelcolumnname.empty() ) {
                        if( islabeled ) {
                            // since no name was specified, assume last column is the label
                            columnmap[ numfilecolumns - 1 ] = LABEL;
                            labelcolumnname = names.back();
                            //cout << "assuming label column is: " << labelcolumnname << "\n";
                            names.pop_back();
                        }
                    }
                    else {
                        // else a label name was specified
                        if( !labelfound ) { // but it wasn't found
                            if( islabeled ) { // and this is a problem
                                cerr << "couldn't find label column: \""
                                    << labelcolumnname << "\", check data!" << endl;
                                return false;
                            } // else we don't care that it wasn't found:
                              //   labels and labelset will remain empty
                        }
                        else { // else it was found, so set islabeled = true
                            islabeled = true;
                        }
                    }

                    if( exnames.size() != excludefeatures.size() ) {
                        cerr << "couldn't find all excluded columns, check data!" << endl;
                        return false;
                    }

                } // done reading header
                labelname = labelcolumnname;

                // read values
                size_t linenum = 2;
                for( string line; getline( infile, line ); linenum++ ) {

                    featurevector_t featurevector;
                    vector< string > exfeaturevector;
                    stringstream linestream( line );

                    size_t index = 0;
                    for( string token; getline( linestream, token, ',' ); index++ ) {

                        if( index >= numfilecolumns ) {
                            cerr << filename << ": wrong numer of columns on line: "
                                << linenum << "\n"
                                << "expected " << numfilecolumns
                                << ", found " << index << " instead" << endl;
                            return false;
                        }

                        switch( columnmap[ index ] ) {
                            case LABEL: {
                                label_t label;
                                try {
                                    label = stoi( token );
                                } catch( invalid_argument &ex ) {
                                    cerr << filename
                                        << ": line " << linenum
                                        << ": column " << index
                                        << ": trying to read int, but: "
                                        << ex.what() << endl;
                                    return false;
                                }
                                labels.push_back( label );
                                labelset.insert( label );
                                break;
                            }
                            case FEATURE: {
                                feature_t featurevalue;
                                try {
                                    featurevalue = stof( token );
                                } catch( invalid_argument &ex ) {
                                    cerr << filename
                                        << ": line " << linenum
                                        << ": column " << index
                                        << ": trying to read float, but: "
                                        << ex.what() << endl;
                                    return false;
                                }
                                featurevector.push_back( featurevalue );
                                break;
                            }
                            default: { // EXCLUDE
                                exfeaturevector.push_back( token );
                                break;
                            }
                        }
                    }

                    if( index != numfilecolumns ) {
                        cerr << filename
                            << ": wrong number of columns on line: " << linenum << "\n"
                            << "expected " << numfilecolumns << ", found " << index << endl;
                        return false;
                    }

                    featurevectors.push_back( move( featurevector ) );
                    if( !exfeaturevector.empty() ) {
                        exfeaturevectors.push_back( move( exfeaturevector ) );
                    }

                } // for

                // check for some io error that means this file didn't actually read properly
                if( infile.bad() ) {
                    cerr << "IO error reading file: " << filename << endl;
                    return false;
                }

                // success!
                return true;

            } // loadfromfile

            void
            normalizefeatures() {
                // normalize each feature separately using its mean and standard deviation:
                // feature <- (feature - mean) / stddev

                if( normalized ) return;
                if( !statsed ) computefeaturestats();

                // precompute reciprocals for fastness
                vector< double > rstddevs( names.size(), 0.0 );
                auto rstddevsi = rstddevs.begin();
                for( const auto &stddev : featurestddevs ) {
                    *rstddevsi++ = stddev == 0.0 ? 1.0 : 1.0 / stddev;
                }

                // process all feature vectors
                for( auto &X : featurevectors ) {

                    auto meansi = featuremeans.cbegin();
                    auto rstddevsi = rstddevs.cbegin();
                    for( auto &x : X ) x = (x - *meansi++) * *rstddevsi++;

                }

                // set flag
                statsed = false; // old stats won't apply anymore
                normalized = true;

            }

            void
            normalizefeatures(
                const vector< double > &means,
                const vector< double > &stddevs
            ) {

                featuremeans = means;
                featurestddevs = stddevs;
                statsed = true;
                normalizefeatures();

            }

            vector< cexampleset >
            split(
                double proportionfortraining
            ) {

                if( proportionfortraining < 0.0 ) proportionfortraining = 0.0;
                else if( proportionfortraining > 1.0 ) proportionfortraining = 1.0;

                size_t numexamples = labels.size();

                // build ordered list of indices
                vector< size_t > indices( numexamples );
                for( size_t i = 0; i < numexamples; i++ ) indices[i] = i;

                // shuffle the list of indices
                for( size_t i = 0; i + 1 < numexamples; i++ ) {
                    size_t target = i + urand_sizet( twister ) % (numexamples - i);
                    swap( indices[ i ], indices[ target ] );
                }

                // calculate divider
                size_t divider = (size_t)( proportionfortraining * numexamples );

                // initialize subsets
                vector< cexampleset > subsets( 2 );
                for( cexampleset &subset : subsets ) {
                    subset.names = names;
                    subset.labelset = labelset;
                }
                subsets[0].featurevectors.reserve( divider );
                subsets[0].labels.reserve( divider );
                subsets[1].featurevectors.reserve( numexamples - divider );
                subsets[1].labels.reserve( numexamples - divider );

                // fill subsets
                for( size_t i = 0; i < divider; i++ ) {
                    subsets[0].featurevectors.push_back( featurevectors[ indices[ i ] ] );
                    subsets[0].labels.push_back( labels[ indices[ i ] ] );
                }
                for( size_t i = divider; i < numexamples; i++ ) {
                    subsets[1].featurevectors.push_back( featurevectors[ indices[ i ] ] );
                    subsets[1].labels.push_back( labels[ indices[ i ] ] );
                }

                return subsets;

            }

    }; // class cexampleset

} // namespace nexamples
