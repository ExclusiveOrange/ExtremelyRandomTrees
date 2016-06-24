// nextratrees.hpp - 2016 - Atlee Brink
// Extremely Random Trees module
//
// based on the Extra-Trees algorithm as described in:
//   "Extremely randomized trees", DOI 10.1007/s10994-006-6226-1,
//   by Pierre Geurts, Damien Ernst, Louis Wehenkel, 2005
//

#pragma once

#include "ndectree.hpp"
#include "nexamples.hpp"
#include "nrandom.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace nextratrees {
    using namespace std;
    using namespace nrandom;

    typedef nexamples::feature_t attr_t; // probably float or double
    typedef nexamples::label_t label_t; // probably int

    typedef ndectree::dectree_t< attr_t > dectree_t;
    typedef ndectree::forest_t< attr_t, label_t > forest_t;

    struct exampleset_t {
        vector< string > attrnames; // actual attribute (feature) names
        vector< vector< attr_t > > attrs; // all attribute values from all examples
        vector< label_t > labelvalues; // actual labels (maybe -2, 5, 1000, whatever)
        vector< size_t > labels; // all label indexes from all examples
        size_t numclasses; // number of unique labels across all examples
    };

    double
    score(
        const vector< attr_t > &attr, // one attribute from all examples
        const vector< size_t > &labels, // label indices from all examples
        size_t numclasses, // number of unique labels
        const vector< size_t > &indices, // description of subset
        attr_t split // {a < split} => left, {a >= split} => right
    ) {

        // count stuff
        size_t numsplit[2] = {}; // {left, right}
        size_t numclass[ numclasses ][2] = {}; // initializes to 0
        for( auto &i : indices ) {
            bool isright = attr[i] >= split;
            label_t label = labels[i];

            numsplit[ isright ]++;
            numclass[ label ][ isright ]++;
        }

        // check that a split actually occurs
        if( !numsplit[0] || !numsplit[1] ) return 0.0;

        // precompute 1 / size_of_subset
        double rsize = 1.0 / indices.size();
        
        // mutual information and class entropy
        double mutualinf = 0.0;
        double classent = 0.0;
        for( auto &ci : numclass ) {

            if( ci[0] || ci[1] ) { // class is non-empty

                double p_ci = (ci[0] + ci[1]) * rsize;

                classent -= p_ci * log2( p_ci );

                if( ci[0] ) { // class and left is non-empty

                    double p_ci_and_left = ci[0] * rsize;
                    double p_ci_given_left = ci[0] / (double)numsplit[0];

                    mutualinf -= p_ci_and_left * log2( p_ci / p_ci_given_left );
                }

                if( ci[1] ) { // class and right is non-empty

                    double p_ci_and_right = ci[1] * rsize;
                    double p_ci_given_right = ci[1] / (double)numsplit[1];

                    mutualinf -= p_ci_and_right * log2( p_ci / p_ci_given_right );
                }
            }
        }

        // split entropy
        double splitent;
        {
            double p_left = numsplit[0] * rsize;
            double p_right = numsplit[1] * rsize;

            splitent = -p_left * log2( p_left ) - p_right * log2( p_right );
        }

        // score
        return 2.0 * mutualinf / ( splitent + classent );
    }

    dectree_t*
    build_an_extra_tree(
        const exampleset_t &exampleset,
        const vector< size_t > &attrindices, // description of attribute subset
        const vector< size_t > &indices, // description of example subset
        size_t nmin, // if indices.size() < nmin then just return a new leaf
        size_t numattr // also known as K: max number of attributes to consider per node
    ) {

        // count class frequencies
        vector< size_t > classfreqs( exampleset.numclasses, 0 );
        for( auto &i : indices ) {
            auto ci = exampleset.labels[ i ];
            classfreqs[ ci ]++;
        }
        
        // check if subset size is small enough to leaf
        if( indices.size() < nmin ) return new dectree_t( move( classfreqs ) );

        { // check if output variable is constant => then leaf
            size_t numclassesnonzero = 0;
            for( auto &c : classfreqs ) numclassesnonzero += c != 0;
            if( numclassesnonzero == 1 ) return new dectree_t( move( classfreqs ) );
        }
        
        // compute min and max of each attribute,
        // collect list of candidate attributes (those which are non-constant)
        attr_t mins[ exampleset.attrs.size() ];
        attr_t maxs[ exampleset.attrs.size() ];
        vector< size_t > attrcandidates;
        {
            for( auto &ai : attrindices ) {
                auto &attr = exampleset.attrs[ ai ]; 
                auto &min = mins[ ai ];
                auto &max = maxs[ ai ];
                min = max = attr[ indices[0] ];
                for( size_t i = 1; i < indices.size(); i++ ) {
                    auto a = attr[ indices[i] ];
                    if( a < min ) min = a;
                    else if( a > max ) max = a;
                }
                if( min < max ) attrcandidates.push_back( ai );
            }
        }
        
        // check if all candidate attributes are constant
        if( attrcandidates.size() < 1 ) return new dectree_t( move( classfreqs ) );

        // randomly pick (without replacement) numattr attributes to split on
        vector< size_t > attrs;
        if( attrcandidates.size() == 1 || attrcandidates.size() <= numattr ) {
            attrs = attrcandidates;
        }
        else {
            for( size_t k = 0; k < numattr; k++ ) {
                size_t target = urand_sizet( twister ) % attrcandidates.size();
                attrs.push_back( attrcandidates[ target ] );
                swap( attrcandidates[ target ], attrcandidates.back() );
                attrcandidates.pop_back();
            }
        }

        // randomly split among selected attributes,
        // choose that split with the highest score
        size_t bestattr = attrs[0];
        attr_t bestsplit = 0.0;
        double bestscore = -1.0;
        for( auto &ai : attrs ) {
            
            auto &min = mins[ ai ], &max = maxs[ ai ];
            uniform_real_distribution< attr_t > urand( min, max );
            attr_t split = urand( twister );
            double scr = score(
                exampleset.attrs[ ai ],
                exampleset.labels,
                exampleset.numclasses,
                indices,
                split
            );
            if( scr > bestscore ) {
                bestattr = ai;
                bestsplit = split;
                bestscore = scr;
            }
        }

        // split indices in two
        vector< size_t > leftindices, rightindices;
        {
            auto &attr = exampleset.attrs[ bestattr ];
            for( auto &i : indices ) {
                if( attr[ i ] < bestsplit ) leftindices.push_back( i );
                else rightindices.push_back( i );
            }
        }

        // build left and right subtrees
        dectree_t *left = build_an_extra_tree(
            exampleset,
            attrs,
            leftindices,
            nmin,
            numattr
        );
        dectree_t *right = build_an_extra_tree(
            exampleset,
            attrs,
            rightindices,
            nmin,
            numattr
        );


        return new dectree_t( bestattr, bestsplit, left, right ); 
    }

    forest_t
    build_an_extra_ensemble(
        const nexamples::cexampleset &exampleset,
        size_t numtrees, // also known as M
        size_t nmin, // controls minimum subset size during tree building
        size_t numattr // also known as K: max number of attributes to consider per node
    ) {

        // create exampleset_t from cexampleset:
        // fundamentally this means arranging data by attribute instead of by example,
        // and converting unique label values to label indices.
        exampleset_t exset = {
            exampleset.names,
            vector< vector< attr_t > >( exampleset.names.size(), vector< attr_t >() ),
            vector< label_t >(),
            vector< size_t >(),
            exampleset.labelset.size()
        };

        // copy all attribute values
        for( auto &featurevector : exampleset.featurevectors ) {
            for( size_t i = 0; i < featurevector.size(); i++ ) {
                exset.attrs[ i ].push_back( featurevector[ i ] );
            }
        }

        // copy label values and generate list of label indices and label->index map
        map< label_t, size_t > labeltoindex;
        for( auto &l : exampleset.labelset ) {
            size_t index = exset.labelvalues.size();
            exset.labelvalues.push_back( l );
            labeltoindex.insert( make_pair( l, index ) );
        }

        // create list of label indices
        for( auto &l : exampleset.labels ) {
            exset.labels.push_back( labeltoindex[ l ] );
        }

        // build ensemble
        vector< dectree_t* > ensemble;
        {
            // initial attribute indices (all of them)
            vector< size_t > attrindices;
            for( size_t i = 0; i < exset.attrnames.size(); i++ ) {
                attrindices.push_back( i );
            }

            // initial example indices (all of them)
            vector< size_t > indices;
            for( size_t i = 0; i < exset.labels.size(); i++ ) {
                indices.push_back( i );
            }

            // build decision trees
            for( size_t i = 0; i < numtrees; i++ ) {
                dectree_t *ptree = build_an_extra_tree(
                    exset,
                    attrindices,
                    indices,
                    nmin,
                    numattr
                );
                ensemble.push_back( ptree );
            }
        }

        vector< label_t > indextolabel;
        for( auto &l : exampleset.labelset ) indextolabel.push_back( l );

        return forest_t( move( ensemble ), move( indextolabel ) );
    }

    // TODO: deprecate: can use the classifier in ndectree instead
    label_t
    classify_from_tree(
        const dectree_t *pdectree,
        const vector< label_t > &indextolabel,
        const vector< attr_t > &featurevector
    ) {
        
        const dectree_t *pt = pdectree;
        while( !pt->isleaf ) {
            attr_t a = featurevector[ pt->attrindex ];
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

        return indextolabel[ maxlabelindex ];
    }

    bool
    loadmodelfromfile(
        const string &filename, // read-only
        string &dest_labelname,
        vector< string > &dest_exnames,
        vector< string > &dest_attrnames,
        forest_t &dest_forest,
        size_t &dest_nmin,
        size_t &dest_numattr,
        size_t &dest_optimizationlayers
    ) {

        ifstream infile( filename );
        if( !infile ) {
            cerr << "error opening input file: " << filename << endl;
            return false;
        }

        dest_forest = move( forest_t() );

        // labelname, ordered list of label/class values
        {
            stringstream linestream( nutil::getline( infile ) );
            linestream >> dest_labelname;
            for( label_t label; linestream >> label; ) {
                dest_forest.indextolabel.push_back( label );
            }
        }

        // excluded feature names
        dest_exnames.clear();
        {
            stringstream linestream( nutil::getline( infile ) );
            for( string name; linestream >> name; ) {
                dest_exnames.push_back( move( name ) );
            }
        }

        // attribute/feature names
        dest_attrnames.clear();
        {
            stringstream linestream( nutil::getline( infile ) );
            for( string name; linestream >> name; ) {
                dest_attrnames.push_back( move( name ) );
            }
        }

        // numtrees nmin numattr optimization layers (grow parameters)
        size_t numtrees = 0;
        {
            stringstream linestream( nutil::getline( infile ) );
            if( !(linestream >> numtrees >> dest_nmin
                >> dest_numattr >> dest_optimizationlayers )
            ) {
                cerr << "error reading input file: " << filename << endl;
                return false;
            }
        }

        // trees
        for( size_t t = 0; t < numtrees; t++ ) {

            dectree_t *pt = dectree_t::loadfromstream( infile, dest_forest.indextolabel.size() );
            if( pt == nullptr ) return false;
            dest_forest.trees.push_back( pt );
        }

        return true;
    }

    bool
    storemodeltofile(
        const string &filename, // overwritten
        const nexamples::cexampleset &exampleset,
        const forest_t &forest,
        size_t nmin,
        size_t numattr,
        size_t optimizationlayers
    ) {

        ofstream outfile( filename );
        if( !outfile ) {
            cerr << "error creating output file: " << filename << endl;
            return false;
        }

        // labelname
        outfile << exampleset.labelname;
        // ordered list of label/class values
        for( auto &l : exampleset.labelset ) outfile << " " << l;
        outfile << "\n";

        // excluded feature names
        auto ei = exampleset.exnames.cbegin();
        for( size_t i = 0, max = exampleset.exnames.size(); i < max; i++ ) {
            outfile << (*ei++);
            if( i + 1 < max ) outfile << " ";
        }
        outfile << "\n";

        // attribute / feature names
        auto ai = exampleset.names.cbegin();
        for( size_t i = 0, max = exampleset.names.size(); i < max; i++ ) {
            outfile << (*ai++);
            if( i + 1 < max ) outfile << " ";
        }
        outfile << "\n";

        // numtrees nmin numattr optlayers
        outfile
            << forest.trees.size() << " "
            << nmin << " "
            << numattr << " "
            << optimizationlayers << "\n";

        for( const auto pt : forest.trees ) {
            pt->storetostream( outfile );
        }

        if( outfile.bad() ) {
            cerr << "IO error writing file: " << filename << endl;
            return false;
        }

        return true;
    }
}
