// etgrow.cpp - 2016 - Atlee Brink
// Extremely Randomized Trees grower

#include "ndectree.hpp"
#include "nexamples.hpp"
#include "nextratrees.hpp"

#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
    using namespace std;
    using namespace nextratrees;

    const string executablename = "etgrow";

    const string MANDATORY_PARAMETERS = "-t <in:trainfile.csv> -m <out:modelfile>";

    const string PARAMETER_DESCRIPTIONS[] = {
        "\t-e <comma-separated_names_to_exclude>"
            "\n\t\t(default is none) a no-spaces, comma-separated, list of columns to exclude",
        "\t-l <number_of_optimization_layers>"
            "\n\t\t(enables optimization) (default 3) the number of times to re-check"
            "\n\t\teach unique set of hyperparameters: a bigger number reduces bad luck,"
            "\n\t\tbut multiplies run time linearly",
        "\t-nmin <minimum_number_of_examples_for_split>"
            "\n\t\tcontrols complexity of each tree: a bigger number reduces sensitivity",
        "\t-numattr <number_of_attributes_per_split>"
            "\n\t\tcontrols randomness of each tree: a smaller number increases randomness."
            "\n\t\t0 is a special value which indicates to use ceil(sqrt(num_actual))",
        "\t-numtrees <number_of_trees>"
            "\n\t\tnumber of decision trees to plant in the forest",
        "\t-m <out:modelfile>"
            "\n\t\tspecify output model file",
        "\t-t <in:trainfile.csv>"
            "\n\t\tspecify input training data file, in comma-separated-value format",
        "\t-y <label_column_name>"
            "\n\t\t(default is last column) the name of the column that contains labels/class/Y"
    };

    string trainfile;
    string modelfile;
    string labelcolumnname;
    set< string > excludedfeatures;
    bool dooptimize = false;
    size_t optimizationlayers = 3;

    // number of trees in ensemble,
    // linearly affects memory and time
    size_t numtrees_def = 10;
    bool numtrees_specified = false; // if true, don't optimize numtrees
    const size_t numtrees_p2_min = 3; // 2^3 = 8      (small forest)
    const size_t numtrees_p2_max = 10; // 2^10 = 1024 (big forest)

    // minimum number of remaining examples needed to create a branch in a decision tree,
    // inversely affects memory and time: bigger number reduces size of each tree
    size_t nmin_def = 4;
    bool nmin_specified = false; // if true, don't optimize nmin
    const size_t nmin_p2_min = 1; // 2^1 = 2    (detailed)
    const size_t nmin_p2_max = 8; // 2^8 = 256  (smoothed)

    // number of attributes to choose randomly when creating a branch,
    // K = 1 creates fully-random trees, with no choice of "best" attribute,
    // while K = n minimizes randomness to just the split values, but also takes
    //   much longer to train.
    // typical value is ceil( sqrt( num_attr ) ), but the best value depends
    //   on the dataset.
    // K > n has no effect.
    size_t numattr_def = 0; // 0 => ceil( sqrt( num_attr ) ), else literal
    bool numattr_specified = false; // if true, don't optimize numattr

    void
    showusage() {
        cout << "usage: " << executablename << " " << MANDATORY_PARAMETERS;
        cout << " [optional other parameters]\n\n";
        for( auto &desc : PARAMETER_DESCRIPTIONS ) cout << desc << "\n\n";
    }

    bool
    processarguments( int argc, char *argv[] ) {
        // need at least four arguments: -t trainfile and -m modelfile
        if( argc < 1 + 2 + 2 ) return false;
        
        // process command line
        for( int argi = 1; argi < argc; argi++ ) {
            string arg( argv[ argi ] );

            bool good = false;

            // check for arguments that take the nextarg argument also, like "-t <trainfile.csv>"
            if( argi + 1 < argc ) {
                string nextarg( argv[ argi + 1 ] );

                if( arg == "-e" ) { // excluded columns (names)
                    stringstream namestream( nextarg );
                    for( string name; getline( namestream, name, ',' ); ) {
                        excludedfeatures.insert( name );
                    }
                    good = true;
                }
                else if( arg == "-l" ) { // optimization layers
                    try {
                        long num = stol( nextarg );
                        if( num < 1 ) {
                            cout << "expected something like: -l (a positive number), "
                                "got -l " << nextarg << "\n";
                        } else {
                            dooptimize = true;
                            optimizationlayers = (size_t)num;
                            good = true;
                        }
                    } catch( exception &e ) {
                        cout << "expected something like: -l 5, "
                            "got -l " << nextarg << "\n";
                    }
                }
                else if( arg == "-nmin" ) { // min num examples per split
                    try {
                        long num = stol( nextarg );
                        if( num < 1 ) {
                            cout << "expected something like: -nmin (a positive number), "
                                "got -nmin " << nextarg << "\n";
                        } else {
                            nmin_def = (size_t)num;
                            nmin_specified = true;
                            good = true;
                        }
                    } catch( ... ) {
                        cout << "expected something like: -nmin 2, "
                            "got -nmin " << nextarg << "\n";
                    }
                }
                else if( arg == "-numattr" ) { // num attributes per split
                    try {
                        long num = stol( nextarg );
                        if( num < 0 ) {
                            cout << "expected something like: -numattr (a nonnegative number), "
                                "got -numattr " << nextarg << "\n";
                        } else {
                            numattr_def = (size_t)num;
                            numattr_specified = true;
                            good = true;
                        }
                    } catch( ... ) {
                        cout << "expected something like: -numattr 10, "
                            "got -numattr " << nextarg << "\n";
                    }
                }
                else if( arg == "-numtrees" ) { // num trees per forest
                    try {
                        long num = stol( nextarg );
                        if( num < 1 ) {
                            cout << "expected something like: -numtrees (a positive number), "
                                "got -numtrees " << nextarg << "\n";
                        } else {
                            numtrees_def = (size_t)num;
                            numtrees_specified = true;
                            good = true;
                        }
                    } catch( ... ) {
                        cout << "expected something like: -numtrees 100, "
                            "got -numtrees " << nextarg << "\n";
                    }
                }
                else if( arg == "-m" ) { // model file (output)
                    modelfile = nextarg;
                    good = true;
                }
                else if( arg == "-t" ) { // training file (input)
                    trainfile = nextarg;
                    good = true;
                }
                else if( arg == "-y" ) { // name of dependent variable / class column / Y
                    labelcolumnname = nextarg;
                    good = true;
                }

                if( good ) argi++;
            }

            if( !good ) {
                cout << "unrecognized command line parameter: " << arg << "\n\n";
                return false;
            }
        }

        // check that at least trainfile and modelfile were set
        bool isgood = true;
        if( trainfile.empty() ) {
            cout << "command line parameter needed: -t <in:trainfile.csv>" << "\n";
            isgood = false;
        }
        if( modelfile.empty() ) {
            cout << "command line parameter needed: -m <out:modelfile>" << "\n";
            isgood = false;
        }
        
        if( !isgood ) cout << "\n";

        return isgood;
    }
}

int
main(
    int argc,
    char *argv[]
) {

    cout << "extremely randomized trees grower, coded by Atlee Brink\n\n";

    if( !processarguments( argc, argv ) ) {
        showusage();
        return 0;
    }

    // load training set
    cout << "reading examples..." << flush;
    bool islabeled = true;
    bool waslabeled = !labelcolumnname.empty();
    nexamples::cexampleset trainsuperset;
    if( !trainsuperset.loadfromfile(
        trainfile,
        labelcolumnname,
        islabeled,
        excludedfeatures )
    ) {
        return 1;
    }
    cout << " " << trainsuperset.labels.size() << " examples read" << endl;
    if( !waslabeled ) {
        cout << "assuming label column is: " << trainsuperset.labelname << endl;
    }

    forest_t bestforest;
    double bestaccuracy = -1.0;
    size_t bestnumtrees = 0;
    size_t bestnmin = 0;
    size_t bestnumattr = 0;

    if( !numattr_def ) numattr_def = (size_t)ceil( sqrt( (double)trainsuperset.names.size() ) );

    if( dooptimize ) { // optimize hyperparameters

        size_t totalcombos =
            (numtrees_specified ? 1 : (numtrees_p2_max - numtrees_p2_min + 1)) *
            (nmin_specified ? 1 : (nmin_p2_max - nmin_p2_min + 1)) *
            (numattr_specified ? 1 : trainsuperset.names.size());

        cout << "total combinations to check: " << totalcombos << endl;
        cout << "total ensembles to build: " << (optimizationlayers * totalcombos) << endl;

        double trainproportion = 0.7;

        // multithreading stuff
        mutex best_mutex;
        size_t numcombos = 0; // keeps track of how many have been tested so far

        mutex numproc_mutex;
        size_t numproc = 0;
        size_t maxproc = thread::hardware_concurrency();

        mutex done_mutex;
        unique_lock<mutex> done_lock( done_mutex, defer_lock );
        condition_variable done_condition;

        auto notfull_predicate = [&]() {
            bool ret;
            numproc_mutex.lock();
            ret = numproc < maxproc;
            numproc_mutex.unlock();
            return ret;
        };

        auto empty_predicate = [&]() {
            bool ret;
            numproc_mutex.lock();
            ret = numproc < 1;
            numproc_mutex.unlock();
            return ret;
        };

        // prepare multiple random training/testing splits
        vector< nexamples::cexampleset > trainsets, testsets;
        for( size_t layer = 0; layer < optimizationlayers; layer++ ) {
            auto traintest = trainsuperset.split( trainproportion );
            trainsets.push_back( move( traintest[0] ) );
            testsets.push_back( move( traintest[1] ) );
        }

        // tests a particular set of hyperparameters,
        // by averaging the accuracy of optimizationlayers different forests.
        auto dotest = [&](
            size_t numtrees, size_t nmin, size_t numattr
        ) {

            double accuracysum = 0.0;

            for( size_t layer = 0; layer < optimizationlayers; layer++ ) {

                auto &trainset = trainsets[layer];
                auto &testset = testsets[layer];
                
                forest_t forest = build_an_extra_ensemble( trainset, numtrees, nmin, numattr );

                size_t numcorrect = 0;
                for( size_t i = 0; i < testset.labels.size(); i++ ) {
                    label_t predicted = forest.classify( testset.featurevectors[i] );
                    numcorrect += predicted == testset.labels[i];
                }

                accuracysum += (double)numcorrect / testset.labels.size();
            }

            double accuracy = accuracysum / optimizationlayers;

            best_mutex.lock();

            double progress = (double)(++numcombos) / totalcombos;
            bool isbest = accuracy > bestaccuracy;
            if( isbest ) {
                cout << "\x1B[7m";
                bestaccuracy = accuracy;
                bestnumtrees = numtrees;
                bestnmin = nmin;
                bestnumattr = numattr;
            }
            if( bestaccuracy >= 0.0 ) cout << "\r";
            cout
                << setw(3) << right << (long)(progress * 100.0) << "%, "
                << "numtrees = " << setw(4) << left << numtrees << ", "
                << "nmin = " << setw(3) << left << nmin << ", "
                << "numattr = " << setw(3) << left << numattr << ", "
                << "accuracy = " << setw(7) << left << setprecision(5) << accuracy << " ";
            cout
                << " (best: "
                << setprecision(3) << bestaccuracy << ", "
                << bestnumtrees << ", "
                << bestnmin << ", "
                << bestnumattr << ")   " << flush;
            if( isbest ) cout << "\x1B[0m";

            best_mutex.unlock();

            numproc_mutex.lock();
            numproc--;
            numproc_mutex.unlock();

            done_condition.notify_one();
        };

        for( size_t M2 = numtrees_p2_min; M2 <= numtrees_p2_max; M2++ ) {

            size_t numtrees = numtrees_specified ? numtrees_def : 1 << M2;

            for( size_t nmin2 = nmin_p2_min; nmin2 <= nmin_p2_max; nmin2++ ) {
                
                size_t nmin = nmin_specified ? nmin_def : 1 << nmin2;

                for( size_t K = 1; K <= trainsuperset.names.size(); K++ ) {

                    size_t numattr = numattr_specified ? numattr_def : K;

                    done_condition.wait( done_lock, notfull_predicate );

                    numproc_mutex.lock();
                    numproc++;
                    new thread( dotest, numtrees, nmin, numattr );
                    numproc_mutex.unlock();

                    if( numattr_specified ) break;
                }

                if( nmin_specified ) break;
            }

            if( numtrees_specified ) break;
        }

        // wait for procs to finish
        done_condition.wait( done_lock, empty_predicate );

        cout << "\nbest result: "
            << "numtrees = " << bestnumtrees << ", "
            << "nmin = " << bestnmin << ", "
            << "numattr = " << bestnumattr << ", "
            << "accuracy = " << bestaccuracy << endl;


        cout << "building best forest over whole training set..." << flush;
        bestforest = build_an_extra_ensemble(
            trainsuperset, bestnumtrees, bestnmin, bestnumattr
        );
        cout << " done." << endl;

    }
    else { // don't optimize, just use default hyperparameters

        auto &trainset = trainsuperset;
        auto &testset = trainsuperset;

        // train an forest
        size_t numtrees = numtrees_def;
        size_t nmin = nmin_def;
        size_t numattr = numattr_def;

        cout << "parameters: "
            << "numtrees = " << numtrees << ", "
            << "nmin = " << nmin << ", "
            << "numattr = " << numattr << "\n";
        cout << "building forest of " << numtrees << " trees..." << flush;

        forest_t forest = build_an_extra_ensemble(
            trainset,
            numtrees,
            nmin,
            numattr
        );

        cout << " done" << endl;

        // test accuracy
        vector< label_t > indextolabel;
        for( auto &l : testset.labelset ) indextolabel.push_back( l );

        cout << "accuracy on training set with one tree..." << flush;
        const auto pt = forest.trees[0];
        size_t numcorrect = 0;
        for( size_t i = 0; i < testset.labels.size(); i++ ) {

            label_t predicted = classify_from_tree(
                pt,
                indextolabel,
                testset.featurevectors[i]
            );

            numcorrect += predicted == testset.labels[i];
        }

        double accuracy = (double)numcorrect / testset.labels.size();
        cout << " " << accuracy << endl;

        cout << "accuracy on training set with forest..." << flush;
        numcorrect = 0;
        for( size_t i = 0; i < testset.labels.size(); i++ ) {
            label_t predicted = forest.classify( testset.featurevectors[i] );
            numcorrect += predicted == testset.labels[i];
        }

        accuracy = (double)numcorrect / testset.labels.size();
        cout << " " << accuracy << endl;

        bestforest = move( forest );
        bestaccuracy = accuracy;
        bestnumtrees = numtrees;
        bestnmin = nmin;
        bestnumattr = numattr;
    }

    // write model to file
    if( !nextratrees::storemodeltofile(
        modelfile,
        trainsuperset,
        bestforest,
        bestnmin,
        bestnumattr,
        dooptimize ? optimizationlayers : 1
    ) ) {
        return 1;
    }

    return 0;
}
