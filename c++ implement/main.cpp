#include <algorithm>
#include <iostream>
#include <fstream>
#include "mfcc.cc"

// A simple option parser
char* getCmdOption(char **begin, char **end, const std::string &value) {
    char **iter = std::find(begin, end, value);
    if (iter != end && ++iter != end)
        return *iter;
    return nullptr;
}

// Process each file
int processFile (MFCC &mfccComputer, const char* wavPath, const char* mfcPath) {
    // Initialise input and output streams    
    std::ifstream wavFp;
    std::ofstream mfcFp;
    
    // Check if input is readable
    wavFp.open(wavPath);
    if (!wavFp.is_open()) {
        std::cerr << "Unable to open input file: " << wavPath << std::endl;
        return 1;
    }
    
    // Check if output is writable
    mfcFp.open(mfcPath);
    if (!mfcFp.is_open()) {
        std::cerr << "Unable to open output file: " << mfcPath << std::endl;
        wavFp.close();
        return 1;
    }
   
    // Extract and write features
    if (mfccComputer.process (wavFp, mfcFp))
        std::cerr << "Error processing " << wavPath << std::endl;

    wavFp.close();
    mfcFp.close();
    return 0;
}

// Process lists
int processList (MFCC &mfccComputer, const char* wavListPath, const char* mfcListPath) {
    std::ifstream wavListFp, mfcListFp;

    // Check if wav list is readable
    wavListFp.open(wavListPath);
    if (!wavListFp.is_open()) {
        std::cerr << "Unable to open input list: " << wavListPath << std::endl;
        return 1;
    }

    // Check if mfc list is readable
    mfcListFp.open(mfcListPath);
    if (!mfcListFp.is_open()) {
        std::cerr << "Unable to open output list: " << mfcListPath << std::endl;
        return 1;
    }

    // Process lists
    std::string wavPath, mfcPath;
    while (true) {
        std::getline (wavListFp, wavPath);
        std::getline (mfcListFp, mfcPath);
        if (wavPath.empty() || mfcPath.empty()) {
            wavListFp.close();
            mfcListFp.close();
            return 0;
        }
        if (processFile (mfccComputer, wavPath.c_str(), mfcPath.c_str())) {
            wavListFp.close();
            mfcListFp.close();
            return 1;
        }
    }
}

// Main
int main(int argc, char* argv[]) {
    std::string USAGE = "compute-mfcc : MFCC Extractor\n";
    USAGE += "OPTIONS\n";
    USAGE += "--input           : Input 16 bit PCM Wave file\n";
    USAGE += "--output          : Output MFCC file in CSV format, each frame in a line\n";
    USAGE += "--inputlist       : List of input Wave files\n";
    USAGE += "--outputlist      : List of output MFCC CSV files\n";
    USAGE += "--numMfccFilters      : Number of output cepstra, excluding log-energy (default=12)\n";
    USAGE += "--numMelFilters      : Number of Mel warped filters in filterbank (default=40)\n";
    USAGE += "--samplingrate    : Sampling rate in Hertz (default=16000)\n";
    USAGE += "--winlength       : Length of analysis window in milliseconds (default=25)\n";
    USAGE += "--frameshift      : Frame shift in milliseconds (default=10)\n";
    USAGE += "--lowfreq         : Filterbank low frequency cutoff in Hertz (default=50)\n";
    USAGE += "--highfreq        : Filterbank high freqency cutoff in Hertz (default=samplingrate/2)\n";
    USAGE += "USAGE EXAMPLES\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc --samplingrate 8000\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list --numMfccFilters 17 --samplingrate 44100\n";

    char *wavPath = getCmdOption(argv, argv+argc, "--input");
    char *mfcPath = getCmdOption(argv, argv+argc, "--output");
    char *wavListPath = getCmdOption(argv, argv+argc, "--inputlist");
    char *mfcListPath = getCmdOption(argv, argv+argc, "--outputlist");
    char *numMfccFiltersC = getCmdOption(argv, argv+argc, "--numMfccFilters");
    char *numMelFiltersC = getCmdOption(argv, argv+argc, "--numMelFilters");
    char *samplingRateC = getCmdOption(argv, argv+argc, "--samplingrate");
    char *winLengthC = getCmdOption(argv, argv+argc, "--winlength");
    char *frameShiftC = getCmdOption(argv, argv+argc, "--frameshift");
    char *lowFreqC = getCmdOption(argv, argv+argc, "--lowfreq");
    char *highFreqC = getCmdOption(argv, argv+argc, "--highfreq");

    // Check arguments
    if ((argc<3) || (!(wavPath && mfcPath) && !(wavListPath && mfcListPath))) {
        std::cout << USAGE;
        return 1;
    }
    
    // Assign variables
    int numMfccFilters = (numMfccFiltersC ? atoi(numMfccFiltersC) : 12);
    int numMelFilters = (numMelFiltersC ? atoi(numMelFiltersC) : 40);
    int samplingRate = (samplingRateC ? atoi(samplingRateC) : 44100);
    int winLength = (winLengthC ? atoi(winLengthC) : 25);
    int frameShift = (frameShiftC ? atoi(frameShiftC) : 10);
    int lowFreq = (lowFreqC ? atoi(lowFreqC) : 0);
    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate/2);

    // Initialise MFCC class instance
    MFCC mfccComputer (samplingRate, numMfccFilters, winLength, frameShift, numMelFilters, lowFreq, highFreq);

    // Process wav files
    if (wavPath && mfcPath)
        if (processFile (mfccComputer, wavPath, mfcPath))
            return 1;

    // Process lists
    if (wavListPath && mfcListPath)
        if (processList (mfccComputer, wavListPath, mfcListPath))
            return 1;

    return 0;
}



