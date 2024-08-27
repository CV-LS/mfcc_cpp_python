#include "mfcc.cpp"
#include "main.cpp"
#include "ap_fixed.h"
#include "hls_stream.h"

// HLS-compatible stream types
typedef hls::stream<char> char_stream;
//typedef hls::stream<std::string> string_stream; // Note: std::string is not directly supported in HLS

// Simulated file input stream
class SimulatedInputStream {
public:
    void push(const char* line) {
        lines.push(line);
    }

    bool empty() const {
        return lines.empty();
    }

    const char* readLine() {
        return lines.read();
    }

    // Simulated input stream operator overloads
    bool operator>>(const char*& s) {
        if (!lines.empty()) {
            s = lines.read();
            return true;
        }
        return false;
    }

    bool is_open() const {
        return !lines.empty();
    }

private:
    char_stream lines;
};

// Simulated file output stream
class SimulatedOutputStream {
public:
    void writeLine(const char* line) {
        lines.push(line);
    }

    void flush() {
        // In a real HLS scenario, this would flush the stream to a file or another output mechanism.
        // For simulation purposes, we just clear the stream.
        while (!lines.empty()) {
            lines.read();
        }
    }

    // Simulated output stream operator overload
    void operator<<(const char* s) {
        lines.push(s);
    }

    bool is_open() const {
        return true; // Always open for writing
    }

private:
    char_stream lines;
};

// A simple option parser
char* getCmdOption(char **begin, char **end, const char *option) {
    char **iter = std::find(begin, end, option);
    if (iter != end && ++iter != end) {
        return *iter;
    }
    return nullptr;
}

// Main function
void mainFunction(char **argv) {
    const char *USAGE = "compute-mfcc : MFCC Extractor\n";
    USAGE += "OPTIONS\n";
    USAGE += "--input           : Input 16 bit PCM Wave file\n";
    USAGE += "--output          : Output MFCC file in CSV format, each frame in a line\n";
    USAGE += "--inputlist       : List of input Wave files\n";
    USAGE += "--outputlist      : List of output MFCC CSV files\n";
    USAGE += "--numcepstra      : Number of output cepstra, excluding log-energy (default=12)\n";
    USAGE += "--numfilters      : Number of Mel warped filters in filterbank (default=40)\n";
    USAGE += "--samplingrate    : Sampling rate in Hertz (default=16000)\n";
    USAGE += "--winlength       : Length of analysis window in milliseconds (default=25)\n";
    USAGE += "--frameshift      : Frame shift in milliseconds (default=10)\n";
    USAGE += "--lowfreq         : Filterbank low frequency cutoff in Hertz (default=50)\n";
    USAGE += "--highfreq        : Filterbank high freqency cutoff in Hertz (default=samplingrate/2)\n";
    USAGE += "USAGE EXAMPLES\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc\n";
    USAGE += "compute-mfcc --input input.wav --output output.mfc --samplingrate 8000\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list\n";
    USAGE += "compute-mfcc --inputlist input.list --outputlist output.list --numcepstra 17 --samplingrate 44100\n";

    char *wavPath = getCmdOption(argv, argv + 10, "--input");
    char *mfcPath = getCmdOption(argv, argv + 10, "--output");
    char *wavListPath = getCmdOption(argv, argv + 10, "--inputlist");
    char *mfcListPath = getCmdOption(argv, argv + 10, "--outputlist");
    char *numCepstraC = getCmdOption(argv, argv + 10, "--numcepstra");
    char *numFiltersC = getCmdOption(argv, argv + 10, "--numfilters");
    char *samplingRateC = getCmdOption(argv, argv + 10, "--samplingrate");
    char *winLengthC = getCmdOption(argv, argv + 10, "--winlength");
    char *frameShiftC = getCmdOption(argv, argv + 10, "--frameshift");
    char *lowFreqC = getCmdOption(argv, argv + 10, "--lowfreq");
    char *highFreqC = getCmdOption(argv, argv + 10, "--highfreq");

    // Assign variables
    int numCepstra = (numCepstraC ? atoi(numCepstraC) : 12);
    int numFilters = (numFiltersC ? atoi(numFiltersC) : 40);
    int samplingRate = (samplingRateC ? atoi(samplingRateC) : 16000);
    int winLength = (winLengthC ? atoi(winLengthC) : 25);
    int frameShift = (frameShiftC ? atoi(frameShiftC) : 10);
    int lowFreq = (lowFreqC ? atoi(lowFreqC) : 50);
    int highFreq = (highFreqC ? atoi(highFreqC) : samplingRate / 2);

    // Initialise MFCC class instance
    MFCC mfccComputer(samplingRate, numCepstra, winLength, frameShift, numFilters, lowFreq, highFreq);

    // Simulate input and output streams
    SimulatedInputStream wavListFp, mfcListFp;
    SimulatedOutputStream wavFp, mfcFp;

    // Load input data into the simulated streams
    // This part should be adapted to your specific input data
    // For example, you could load strings from a vector or a file into the streams
    // For demonstration, we'll just push some dummy data
    wavListFp.push("1.wav");
    mfcListFp.push("output.mfc");

    // Process wav files
    if (wavPath && mfcPath) {
        // Simulate file opening
        wavFp.push(wavPath);
        mfcFp.push(mfcPath);
        processFile(mfccComputer, wavFp, mfcFp);
    }

    // Process lists
    if (wavListPath && mfcListPath) {
        processList(mfccComputer, wavListFp, mfcListFp);
    }

    // Simulate flushing the output stream
    mfcFp.flush();
}

// Entry point for HLS synthesis
void compute_mfcc_entry_point(char **argv) {
    mainFunction(argv);
}
