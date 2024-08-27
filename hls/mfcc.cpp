#include "ap_fixed.h"
#include "hls_stream.h"
#include <string>

// Define fixed-point types
typedef ap_fixed<32, 16> fixed_t; // 32-bit fixed point with 16 fractional bits
typedef ap_complex<fixed_t> c_fixed_t;

// Define constants
const fixed_t PI = 3.1415926;
const fixed_t MEL_FLOOR = 0.000001; // Floor value for Mel-frequency filtering

// Define the MFCC class
class MFCC {
public:
    // Constructor
    MFCC(int sampFreq = 16000, int nCep = 12, int winLength = 25, int frameShift = 10, int numFilt = 40, double lf = 50, double hf = 6500) {
        fs = sampFreq; // Sampling frequency
        numCepstra = nCep; // Number of cepstra
        numFilters = numFilt; // Number of Mel warped filters
        preEmphCoef = 0.97; // Pre-emphasis coefficient
        lowFreq = lf; // Filterbank low frequency cutoff in Hertz
        highFreq = hf; // Filterbank high frequency cutoff in Hertz
        numFFT = (fs <= 20000) ? 512 : 2048; // FFT size
        winLengthSamples = winLength * fs / 1e3; // winLength in milliseconds
        frameShiftSamples = frameShift * fs / 1e3; // frameShift in milliseconds

        numFFTBins = numFFT / 2 + 1;
        powerSpectralCoef.resize(numFFTBins, 0);
        prevsamples.resize(winLengthSamples - frameShiftSamples, 0);

        initFilterbank();
        initHamDct();
        compTwiddle();
    }

    // Process each frame and extract MFCC
    std::string processFrame(fixed_t *samples, size_t N) {
        // Add samples from the previous frame that overlap with the current frame
        // to the current samples and create the frame.
        for (int i = 0; i < prevsamples.size(); i++) {
            frame[i] = prevsamples[i];
        }
        for (int i = 0; i < N; i++) {
            frame[i + prevsamples.size()] = samples[i];
        }
        prevsamples = hls::stream<fixed_t>(frame.begin() + frameShiftSamples, frame.end());

        preEmphHam();
        computePowerSpec();
        applyLMFB();
        applyDct();

        return v_d_to_string(mfcc);
    }

    // Read input file stream, extract MFCCs and write to output file stream
    void process(std::istream &wavFp, std::ostream &mfcFp) {
        // Read the wav header
        wavHeader hdr;
        wavFp.read(reinterpret_cast<char *>(&hdr), sizeof(wavHeader));

        // Check audio format
        if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
            mfcFp << "Unsupported audio format, use 16 bit PCM Wave" << std::endl;
            return;
        }

        // Check sampling rate
        if (hdr.SamplesPerSec != fs) {
            mfcFp << "Sampling rate mismatch: Found " << hdr.SamplesPerSec << " instead of " << fs << std::endl;
            return;
        }

        // Check number of channels
        if (hdr.NumOfChan != 1) {
            mfcFp << hdr.NumOfChan << " channel files are unsupported. Use mono." << std::endl;
            return;
        }

        // Initialise buffer
        size_t bufferLength = winLengthSamples - frameShiftSamples;
        fixed_t *buffer = new fixed_t[bufferLength];

        // Read and set the initial samples
        wavFp.read(reinterpret_cast<char *>(buffer), bufferLength * sizeof(fixed_t));
        for (int i = 0; i < bufferLength; i++) {
            prevsamples[i] = buffer[i];
        }
        delete[] buffer;

        // Recalculate buffer size
        bufferLength = frameShiftSamples;
        buffer = new fixed_t[bufferLength];

        // Read data and process each frame
        while (true) {
            wavFp.read(reinterpret_cast<char *>(buffer), bufferLength * sizeof(fixed_t));
            if (wavFp.gcount() != bufferLength * sizeof(fixed_t)) {
                break;
            }
            mfcFp << processFrame(buffer, bufferLength);
        }
        delete[] buffer;
    }

private:
    // Constants
    const fixed_t PI;
    int fs;
    size_t winLengthSamples, numCepstra, numFFT, numFFTBins, numFilters;
    fixed_t preEmphCoef, lowFreq, highFreq;
    fixed_t frame[winLengthSamples];
    fixed_t powerSpectralCoef[numFFTBins];
    fixed_t lmfbCoef[numFilters];
    fixed_t hamming[winLengthSamples];
    fixed_t mfcc[numCepstra];
    fixed_t prevsamples[winLengthSamples - frameShiftSamples];
    fixed_t fbank[numFFTBins][numFilters];
    fixed_t dct[numCepstra][numFFTBins];
    c_fixed_t twiddle[numFFT+1][numFFTBins];

    // Hertz to Mel conversion
    inline fixed_t hz2mel(fixed_t f) {
        return 2595 * log10(1 + f / 700);
    }

    // Mel to Hertz conversion
    inline fixed_t mel2hz(fixed_t m) {
        return 700 * (pow(10, m / 2595) - 1);
    }

    // Non-recursive FFT implementation
    void fft(c_fixed_t *x, int N) {
        if (N == 1) {
            return; // Base case: no further recursion needed
        }

        c_fixed_t *xe = x; // Even part
        c_fixed_t *xo = x + N / 2; // Odd part

        // Construct arrays from even and odd indices
        for (int i = 0; i < N / 2; i++) {
            xe[i] = x[2 * i];
            xo[i] = x[2 * i + 1];
        }

        // Compute N/2-point FFT
        fft(xe, N / 2);
        fft(xo, N / 2);

        // Butterfly computations
        for (int i = 0; i <= N / 2 - 1; i++) {
            c_fixed_t t = xe[i];
            c_fixed_t tw = twiddle[N][i];
            c_fixed_t xOdd = xo[i];

            // Butterfly computation
            c_fixed_t Xjo_i = t + tw * xOdd;
            c_fixed_t Xjo_i_N_2 = t - tw * xOdd;

            // Store results back in the input array
            x[i] = Xjo_i;
            x[i + N / 2] = Xjo_i_N_2;
        }
    }

    // Pre-emphasis and Hamming window
    void preEmphHam() {
        for (int i = 0; i < winLengthSamples; ++i) {
            frame[i] = hamming[i] * (frame[i] - preEmphCoef * ((i == 0) ? 0 : frame[i - 1]));
        }
    }

    // Power spectrum computation
    void computePowerSpec() {
        c_fixed_t framec[numFFT];
        for (int i = 0; i < winLengthSamples; ++i) {
            framec[i] = c_fixed_t(frame[i], 0);
        }

        fft(framec, numFFT);

        for (int i = 0; i < numFFTBins; ++i) {
            powerSpectralCoef[i] = abs(framec[i]) * abs(framec[i]);
        }
    }

    // Applying log Mel filterbank (LMFB)
    void applyLMFB() {
        for (int i = 0; i < numFilters; ++i) {
            lmfbCoef[i] = 0;
            for (int j = 0; j < numFFTBins; ++j) {
                lmfbCoef[i] += fbank[j][i] * powerSpectralCoef[j];
            }
            lmfbCoef[i] = log(lmfbCoef[i] + MEL_FLOOR);
        }
    }

    // Computing discrete cosine transform
    void applyDct() {
        for (int i = 0; i < numCepstra; ++i) {
            mfcc[i] = 0;
            for (int j = 0; j < numFilters; ++j) {
                mfcc[i] += dct[i][j] * lmfbCoef[j];
            }
        }
    }

    // Pre-computing Hamming window and dct matrix
    void initHamDct() {
        for (int i = 0; i < winLengthSamples; ++i) {
            hamming[i] = 0.54 - 0.46 * cos(2 * PI * i / (winLengthSamples - 1));
        }

        for (int i = 0; i <= numCepstra; ++i) {
            for (int j = 0; j < numFilters; ++j) {
                fixed_t angle = PI / numFilters * i * (j + 0.5);
                dct[i][j] = (i == 0 ? 1.0 / sqrt(2.0) : 1.0) * sqrt(2.0 / numFilters) * cos(angle);
            }
        }
    }

    // Precompute filterbank
    void initFilterbank() {
        // Convert low and high frequencies to Mel scale
        fixed_t lowFreqMel = hz2mel(lowFreq);
        fixed_t highFreqMel = hz2mel(highFreq);

        // Calculate filter centre-frequencies
        fixed_t filterCentreFreq[numFilters + 2];
        for (int i = 0; i < numFilters + 2; ++i) {
            filterCentreFreq[i] = mel2hz(lowFreqMel + (highFreqMel - lowFreqMel) / (numFilters + 1) * i);
        }

        // Calculate FFT bin frequencies
        fixed_t fftBinFreq[numFFTBins];
        for (int i = 0; i < numFFTBins; ++i) {
            fftBinFreq[i] = fs / 2.0 / (numFFTBins - 1) * i;
        }

        // Populate the fbank matrix
        for (int filt = 1; filt <= numFilters; ++filt) {
            for (int bin = 0; bin < numFFTBins; ++bin) {
                fixed_t down_slope = (fftBinFreq[bin] - filterCentreFreq[filt - 1]) / (filterCentreFreq[filt] - filterCentreFreq[filt - 1]);
                fixed_t up_slope = (filterCentreFreq[filt + 1] - fftBinFreq[bin]) / (filterCentreFreq[filt + 1] - filterCentreFreq[filt]);

                fixed_t weight = std::max(down_slope, up_slope);
                weight = std::min(weight, 1.0);

                fbank[bin][filt - 1] = weight;
            }
        }
    }

    // Convert vector of double to string (for writing MFCC file output)
    std::string v_d_to_string(fixed_t vec[]) {
        std::stringstream vecStream;
        for (int i = 0; i < numCepstra - 1; ++i) {
            vecStream << std::scientific << vec[i];
            vecStream << ", ";
        }
        vecStream << std::scientific << vec[numCepstra - 1];
        vecStream << "\n";
        return vecStream.str();
    }

    // Twiddle factor computation
    void compTwiddle() {
        c_fixed_t J(0, 1); // Imaginary number 'j'
        for (int N = 2; N <= numFFT; N *= 2) {
            for (int k = 0; k <= N / 2 - 1; ++k) {
                twiddle[N / 2][k] = exp(-2 * PI * k / N * J);
            }
        }
    }
};
