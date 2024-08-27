#include<algorithm>
#include<numeric>
#include<complex>
#include<vector>
#include<map>
#include<math.h>

typedef std::vector<double> v_d;
typedef std::complex<double> c_d;
typedef std::vector<v_d> m_d;
typedef std::vector<c_d> v_c_d;
typedef std::map<int,std::map<int,c_d> > twmap;
struct wavHeader {
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Stereo
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
};

class MFCC {

private:
    const double PI = 3.1415926;
    int fs;
    twmap twiddle;
    size_t winLengthSamples, frameShiftSamples, numMfccFilters, numFFT, numFFTBins, numMelFilters;
    double preEmph, lowFreq, highFreq;
    v_d frame, powerSpectral, melFilterBanks, hamming, mfcc, prevsamples;
    m_d fbank, dct;

private:
    // Hertz to Mel conversion
    inline double hz2mel (double f) {
        return 2595*std::log10 (1+f/700);
    }

    // Mel to Hertz conversion
    inline double mel2hz (double m) {
        return 700*(std::pow(10,m/2595)-1);
    }

    // Twiddle factor computation
    void compTwiddle(void) {
        const c_d J(0,1);      // Imaginary number 'j' 实部为0，虚部为1
        for (int N=2; N<=numFFT; N*=2)
            for (int k=0; k<=N/2-1; k++)
                twiddle[N][k] = exp(2*PI*k/N*J);
    }
    
    // Cooley-Tukey DIT-FFT recursive function
    v_c_d fft(v_c_d x) {
        int N = x.size();
        if (N==1)
            return x;
    
        v_c_d xe(N/2,0), xo(N/2,0), Xjo, Xjo2;
        int i;
    
        // Construct arrays from even and odd indices
        for (i=0; i<N; i+=2)
            xe[i/2] = x[i];
        for (i=1; i<N; i+=2)
            xo[(i-1)/2] = x[i];
    
        // Compute N/2-point FFT
        Xjo = fft(xe);
        Xjo2 = fft(xo);
        Xjo.insert (Xjo.end(), Xjo2.begin(), Xjo2.end());
    
        // Butterfly computations
        for (i=0; i<=N/2-1; i++) {
            c_d t = Xjo[i], tw = twiddle[N][i];
            Xjo[i] = t + tw * Xjo[i+N/2];
            Xjo[i+N/2] = t - tw * Xjo[i+N/2];
        }
        return Xjo;
    }

    //// Frame processing routines
    // Pre-emphasis and Hamming window
    void preEmphHam(void) {
        v_d procFrame(frame.size(), hamming[0]*frame[0]);
        for (int i=1; i<frame.size(); i++)
            procFrame[i] = hamming[i] * (frame[i] - preEmph * frame[i-1]);
        frame = procFrame;
    }

    // Power spectrum computation
    void computePowerSpec(void) {
        frame.resize(numFFT); // Pads zeros
        v_c_d framec (frame.begin(), frame.end()); // Complex frame
        v_c_d fftc = fft(framec);
    
        for (int i=0; i<numFFTBins; i++)
            powerSpectral[i] = pow(abs(fftc[i]),2);
    }

    // Applying log Mel filterbank (melFilterBanks)
    void applymelFilterBanks(void) {
        melFilterBanks.assign(numMelFilters,0);
        
        for (int i=0; i<numMelFilters; i++) {
            // Multiply the filterbank matrix
            for (int j=0; j<fbank[i].size(); j++)
                melFilterBanks[i] += fbank[i][j] * powerSpectral[j];
            // Apply Mel-flooring
            //if (melFilterBanks[i] < 1.0)
            //    melFilterBanks[i] = 1.0;
        }
        
        // Applying log on amplitude
        for (int i=0; i<numMelFilters; i++)
            melFilterBanks[i] = std::log (melFilterBanks[i] + 0.000001);
    }
    
    // Computing discrete cosine transform
    void applyDct(void) {
        mfcc.assign(numMfccFilters+1,0);
        for (int i=0; i<=numMfccFilters; i++) {
            for (int j=0; j<numMelFilters; j++)
                mfcc[i] += dct[i][j] * melFilterBanks[j];
        }
    }

    // Initialisation routines
    // Pre-computing Hamming window and dct matrix
    void initHamDct(void) {
        int i, j;

        hamming.assign(winLengthSamples,0);
        for (i=0; i<winLengthSamples; i++)
            hamming[i] = 0.54 - 0.46 * cos(2 * PI * i / (winLengthSamples-1));

        v_d v1(numMfccFilters+1,0), v2(numMelFilters,0);
        for (i=0; i <= numMfccFilters; i++)
            v1[i] = i;
        for (i=0; i < numMelFilters; i++)
            v2[i] = i + 0.5;

        dct.reserve (numMelFilters*(numMfccFilters+1));        
        double c = sqrt(2.0/numMelFilters);
        double init_c = 1.0 / sqrt(2.0);
        for (i=0; i<=numMfccFilters; i++) {
            v_d dtemp;
            if(i!=0)
                init_c = 1;
            for (j=0; j<numMelFilters; j++)
                dtemp.push_back (c * init_c * cos(PI / numMelFilters * v1[i] * v2[j]));
            dct.push_back(dtemp);
        }
    }

    // Precompute filterbank
    void initFilterbank () {
        // Convert low and high frequencies to Mel scale
        double lowFreqMel = hz2mel(lowFreq);
        double highFreqMel = hz2mel (highFreq);

        // Calculate filter centre-frequencies (calculate mel freq bins,f_pts)
        v_d filterCentreFreq;
        filterCentreFreq.reserve (numMelFilters+2);
        for (int i=0; i<numMelFilters+2; i++) //m_pts = torch.linspace(m_min, m_max, n_mels + 2)
            filterCentreFreq.push_back (mel2hz(lowFreqMel + (highFreqMel-lowFreqMel)/(numMelFilters+1)*i));//f_pts

        // Calculate FFT bin frequencies  (all_freqs)
        v_d fftBinFreq;
        fftBinFreq.reserve(numFFTBins);
        for (int i=0; i<numFFTBins; i++)
            fftBinFreq.push_back (fs/2.0/(numFFTBins-1)*i); //需要numFFTBins（n_fft//2+1）份，在低频为0，高频为fs/2中取值
            
        // Filterbank: Allocate memory
        fbank.reserve (numMelFilters*numFFTBins);
        
        // Populate the fbank matrix
        for (int filt=1; filt<=numMelFilters; filt++) { //slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
            v_d ftemp;
            for (int bin=0; bin<numFFTBins; bin++) {
                //这样无法工作，数值很小 x 10^-15
                // double down_slope,up_slope;
                // down_slope = (fftBinFreq[bin] - filterCentreFreq[filt-1]) / (filterCentreFreq[filt] - filterCentreFreq[filt-1]);
                // up_slope   = (filterCentreFreq[filt+1] - fftBinFreq[bin]) / (filterCentreFreq[filt+1] - filterCentreFreq[filt]);
                // double weight;
                // if(down_slope >up_slope)
                //      weight = up_slope;
                // else
                //      weight = down_slope;
                double weight;
                if (fftBinFreq[bin] < filterCentreFreq[filt-1])
                    weight = 0; // f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
                else if (fftBinFreq[bin] <= filterCentreFreq[filt]) //down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
                    weight = (fftBinFreq[bin] - filterCentreFreq[filt-1]) / (filterCentreFreq[filt] - filterCentreFreq[filt-1]);//除数模拟的是f_diff
                else if (fftBinFreq[bin] <= filterCentreFreq[filt+1]) //up_slopes = slopes[:, 2:] / f_diff[1:] 其中起点1对应的就是f_pts[2]-f_pts[1]
                    weight = (filterCentreFreq[filt+1] - fftBinFreq[bin]) / (filterCentreFreq[filt+1] - filterCentreFreq[filt]);
                else
                    weight = 0;
//                if(weight<1.0) //fb = torch.max(zero, torch.min(down_slopes, up_slopes))
//                    weight = 1.0;
                ftemp.push_back (weight);
            }
            fbank.push_back(ftemp);
        }
    }

    // Convert vector of double to string (for writing MFCC file output)
    std::string v_d_to_string (v_d vec) {
        std::stringstream vecStream;
        for (int i=0; i<vec.size()-1; i++) {
            vecStream << std::scientific << vec[i];
            vecStream << ", ";
        }
        vecStream << std::scientific << vec.back();
        vecStream << "\n";
        return vecStream.str();
    }

public:
    // MFCC class constructor
    MFCC(int sampFreq=16000, int nMfcc=12, int winLength=25, int frameShift=10, int nMel=40, double lf=50, double hf=6500) {
        fs          = sampFreq;             // Sampling frequency
        numMfccFilters  = nMfcc;                 // Number of cepstra
        numMelFilters  = nMel;              // Number of Mel warped filters
        preEmph = 0.97;                 // Pre-emphasis ficient
        lowFreq     = lf;                   // Filterbank low frequency cutoff in Hertz
        highFreq    = hf;                   // Filterbank high frequency cutoff in Hertz
        numFFT      = fs<=20000?512:2048;   // FFT size
        winLengthSamples   = winLength * fs / 1e3;  // winLength in milliseconds
        frameShiftSamples  = frameShift * fs / 1e3; // frameShift in milliseconds

        numFFTBins = numFFT/2 + 1;
        powerSpectral.assign (numFFTBins, 0);
        prevsamples.assign (winLengthSamples-frameShiftSamples, 0);

        initFilterbank();
        initHamDct();
        compTwiddle();
    }

    // Process each frame and extract MFCC
    std::string processFrame(int16_t* samples, size_t N) {
        // Add samples from the previous frame that overlap with the current frame
        // to the current samples and create the frame.
        frame = prevsamples;
        for (int i=0; i<N; i++)
            frame.push_back(samples[i]);
        prevsamples.assign(frame.begin()+frameShiftSamples, frame.end());

        preEmphHam();
        computePowerSpec();
        applymelFilterBanks();
        applyDct();

        return v_d_to_string (mfcc);
    }

    // Read input file stream, extract MFCCs and write to output file stream
    int process (std::ifstream &wavFp, std::ofstream &mfcFp) {
        // Read the wav header    
        wavHeader hdr;
        int headerSize = sizeof(wavHeader);
        wavFp.read((char *) &hdr, headerSize);

        // Check audio format
        if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
            std::cerr << "Unsupported audio format, use 16 bit PCM Wave" << std::endl;
            return 1;
        }
        // Check sampling rate
        if (hdr.SamplesPerSec != fs) {
            std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec << " instead of " << fs <<std::endl;
            return 1;
        }

        // Check sampling rate
        if (hdr.NumOfChan != 1) {
            std::cerr << hdr.NumOfChan << " channel files are unsupported. Use mono." <<std::endl;
            return 1;
        }

        
        // Initialise buffer
        uint16_t bufferLength = winLengthSamples-frameShiftSamples;
        int16_t* buffer = new int16_t[bufferLength];
        int bufferBPS = (sizeof buffer[0]);

        // Read and set the initial samples        
        wavFp.read((char *) buffer, bufferLength*bufferBPS);
        for (int i=0; i<bufferLength; i++)
            prevsamples[i] = buffer[i];        
        delete [] buffer;
        
        // Recalculate buffer size
        bufferLength = frameShiftSamples;
        buffer = new int16_t[bufferLength];
        
        // Read data and process each frame
        wavFp.read((char *) buffer, bufferLength*bufferBPS);
        while (wavFp.gcount() == bufferLength*bufferBPS && !wavFp.eof()) {
            mfcFp << processFrame(buffer, bufferLength);
            wavFp.read((char *) buffer, bufferLength*bufferBPS);
        }
        delete [] buffer;
        buffer = nullptr;
        return 0;
    }
};
