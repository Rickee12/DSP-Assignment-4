#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#define PI 3.14159265359
#define L 80
#define M 441
#define P 441
#define Q 1025
#define N 2048
#define Wc  (PI/M)

// WAVHeader structure for PCM WAV files
typedef struct {
    char riff[4];
    uint32_t chunk_size;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
} WAVHeader;

int read_wav_stereo(const char *filename, int16_t **L_buf, int16_t **R_buf, int *N_in, int *fs)
{
	FILE *fp = fopen(filename, "rb");      // Open WAV file in binary read mode
	
	// check wav file
	if(!fp) 
	{
		return -1;
	}
	
	WAVHeader h;                               // WAV file header structure
	fread(&h, sizeof(WAVHeader), 1, fp);        // Read WAV header from file
	
	// Check if the WAV file is stereo and 16-bit PCM
	if(h.num_channels != 2 || h.bits_per_sample != 16)
	{
		fclose(fp);
		return -1;
	}
	
	*fs = h.sample_rate;
	*N_in = h.data_size/4;
	
	*L_buf = (int16_t*)malloc((*N_in) * sizeof(int16_t));    // Allocate memory for left channel
    *R_buf = (int16_t*)malloc((*N_in) * sizeof(int16_t));    // Allocate memory for right channel
	
	//Read stereo samples
	int i;
	for( i = 0; i < *N_in; i++)
	{
		fread(&(*L_buf)[i], sizeof(int16_t), 1, fp);     
		fread(&(*R_buf)[i], sizeof(int16_t), 1, fp);     
	}
	
	fclose(fp);  //close file
	return 0;
}

void write_wav_stereo(const char *filename, const int16_t *L_buf, const int16_t *R_buf, int N_out, int fs)
{
	FILE *fp = fopen(filename, "wb");          // Open file in binary write mode
	
	WAVHeader h = {
        {'R','I','F','F'},
        36 + N_out * 4,
        {'W','A','V','E'},
        {'f','m','t',' '},
        16, 1, 2, fs,
        fs * 4, 4, 16,
        {'d','a','t','a'},
        N_out * 4
    };
    
    fwrite(&h, sizeof(WAVHeader), 1, fp);  // Write WAV header to file
	
	//Write stereo samples
    int i;
	for(i = 0; i < N_out; i++)        
	{
		fwrite(&L_buf[i], sizeof(int16_t), 1, fp);  
		fwrite(&R_buf[i], sizeof(int16_t), 1, fp);   
	}
	
	fclose(fp);   //close file
}

void framing(int N_in, int S, int num_frames, int16_t *L_in, int16_t *R_in, complex double **xL_m, complex double **xR_m)
{
	int i, j, idx;
	complex double w[N];
	for(i = 0; i < N; i++)
	{
		w[i] = 0.54 - 0.46 * cos (2 * PI * i / ( N - 1 ) );   // hamming window
	}
		
	for(i = 0; i < num_frames; i++)  //Initialize
	{
		for(j = 0; j < N; j++)
		{
			xL_m[i][j] = 0;
			xR_m[i][j] = 0;
		}
	}	
	
	for(i = 0; i < num_frames ; i++)    // Loop over frames
	{
		for(j = 0; j < P; j++)          // Loop over samples in frame and multiply mindow
		{
			idx = i * S + j;
			if(idx < N_in)
			{
				xL_m[i][j] = L_in[idx] * w[j];
			    xR_m[i][j] = R_in[idx] * w[j];
			}
		}
	}
}

//filter disign
void fir_design(double *h_pad)
{
	int m, n, idx, mid = (Q - 1) / 2;  // mid is filter center index
	double sum = 0, sinc;
	double h[Q];

	for(n = 0; n < Q; n++)
	{
		m = n - mid;
		if(m == 0)             //sinc center
		{
			sinc = Wc/PI;              
		}
		else
		{
			sinc = sin( Wc * m) / (PI * m);
		}
	    double w = 0.54 - 0.46 * cos(2 * PI * n / (Q - 1));   //hamming window
	    h[n] = sinc*w;	      //impulse response
	}           
	     
    // Normalize filter coefficients    
	for(n = 0; n < Q; n++)
	{
		sum += h[n];
	}
	
	for(n = 0; n < Q; n++)
	{
		h[n] /= sum;
	}
	
	// zero-padding
	for(n = 0; n < N; n++)  //Initialization
	{
		h_pad[n] = 0;
	}
	
	for(n = 0; n < N; n++) 
	{
		if(n < (Q - 1) / 2)                 // take the second half of h and put it at the beginning
		{
			h_pad[n] = h[(Q - 1) / 2 + n];
		}
		else if(n > (Q - 1) / 2)            // take the first half of h and put it after the center
		{
			h_pad[n] = h[n - (Q - 1) / 2];
		}
		else                                // set the center point to 0 
		{
			h_pad[n] = 0;
		}
	}
}

//FFT
void FFT(complex double *x)
{
	// bit reversal
	int i, j, k, log2N = 11;
	complex double tmp[N];
	
	for(i = 0; i < N; i++)           // Loop over all indices
	{
		k = 0;
		for(j = 0; (1 << j) < N; j++ )       // Loop over all bits
		{
			if(i & (1 << j))                  // Check if the j-th bit of i is 1
			{
				k |= 1 << ((int)log2N - 1 - j);
			}
		}
		if(i < k)                     // Swap x[i] and x[k]
		{
			tmp[i] = x[i];
			x[i] = x[k];
			x[k] = tmp[i];
		}
	}
	
	int s, m, m2;
	complex double  u, t, W, Wm;
	
	for(s = 1; (1 << s) <= N; s++)       // Loop over FFT stages
	{
		m = 1 << s;
		m2 = m/2;
		Wm = cexp( -I * 2 * PI  / m);
		
		for(i = 0; i < N; i+=m)    // Loop over each butterfly group
		{
			W = 1;
			for(j = 0 ; j < m2; j++)        // Loop over each butterfly in the group
			{
				u = x[i + j];
				t = W * x[i + j + m2];
				
				x[i + j] = u + t;
				x[i + j + m2] = u - t;
				W *= Wm;
			}
			
		}
	}
}

void IFFT(complex double *y)
{
	int i;
	
	for(i = 0; i < N; i++)        // Take complex conjugate of input
	{
		y[i] = conj(y[i]);
	}
	
	FFT(y);
	
	for(i = 0; i < N; i++)        // Take complex conjugate again and scale by N
	{
		y[i] = conj(y[i]) / N;
	}
}

// frequency_multiply
void frequency_multiply(int num_frames, complex double **xL_m, complex double **xR_m, complex double **yL_m, complex double **yR_m, complex double *H)
{
	int i, j;
	
	for(i = 0; i < num_frames; i++) 
	{
		for(j = 0; j < N; j++)
		{
			yL_m[i][j] = xL_m[i][j] * H[j];
		    yR_m[i][j] = xR_m[i][j] * H[j];
		}
	}
}

// overlap_add
void overlap_add(int num_frames, int S, int N_ola, complex double **yL_m, complex double **yR_m, complex double *yL_tmp, complex double *yR_tmp)
{
	int i, j, offset;
	
	complex double w[N];
	double *norm = (double *)calloc(N_ola, sizeof(double));
	
	for(i = 0; i < N; i++)
	{
		w[i] = 0.54 - 0.46 * cos (2 * PI * i / ( N - 1 ) );   // hamming window
	}
	
	for(i = 0; i < num_frames; i++)        // Overlap and multiply window
	{
		offset = i * S;   
		for(j = 0; j < N; j++)     
		{
			yL_tmp[offset + j] += yL_m[i][j] * w[j];
			yR_tmp[offset + j] += yR_m[i][j] * w[j];
			norm[offset + j] += w[j]* w[j];
		}
	}
	
	for(i = 0; i < N_ola; i++)    // Normalize
	{
		if(norm[i] > 1e-12 )
		{
			yL_tmp[i] /= norm[i];
	        yR_tmp[i] /= norm[i];
		}
	}
}

//SRC
void SRC(complex double *yL_tmp, complex double *yR_tmp, int16_t *yL, int16_t *yR, int N_ola, int *N_out)
{
    double phase = 0.0;
    double step = (double)M / L;  // sampling rate change ratio
    double frac;
    int i, idx0, idx1;
    for(i = 0; (int)phase < N_ola; i++)
    {
        idx0 = (int)phase;     // integer part
        idx1 = idx0 + 1;
        frac = phase - idx0;   // fractional part
        if(idx1 >= N_ola)
        {
            idx1 = N_ola - 1;
        }
        
        // Linear interpolation
        yL[i] = (int16_t)round((1.0 - frac) * creal(yL_tmp[idx0]) + frac * creal(yL_tmp[idx1]));
        yR[i] = (int16_t)round((1.0 - frac) * creal(yR_tmp[idx0]) + frac * creal(yR_tmp[idx1]));
        
        
        // Clipping to 16-bit range
        if(yL[i] > 32767)
        {
            yL[i] = 32767;
        }
        if(yL[i] < -32768)
        {
            yL[i] = -32768;
        }
        if(yR[i] > 32767)
        {
            yR[i] = 32767;
        }
        if(yR[i] < -32768)
        {
            yR[i] = -32768;
        }
        
        phase += step;
    }
    *N_out = i;   //store the total numbers of output samples
}

int main(void)
{
	int i;
	int16_t *xL, *xR, *yL, *yR;
	int N_in, N_out, N_out_L, N_out_R;
	int fs_in, fs_out;
	int S = P / 16;
	int num_frames, N_ola;

	const char *input_wav = "C:\\Users\\user\\Downloads\\blue_giant_fragment_44.4kHz_16bits_stereo.wav";
	const char *output_wav = "C:\\Users\\user\\Downloads\\fft_output_src.wav";
	
	// Read input stereo WAV
	if(read_wav_stereo(input_wav, &xL, &xR, &N_in, &fs_in) != 0)
	{
		printf("Failed to read WAV file\n");
        return -1;
	}
	
	num_frames = (N_in + S - 1) / S;   
    N_ola = (num_frames - 1) * S + N;  
    
	// Allocate memory for frame buffers
    complex double **xL_m = (complex double **)malloc(num_frames * sizeof(complex double *));
    complex double **xR_m = (complex double **)malloc(num_frames * sizeof(complex double *));
    complex double **yL_m = (complex double **)malloc(num_frames * sizeof(complex double *));
    complex double **yR_m = (complex double **)malloc(num_frames * sizeof(complex double *));
    for(i = 0; i < num_frames; i++)
	{
    	xL_m[i] = (complex double*)malloc(N * sizeof(complex double));
        xR_m[i] = (complex double*)malloc(N * sizeof(complex double));
        yL_m[i] = (complex double*)malloc(N * sizeof(complex double));
        yR_m[i] = (complex double*)malloc(N * sizeof(complex double));
    }
   
  
    complex double *yL_tmp = (complex double *)calloc(N_ola, sizeof(complex double));
    complex double *yR_tmp = (complex double *)calloc(N_ola, sizeof(complex double));
    
    yL = (int16_t *)malloc(N_ola * L / M * sizeof(int16_t));
    yR = (int16_t *)malloc(N_ola * L / M * sizeof(int16_t));

    //framing
    framing(N_in, S, num_frames, xL, xR, xL_m, xR_m);
	printf("input samples = %d, fs = %d HZ\n", N_in, fs_in);
	printf("num_frames = %d, S = %d\n", num_frames, S);
	
	//FFT 
	double h[N];
	complex double H[N]; 
	fir_design(h);
	
	//frequency response
	for(i = 0; i < N; i++)
	{
		H[i] = h[i];
	}
	FFT(H);
	
	// signal spectrum
	for(i = 0; i < num_frames; i++)
	{
		FFT(xL_m[i]);
		FFT(xR_m[i]);
	}
	
	//frequency_multiply
	frequency_multiply(num_frames, xL_m, xR_m, yL_m, yR_m, H);
	
	//IFFT
	for(i = 0; i < num_frames; i++)
	{
		IFFT(yL_m[i]);
		IFFT(yR_m[i]);
	}
	
	//overlap_add
	overlap_add(num_frames, S, N_ola, yL_m, yR_m, yL_tmp, yR_tmp);
	
	//sample rate conversion
    SRC(yL_tmp, yR_tmp, yL, yR, N_ola, &N_out);

	// Write output WAV
	fs_out = fs_in * L / M;
    write_wav_stereo(output_wav, yL, yR, N_out, fs_out);

    free(xL); free(xR);
    free(yL); free(yR);

    printf("SRC done. Output Fs = %d Hz, N_out = %d \n", fs_out, N_out );
    return 0;
}









