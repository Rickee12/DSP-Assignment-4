# DSP-Assignment-4
### 學號:711481109 姓名:莊祐儒


---
作業包含以FFT實現的Low-Pass Filter來轉換音檔的Sampling Rate的程式碼實現。

---


###  **FFT 低通濾波取樣率轉換(以C語言撰寫)**

## 目錄

1.標頭與常數定義

2.WAV 檔案結構定義

3.WAV 檔案讀取函數（read_wav_stereo）

4.WAV 檔案寫入函數（write_wav_stereo）

5.訊號分幀與加窗處理(framing)

6.FIR 低通濾波器(fir_design)

7.FFT 與 IFFT 運算(FFT & IFFT)

8.頻域濾波處理(frequency_multiply)

9.Overlap-Add 重建(overlap_add)

10.取樣率轉換(SRC)

11.主程式(main function)

12.總結

---

## 1. 標頭與常數定義

```c
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
```

說明：

- `#include <stdio.h>`：提供檔案輸入輸出功能（如 fopen, fread, fwrite）。

- `#include <stdint.h>`：提供固定長度整數型別（如 int16_t），確保音訊資料位元數正確。

- `#include <complex.h>`：提供複數型別與運算（如 complex double, cexp），用於 FFT 與頻域處理。
  
- `#include <math.h>`：提供數學函式（如 sin, cos），用於 FIR 濾波器設計。

- `#include <memory.h>`：提供記憶體操作相關函式（如 memset）。
  
- `#include <stdlib.h>`：提供動態記憶體配置（malloc, free）。
  
- `#include <string.h>`：提供字串與記憶體操作函式。

- `PI`：圓周率。

- `L` = 80：取樣率轉換中的內插倍率（upsampling factor）。

- `M` = 441：取樣率轉換中的抽取倍率（downsampling factor）。

- `P`= 1025：FIR 低通濾波器的 tap 數。

- `Q` = 1025：FIR 低通濾波器的 tap 數。

- `N` = 2048：FFT 與 IFFT 的運算長度。

- `Wc` = PI / M：正規化截止角頻率。






## 2. WAV檔案結構定義

```c
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
```

說明：

- `riff`：標示檔案為 RIFF 格式

- `chunk_size`：整個 WAV 檔案大小

- `wave`：標示音訊格式為 WAVE

- `fmt`：格式區塊識別字串 "fmt "

- `subchunk1_size`：格式區塊大小（PCM 通常為 16）

- `audio_format`：音訊格式（PCM 為 1）

- `num_channels`：聲道數（1 = 單聲道，2 = 立體聲）

- `sample_rate`：取樣率（Hz）

- `byte_rate`：每秒資料位元組數

- `block_align`：每個取樣框架的位元組數

- `bits_per_sample`：每個取樣的位元數

- `data`：資料區塊識別字串 "data"

- `data_size`：實際音訊資料大小






## 3. WAV 檔案讀取函數（read_wav_stereo）

```c
int read_wav_stereo(const char *filename, int16_t **L_buf, int16_t **R_buf, int *N, int *fs)
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
	*N = h.data_size/4;
	
	
	*L_buf = (int16_t*)malloc((*N) * sizeof(int16_t));    // Allocate memory for left channel
    *R_buf = (int16_t*)malloc((*N) * sizeof(int16_t));    // Allocate memory for right channel
	
	
	//Read stereo samples
	int i;
	for( i = 0; i < *N; i++)
	{
		fread(&(*L_buf)[i], sizeof(int16_t), 1, fp);     
		fread(&(*R_buf)[i], sizeof(int16_t), 1, fp);     
	}
	
	fclose(fp);  //close file
	return 0;
}



```

說明：

- `fopen(filename, "rb")`：以二進位讀取模式開啟輸入 WAV 檔案。
若開檔失敗，回傳 -1。

- `fread(&h, sizeof(WAVHeader), 1, fp)`：讀取 WAV 標頭資料到 WAVHeader 結構。
  
- 格式檢查：
  - `h.num_channels != 2`：確認是否為立體聲。

  - `h.bits_per_sample != 16`：確認是否為 16-bit PCM。
   若不符合，關閉檔案 `fclose(fp)` 並回傳 -1。


- 計算與分配記憶體：

  - `*fs = h.sample_rate`：取得取樣率。

  - `*N = h.data_size / 4`：計算每個聲道的樣本數。

  - `*L_buf、*R_buf`：動態分配記憶體給左右聲道陣列。

- 讀取 PCM 音訊資料：

  - 使用 `for`迴圈依序讀取左、右聲道樣本。

- 關閉檔案：

  - `fclose(fp)`：關閉 WAV 檔案。

- 回傳值：

  - `return 0`: 成功回傳 0。



## 4. WAV 檔案寫入函數（write_wav_stereo）
 
 ```c
void write_wav_stereo(const char *filename, const int16_t *L_buf, const int16_t *R_buf, int N, int fs)
{
	FILE *fp = fopen(filename, "wb");          // Open file in binary write mode
	
	WAVHeader h = {
        {'R','I','F','F'},
        36 + N * 4,
        {'W','A','V','E'},
        {'f','m','t',' '},
        16, 1, 2, fs,
        fs * 4, 4, 16,
        {'d','a','t','a'},
        N * 4
    };
    
    fwrite(&h, sizeof(WAVHeader), 1, fp);  // Write WAV header to file
	
	
	//Write stereo samples
    int i;
	for(i = 0; i < N; i++)        
	{
		fwrite(&L_buf[i], sizeof(int16_t), 1, fp);  
		fwrite(&R_buf[i], sizeof(int16_t), 1, fp);   
	}
	
	fclose(fp);   //close file
}
```


說明：

- `fopen(filename, "wb")`：以二進位寫入模式開啟輸出檔案。

- 建立 `WAVHeader h`：設定 RIFF、格式區塊 (fmt) 及資料區塊 (data) 標頭。

- `fwrite(&h, sizeof(WAVHeader), 1, fp)`：將 WAV 標頭寫入檔案。

- `for` 迴圈：將左右聲道的 PCM 音訊資料分別寫入檔案。

- `fclose(fp)`：關閉檔案，完成寫入。





##  5. 訊號分幀與加窗處理(framing)

```c
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

```


說明:
- #### 此函數主要是處理framing的部分，輸入的訊號需被切分為多個重疊的frames並再加上window，以方便後續能夠有效的進行FFT與頻域處理

- #### 1.初始化窗函數與變數：
   - 宣告 `w[N]`，以儲存 Hamming window。
     
   - `w[i] = 0.54 - 0.46 * cos(2πi / (N - 1))`， 藉由`for(i = 0; i < N; i++)`迴圈逐個計算window的係數，以降低頻譜洩漏。

- #### 2.初始化輸出矩陣：

   - 以雙層`for`迴圈將 `xL_m` 與 `xR_m` 中所有 frame 與樣本初始化為 0。
     
   - 外層迴圈`for(i = 0; i < num_frames; i++)` : 將每一個frame依序處理。
     
   - 內層迴圈`for(j = 0; j < N; j++)` : 將frame裡的每一個樣本點都設為0。

- #### 3.分幀與加窗處理：：

  - 外層迴圈先依序處理每一個frame，而內層迴圈則處理每一個frame的前 P 個樣本。

  - 在`for`迴圈裡不斷計算對應的輸入索引
    
    - `i` 表示目前處理的第 `i` 個 frame。
      
    - `S` 為 hop length，代表兩個相鄰frame在時間軸上的位移量。
    
    - `j`則表示為當下那個frame的第 `j` 個樣本。
      
  - 將 frame 起始位置與 frame 內索引相加，得到對應輸入訊號的位置：`idx = i * S + j`。
    
    - 若 `idx < N_in`，表示尚未超出輸入訊號範圍:
      
      - 將左聲道樣本乘上 Hamming window，存入 `xL_m[i][j]`。
        
      - 將右聲道樣本乘上 Hamming window，存入 `xR_m[i][j]`。
        
  - 此處理完成後，每個 frame 皆為長度 N 的複數序列，可用於後續 FFT 分析。
 
## 6. FIR 低通濾波器(fir_design)

```c
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
```

說明:
- #### 因sinc函數在0的時候分母為0導致程式不好處理，故使用一個if-else來分開處理，在zero-padding時將原本濾波器的係數重新排列，也就是將濾波器左右半段對稱搬移到 padded array 前後並將中心設 0，以保持線性相位並對齊 FFT，但因為原本的濾波器長度只有Q，其餘的部分要再補0直到長度為N。


- #### 1.計算濾波器中心索引 `mid = (P-1)/2`。

- #### 2.`for` 迴圈：

   - 計算對應的 `m = n - mid`。

   - 計算 sinc 函數：

     - 若 `m = 0`，取 `sinc = Wc / PI`（中心點值）。

     - 否則 `sinc = sin( Wc * m) / (PI * m)`。

   - 計算 Hamming window` w = 0.54 - 0.46 * cos(2 * PI * n / (P - 1))`。

   - 將 sinc 與窗函數相乘得到濾波器脈衝響應 `h[n] = sinc * w`。

- #### 3.正規化濾波器係數：

  - 先計算總和 `sum = Σ h[n]`。

  - 再將每個係數除以總和 `h[n] /= sum`，確保 DC gain = 1。

- #### 4.Zero-padding：
  - 初始化 padded array:
    - 使用迴圈 `for(n = 0; n < N; n++)` 將 `h_pad[n]` 全部先初始化為 0，後續就不用再額外補0，以方便之後做FFT運算。
  - 再透過迴圈 `for(n = 0; n < N; n++)` 將原始濾波器 `h[n]` 的 Q 個係數重新排列並填入 `h_pad[n]`：
    
    - 若`n < (Q-1)/2`:
      - 將原本的濾波器後半段（從中心點到最後）放到 padded array 的前面可以讓濾波器在頻域的零頻位置（DC）對齊 FFT 序列的開頭，保持對稱性。
        
    - 若`n > (Q-1)/2`：
      - 將原本的濾波器前半段（從開頭到中心點前）放到 padded array 的中心點之後，可以讓濾波器的右半段與左半段對稱排列，保持線性相位並確保頻域乘法後相位正確。

    - 若`n == (Q-1)/2`：
      - 將 padded array 的中心點設為 0，避免中心樣本重複。
       

## 7. FFT 與 IFFT 運算(FFT & IFFT)

```c
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
```

說明:
- ### 理論解釋:
  - #### 1.Bit Reversal（位元反轉）
     
    - 概念：在 FFT 中，離散傅立葉變換（DFT）可拆成偶數索引與奇數索引的計算，不斷拆分直到每個子問題長度為 1。
      
    - 對應到索引：拆分過程會改變原始訊號的排列順序。例如對 N=8，經過三層拆分後的輸入順序變為 `[0, 4, 2, 6, 1, 5, 3, 7]`。
      
    - 二進位觀點：這個排列正好對應將原始索引的二進位數字反轉（bit reversal）：
    - 0 → 000 → 000 → 0
    - 1 → 001 → 100 → 4
    - 2 → 010 → 010 → 2
    - 3 → 011 → 110 → 6
      
    - 透過 bit reversal 直接排列訊號，可以避免計算時再搬移資料，使蝶形運算可以直接進行，提升 FFT 計算效率。
  - #### 2.Butterfly（蝶形運算）

    - 概念：FFT 利用 DFT 的分解特性，將訊號拆成偶數索引與奇數索引計算，逐層合併，稱為蝶形運算。

    - 拆解原理：對長度 N 的訊號，DFT 可以拆成兩個長度 N/2 的子 DFT：一個是偶數索引子序列，一個是奇數索引子序列。

    - 每個子 DFT 再拆成更小的偶數/奇數序列，直到每個子序列長度為 1。

    - 蝶形計算：每次合併偶數與奇數序列時，用以下公式更新：
      
      $$
      y_{\text{even} = x{\text{even} + W * x_{\text{odd}}
      $$
      
      $$
      y_{\text{odd}}  = x_{\text{even} - W * x_{\text{odd}}
      $$
      
    - W 為旋轉因子 $e^{-j \frac{2\pi k}{N}}$ ，負號表示向量旋轉方向。

    - 這個操作就像「左右互相加減旋轉」，逐層將時域訊號轉換到頻域。

    - 舉例（N=8）：

    - 假設訊號 `[x0, x1, x2, x3, x4, x5, x6, x7]`
    - 拆偶奇後第一層計算：
    - `[x0+x4, x1+x5*W1, x2+x6*W2, x3+x7*W3]`  → 偶數合併奇數
    - `[x0-x4, x1-x5*W1, x2-x6*W2, x3-x7*W3]`  → 奇數合併偶數
    - 第二層繼續拆分偶奇，直到單點，最終得到完整頻域結果。
     
  - #### 3.IFFT
    - 概念說明: IFFT 的目的是將頻域序列轉換回時域。觀察 FFT 與 IFFT 的數學形式可以發現，兩者的差別只在於指數的正負號與是否除以 N。
    - FFT 定義為：
      
$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} Y[k] \cdot e^{j \frac{2\pi k n}{N}}
$$
      
    - 而 IFFT 定義為：
      
$$ 
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{+j \frac{2\pi k n}{N}}
$$
    
	- 故我們可以利用複數指數的共軛性質:
 $$
 \left( e^{-j\theta} \right)^{*} = e^{+j\theta}
 $$
    - 可將 IFFT 表示為：
$$
x[n] = \frac{1}{N} \left( \sum_{k=0}^{N-1} X^{*}[k] e^{-j \frac{2\pi k n}{N}} \right)^{*}
$$
    - 也就是：
      
$$
x[n] = \frac{1}{N} \operatorname{conj}(\operatorname{FFT}(\operatorname{conj}(X[k])))
$$
      
    - 故我們可以將輸入的頻域信號先取共軛，接著再去做FFT的運算，最後再做一次共軛就等效於IFFT，把原本的頻域信號再轉回時域信號。

- ### 程式說明
  
- #### 1. FFT(快速傅立葉轉換)
  - #### Bit reversal(位元反轉)
    - 使用雙層迴圈將輸入陣列 `x[i]`的索引進行位元反轉，將資料重新排列。
      
    - 外層迴圈 `for(i = 0; i < N; i++)`：遍歷所有輸入索引。
      
    - 內層迴圈 `for(j = 0; (1 << j) < N; j++)`：將索引 `i` 的二進位表示反轉得到 `k`。
      
    - 若 `i < k`，交換 `x[i]` 與 `x[k]`，完成 bit reversal。
  -  #### (Butterfly)蝶形運算
     - 外層迴圈 `for(s = 1; (1 << s) <= N; s++)`：對每個 FFT 階段進行運算。
       
       - 計算子組大小 `m = 2^s`，每個子組進行 `m/2` 個蝶形運算。
         
     - 中層迴圈 `for(i = 0; i < N; i += m)`：遍歷每個蝶形子組。
       
     - 內層迴圈 `for(j = 0; j < m/2; j++)`：進行蝶形運算。
       
     - 其中旋轉因子 `Wm = e^(-j*2*PI/m)`，`W` 每次累乘更新。
- #### 2. IFFT
  - 對輸入 `y[i]` 取共軛複數：`y[i] = conj(y[i])`。
  - `FFT(y)`:呼叫 FFT 函數計算頻域運算。
  - 再對頻域運算後的結果取共軛並除以 N：`y[i] = conj(y[i])/N`，得到正確的時域訊號。

## 8. 頻域濾波處理(frequency_multiply)

```c
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

```
說明：

- #### 1.宣告變數：

     - `*L_in, *R_in`：輸入左右聲道 PCM 陣列指標。

     - `*L_out, *R_out`：輸出左右聲道 PCM 陣列指標。

     - `N_in`：輸入樣本數，`N_out_L, N_out_R`：左右輸出樣本數。

     - `fs_in, fs_out`：輸入與輸出取樣率。

     - `input_wav, output_wav`：輸入與輸出 WAV 檔案路徑。

- #### 2.讀取輸入 WAV：

     - 呼叫 `read_wav_stereo`，將檔案讀入左右聲道陣列。

     - 若失敗，輸出錯誤訊息並結束程式。

     - 印出輸入樣本數與取樣率。

- #### 3.設計 FIR 濾波器並做多相分解：

     - 宣告 `h` 為濾波器係數，`h_poly` 為多相濾波器矩陣，`phase_len` 記錄每個相位長度。

     - 呼叫 `fir_design` 設計濾波器。

     - 呼叫 `polyphase_decompose` 將濾波器分解為多相結構。

- #### 4.配置輸出陣列記憶體：

     - 計算最大輸出樣本數 `max_out`。

     - 動態分配記憶體給輸出陣列 `L_out` 與 `R_out`。

- #### 5.執行多相 SRC：

     - 對左、右聲道分別呼叫 `src_polyphase` 進行取樣率轉換。

     - 取左右輸出樣本數最小值作為最終輸出長度 `N_ou`t。

     - 計算輸出取樣率 `fs_out = fs_in * L / M`。

- #### 6.寫入輸出 WAV：

     - 呼叫`write_wav_stereo` 將轉換後音訊寫入檔案。

- #### 7.釋放記憶體與結束程式：

     - 釋放 `*L_in, *R_in, *L_out, *R_out` 的動態記憶體。

	 - 印出輸出取樣率訊息。

     - 程式結束 `return 0`。


## 9. 




## 10. 總結
本次作業利用 FIR 低通濾波器結合 Polyphase 分解方法，完成對立體聲 WAV 檔案的取樣率轉換 (Sample Rate Conversion, SRC)。FIR 濾波器設計採用窗函數法，時域採用 sinc function 再乘上 Hamming window，確保濾波器具有良好的頻率響應特性。由於 sinc 函數在時域對應於矩形函數在頻域，因此濾波器呈現理想低通特性，有效抑制超過目標奈奎斯特頻率的高頻成分，避免aliasing。
為了將原始 44.1 kHz 立體聲 WAV 檔轉換為 80/441 倍的取樣率，透過 Polyphase 分解方法，將 FIR 濾波器拆分為 L 個子濾波器，每個子濾波器只作用於對應相位的輸入樣本。藉由這樣的方式可以使濾波與插值同步進行，大幅減少了運算量。結果顯示，輸出音訊維持良好的音質，無明顯混疊或失真，高頻部分經 FIR 低通濾波器有效抑制，避免超過奈奎斯特頻率的頻譜成分造成aliasing。
