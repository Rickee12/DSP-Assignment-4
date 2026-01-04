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

- `P`= 441：每個 frame 的有效樣本長度。

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
    
    - 若 `idx < N_in`，表示未超出輸入訊號範圍:
      
      - 將左聲道樣本乘上 Hamming window，存入 `xL_m[i][j]`。
        
      - 將右聲道樣本乘上 Hamming window，存入 `xR_m[i][j]`。
        
  - 此處理完成後，每個 frame 皆為長度 N 的複數序列。
 
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
    
      - 若`n < (Q-1)/2`: 將原本的濾波器後半段（從中心點到最後）放到 padded array 的前面可以讓濾波器在頻域的零頻位置（DC）對齊 FFT 序列的開頭，保持對稱性。
      
      - 若`n > (Q-1)/2`： 將原本的濾波器前半段（從開頭到中心點前）放到 padded array 的中心點之後，可以讓濾波器的右半段與左半段對稱排列，保持線性相位並確保頻域乘法後相位正確。
      
      - 若`n == (Q-1)/2`： 將 padded array 的中心點設為 0，避免中心樣本重複。
    
       
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
  - #### 雖然在數學上可以直接在離散頻率點上設定 H[k]，但這種作法在實際濾波設計中會產生嚴重問題。理想低通濾波器在頻域上是矩形函數，其對應的時域衝激響應為無限長的 sinc 函數。當透過 IDFT 取得有限長度的濾波器時，等同於在時域對 sinc 函數進行截斷，這會導致截止頻率附近出現明顯的震盪（吉布斯現象），並使阻帶旁瓣過大。此外，頻率取樣法僅能保證在離散頻率點上響應正確，取樣點之間的頻率響應會產生劇烈起伏，無法維持理想的平坦特性。因此，在實務上通常不直接設定 H[k]，而是採用窗函數法（如 Hamming window）對 sinc 函數進行平滑截斷，或使用 Parks–McClellan 等最佳化方法，亦或在頻率取樣法中加入過渡頻帶，以降低震盪並改善濾波器效能。
    
  - #### 1.Bit Reversal（位元反轉）
     
    概念：在 FFT 中，離散傅立葉變換（DFT）可以不斷的拆成偶數索引與奇數索引的計算。
    拆分過程會改變原始訊號的排列順序。例如對 N=8，經過三層拆分後的輸入順序變為 `[0, 4, 2, 6, 1, 5, 3, 7]`。
      
    二進位觀點：這個排列正好對應將原始索引的二進位數字反轉（bit reversal）：
    0 → 000 → 000 → 0
    1 → 001 → 100 → 4
    2 → 010 → 010 → 2
    3 → 011 → 110 → 6
      
    透過 bit reversal 直接排列訊號，可以避免計算時再搬移資料，使蝶形運算可以直接進行，提升 FFT 計算效率。
  - #### 2.Butterfly（蝶形運算）

    概念：FFT 利用 DFT 的分解特性，將訊號拆成偶數索引與奇數索引計算，逐層合併，稱為蝶形運算。

    拆解原理：對長度 N 的訊號，DFT 可以拆成兩個長度 N/2 的子 DFT：一個是偶數索引子序列，一個是奇數索引子序列。

    每個子 DFT 再拆成更小的偶數/奇數序列，直到每個子序列長度為 1。

    蝶形計算：每次合併偶數與奇數序列時，用以下公式更新：
 
$$
y_{\text{even}} = x_{\text{even}} + W \cdot x_{\text{odd}}
$$

$$
y_{\text{odd}} = x_{\text{even}} - W \cdot x_{\text{odd}}
$$


W 為旋轉因子 $e^{-j \frac{2\pi k}{N}}$ ，負號表示向量旋轉方向。

這個操作就像「左右互相加減旋轉」，逐層將時域訊號轉換到頻域。

舉例（N=8）：

假設訊號 `[x0, x1, x2, x3, x4, x5, x6, x7]`
拆偶奇後第一層計算：
`[x0+x4, x1+x5*W1, x2+x6*W2, x3+x7*W3]`  → 偶數合併奇數

`[x0-x4, x1-x5*W1, x2-x6*W2, x3-x7*W3]`  → 奇數合併偶數
第二層繼續拆分偶奇，直到單點，最終得到完整頻域結果。
     
  - #### 3.IFFT
概念說明: IFFT 的目的是將頻域序列轉換回時域。 FFT 與 IFFT 的數學式的差別基本上只在於指數的正負號以及有沒有除以 N。
FFT 定義為：
      
$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j \frac{2\pi k n}{N}}, \quad n = 0,1,\dots,N-1
$$
      
而 IFFT 定義為：
      
$$ 
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{+j \frac{2\pi k n}{N}}
$$
    
故我們可以利用複數指數的共軛性質:
	
 $$
 \left( e^{-j\theta} \right)^{*} = e^{+j\theta}
 $$
 
可將 IFFT 表示為：

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{j \frac{2\pi k n}{N}}, \quad n = 0,1,\dots,N-1
$$


最後可得：

$$
x[n] = \frac{1}{N}  \text{conj} \Big( \text{FFT} \big( \text{conj}(X[k]) \big) \Big), \quad n = 0,1,\dots,N-1
$$

故我們可以將輸入的頻域信號先取共軛，接著再去做FFT的運算，最後再做一次共軛就等效於IFFT，把原本的頻域信號再轉回時域信號。

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
- #### 理論解釋:在time domain 做 convolution 可以等效於在 frequency domain上相乘，可表示為:

$$
y[n] = x[n] * h[n] \quad \Longleftrightarrow \quad Y[k] = X[k] \cdot H[k]
$$

- #### 程式說明:
  - `for(i = 0; i < num_frames; i++) ` :外層迴圈負責決定正在處理哪一個frame。
    
  - `for(j = 0; j < N; j++)` :內層迴圈負責決定frame裡的哪一個頻域樣本要與濾波器頻譜做相乘。
    
  - `yL_m[i][j] = xL_m[i][j] * H[j]` 與 `yR_m[i][j] = xR_m[i][j] * H[j]` ：分別處理左、右聲道的頻域濾波。

## 9. Overlap-Add 重建(overlap_add)

```c
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

```

說明:
- #### 理論解釋:一般時域線性卷積中，輸出長度為輸入長度加上脈衝響應的長度。然而，FFT-based filtering 中所進行的是有限長度的 circular convolution，FFT 的長度限制了時域輸出的 index 範圍，使得經 IFFT 後的輸出長度仍與 FFT 長度相同。由於輸入訊號以 frame 為單位進行 FFT-based filtering，每個 frame 的長度皆為 N，經過頻域相乘與 IFFT 後，輸出 frame 的長度仍為 N。但是frame hop length比輸出的長度還小那overlap_add 基本上就是不同的 frame 在時間軸上的輸出區段會產生部分重疊 這樣才會對應到一般時域的線性捲積。
  
- #### 程式說明:
  - #### 1. 初始化
    - `complex double w[N]`： Hamming window 陣列定義，並以迴圈`for(i = 0; i < N; i++)`計算每個樣本的窗函數值。
    - `double *norm = calloc(N_ola, sizeof(double))`：建立歸一化陣列，用於累計重疊區段窗函數平方和。
      
  - #### 2. Frame 重疊加總
    - 外層迴圈 `for(i = 0; i < num_frames; i++)`：每個 frame都跑過一次。

    - 計算偏移量 `offset = i * S`，將當前 frame 對應到整體輸出訊號的位置。

    - 內層迴圈 `for(j = 0; j < N; j++)`：將 frame 內的每個樣本都跑過一次，並將其加窗後累加到輸出的暫存陣列裡。

    - 將加窗後的樣本累加到整體輸出陣列：
      
      - `yL_tmp[offset + j] += yL_m[i][j] * w[j]`, `yR_tmp[offset + j] += yR_m[i][j] * w[j];` : 把第 i 個 frame 的第 j 個樣本，放回到全域時間軸的第 (i·S + j) 點
        
  - #### 3. 歸一化(Normalization)
    - 外層迴圈 `for(i = 0; i < N_ola; i++)`：遍歷整個輸出長度。

      -  `if(norm[i] > 1e-12)`: 若累積窗平方和大於 1e-12 
        
      - `yL_tmp[i] /= norm[i]`, `yR_tmp[i] /= norm[i]` : 將累加後訊號除以窗平方和，完成歸一化
        
    


## 10.取樣率轉換(SRC)
```c
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


```
說明:
- #### 理論解釋: 由於在前面的處理階段已經呼叫過 framing, FFT等函數，若在此之前直接對音訊進行 upsampling，會使 frame 長度與 hop size 發生改變，進而影響 FFT-based filtering 與overlap-add的正確性。因此未採用傳統的先upsample再濾波後再downsample的方式，而是改在時域中使用線性內插（linear interpolation）來進行取樣率轉換。線性內插的概念為：當輸出取樣位置對應到輸入訊號中的非整數索引時，會利用該位置左右相鄰的兩個輸入樣本，依其小數距離來進行加權平均，以近似該取樣點的訊號值。這個方法可以在不改變原有 frame 結構的情況下完成取樣率轉換，但由於線性內插並非理想重建，雖能有效降低直接取整數樣本所造成的失真，但對高頻成分仍可能產生些許誤差。
  
- #### 程式說明
  - #### 1. 參數與初始化

    - `phase`：表示目前在輸入訊號中的對應位置（可為非整數）。

    - `step = M / L`：取樣率轉換比例，代表每產生一個輸出樣本，輸入訊號索引前進的量。

    - `N_ola`：輸入訊號長度。

    - `yL_tmp`、`yR_tmp`：SRC 前5暫存的左右聲道輸入訊號（double 精度）。

    - `yL`、`yR`：SRC 後的左右聲道輸出訊號（16-bit 整數）。
   
  - #### 2. 取樣位置計算
    - `for(i = 0; (int)phase < N_ola; i++)`: 當對應到的輸入索引還小於輸入訊號長度時，藉由迴圈持續產生輸出樣本。
      
    - 將 `phase` 拆成整數與小數部分：
      - `idx0 = (int)phase` 與 `idx1 = idx0 + 1`為線性內插所使用的兩個相鄰輸入樣本(整數部分) 。
      - `frac = phase - idx0` 為線性內插使用的小數部分。
        
    - 若 `idx1` 超出輸入長度，則限制在最後一個樣本。
 
    - 使用線性內插估計非整數位置的取樣值：  
      - `yL[i] = (1 - frac) * yL_tmp[idx0] + frac * yL_tmp[idx1]`
      - `yR[i] = (1 - frac) * yR_tmp[idx0] + frac * yR_tmp[idx1]`
        
    - 量化與截斷（Clipping):
      - 將內插後的結果四捨五入並轉換為 16-bit 整數格式。
      - 若數值超出 16-bit 可表示範圍 `[−32768,32767]`，則進行截斷，避免溢位失真。
     
    - 相位更新與儲存輸出長度:

      - `phase += step` : 每產生一個輸出樣本後，更新對應輸入位置。
       
      - `*N_out = i` : 迴圈結束後，將實際輸出的樣本數存入 N_out。

 
## 11.主程式(main function)
```c
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

```
說明:
- #### 1. 宣告變數:

  - `xL`, `xR`：輸入左右聲道 PCM 陣列指標。

  - `yL`, `yR`：輸出左右聲道 PCM 陣列指標。

  - `N_in`, `N_out`：輸入樣本數與輸出樣本數。

  - `fs_in`, `fs_out`：輸入與輸出取樣率。

  - `S`：frame hop length，控制重疊量。

  - `num_frames`：frame 數量。

  - `N_ola`：overlap-add 所需的暫存總長度。

  - `input_wav`, `output_wav`：輸入與輸出 WAV 檔案路徑。

  - `xL_m`, `xR_m`：輸入 frame buffer，存放每個 frame 的 FFT 後資料。

  - `yL_m`, `yR_m`：輸出 frame buffer，存放每個 frame 經濾波後的資料。

  - `yL_tmp`, `yR_tmp`：overlap-add 暫存陣列。

  - `h`：FIR 濾波器係數。

  - `H`：FIR 濾波器 FFT 後的頻率響應。

- #### 2. 讀取輸入 WAV：

  - 呼叫 `read_wav_stereo`，將檔案讀入左右聲道陣列。

  - 若失敗，輸出錯誤訊息並結束程式。

  - 印出輸入樣本數與取樣率。
       
- #### 3. 計算 frame 數與 OLA 長度

  - `num_frames = (N_in + S - 1) / S` 計算 frame 數量。

  - `N_ola = (num_frames - 1) * S + N` 計算 overlap-add 所需暫存長度。
 
  
- #### 4. 配置 frame buffer 記憶體

  - 動態分配 `xL_m`, `xR_m` 以存放每個 frame 的 FFT 輸入。

  - 動態分配 `yL_m`, `yR_m` 以存放每個 frame 經濾波後的輸出。

  - 動態分配 `yL_tmp`, `yR_tmp` 作為 overlap-add 暫存。

  - 動態分配 `yL`, `yR` 作為最終輸出 PCM 陣列。

- #### 5. Framing 與加窗

  - 呼叫 `framing` 將輸入訊號依 frame 切分，並乘以 Hamming window。

  - 每個 frame 長度為 `N`，前面不足的部分以零填充。

- #### 6. FIR 濾波器設計與 FFT

  - 呼叫 `fir_design` 設計低通 FIR 濾波器。

  - 將濾波器係數零填充到 `N`，再呼叫 `FFT` 得到頻域表示 `H`。

  - 對每個輸入 frame 呼叫 `FFT`，轉到頻域進行濾波。
    
- #### 7. 頻域相乘與 IFFT

  - 呼叫 `frequency_multiply` 將每個 frame 的 FFT 與濾波器頻域 `H` 相乘。

  - 對每個 frame 呼叫 `IFFT`，將頻域結果轉回時域。

- #### 8. Overlap-add 重建訊號

  - 呼叫 `overlap_add`，將每個 frame 的時域輸出依 hop length `S`重疊相加，並做窗函數歸一化。

  - 結果存放在 `yL_tmp`, `yR_tmp`。
  
- #### 9. Sample Rate Conversion (線性內插)

  - 呼叫 `SRC`，利用 linear interpolation 進行取樣率轉換。

  - 每個輸出點以左右相鄰輸入樣本依小數距離加權平均。

  - 得到最終輸出 PCM 陣列 `yL`, `yR`，並更新輸出樣本數 `N_out`。

  - 計算輸出取樣率 `fs_out = fs_in * L / M`。

- #### 10. 寫入輸出 WAV
  
  - 呼叫`write_wav_stereo` 將轉換後音訊寫入檔案。

- #### 11. 釋放記憶體與結束程式
 
  - 釋放 `xL`, `xR`, `yL`, `yR` 的動態記憶體。

  - 釋放 frame buffer 與 overlap-add 暫存陣列。

  - 印出輸出取樣率訊息。

  - 結束程式 `return 0`。

## 12. 總結
本次作業利用 FFT-based FIR 低通濾波器，結合 frame-based 處理與 overlap-add 方法，完成對立體聲 WAV 檔案的取樣率轉換 (Sample Rate Conversion, SRC)。FIR 濾波器設計採用窗函數法，時域使用 sinc 函數再乘上 Hamming window，以確保濾波器具有理想的頻率響應特性，有效抑制超過目標奈奎斯特頻率的高頻成分，避免 aliasing。在 frame-based 處理中，輸入訊號先以長度為 P 的時間區段進行分割，並乘以 Hamming window，之後再透過 zero-padding 將每個 frame 擴展至長度 N，以利後續進行 FFT-based 濾波處理。經過 zero-padding 後，每個 frame 皆進行 FFT，將訊號轉換至頻域，並與同樣經 FFT 處理之 FIR 濾波器頻率響應進行逐點相乘。此頻域相乘等效於時域的線性捲積，可大幅降低濾波運算的計算複雜度。完成頻域相乘後，透過 IFFT 將訊號轉回時域，得到每個 frame 的濾波結果。為了避免線性內插在取樣率轉換過程中，因 frame 邊界不連續而產生的雜訊（如沙沙聲），hop length 選擇 P/16，即約 98.6% 的重疊率。藉由這樣的設定可使相鄰 frame 在 overlap-add 過程中平滑銜接，確保時域訊號連續且不產生明顯 artefacts。雖然重疊率較高會增加計算量，但能有效降低由 frame 切分與線性內插共同引起的雜音。在取樣率轉換階段，不是直接進行 upsampling，而是改採用線性內插（linear interpolation）。線性內插透過計算輸出取樣位置對應到輸入訊號的非整數索引，並依小數距離對左右相鄰樣本加權平均，完成近似取樣值。這種做法可以維持原本 frame 結構與 FFT-based 濾波的一致性，避免直接 upsampling 改變 frame 長度而影響後續 FFT 運算，同時有效降低失真，保持音質。最後的結果顯示，輸出音訊維持良好的立體聲效果，沒有明顯混疊或失真，高頻部分經 FIR 低通濾波器有效抑制，而 hop length 與線性內插策略則共同確保音檔沒有受到太多的雜音干擾。
