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

- `#include <math.h>`：提供數學函式（如 sin, cos），用於 FIR 濾波器設計。

- `#include <stdlib.h>`：提供動態記憶體配置（malloc, free）。

- `#include <string.h>`：提供字串與記憶體操作函式。

- `PI`：圓周率。

- `L` = 80：取樣率轉換中的內插倍率（upsampling factor）。

- `M` = 441：取樣率轉換中的抽取倍率（downsampling factor）。

- `P`= 1025：FIR 低通濾波器的 tap 數。

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





##  5. FIR 濾波器設計函數（fir_design）

```c
void fir_design(double *h)
{
	int m, n, mid = (P - 1) / 2;  // mid is filter center index
	double sum = 0;
	double sinc;
	
	for(n = 0; n < P; n++)
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
	    double w = 0.54 - 0.46 * cos(2 * PI * n / (P - 1));   //hamming window
	    h[n] = sinc*w;	      //impulse response
	}           
	     
    // Normalize filter coefficients    
	for(n = 0; n < P; n++)
	{
		sum += h[n];
	}
	
	for(n = 0; n < P; n++)
	{
		h[n] /= sum;
	}
}

```


說明:
- #### 因sinc函數在0的時候分母為0導致程式不好處理，故使用一個if-else來分開處理


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


## 6. 多相濾波器分解函數（polyphase_decompose）

```c
void polyphase_decompose(const double *h, double h_poly[L][(P+L-1)/L], int *phase_len)
{
	int n, r, idx;
	int max_len = 0;  // maximum length among all polyphase filters
	for(r = 0; r < L; r++)
	{
		idx = 0;
		for(n = r; n < P; n+=L)        // Extract every L-th coefficient starting from index r
		{
			h_poly[r][idx] = h[n];
			idx++;
		}
		phase_len[r] = idx;     // Store the number of taps for this phase
		if(idx > max_len)
		{
			max_len = idx;
		}
	}
}
```

說明:
- ##### polyphase_decompose使用原因: 因為upsample時會再輸入之間差L-1個0，那因為convolution的運算是輸入和頻率響應相乘後再不斷累加，所以這些為0的值去做convolution基本上就是浪費時間和計算量，而且之後做downsample時也會把大部分的值丟棄，那為了解決這個問題才會使用polyphase_decompose，以跳過某些頻率響應的值以減少花
- #### polyphase_decompose 中外層與內層迴圈之設定原因與意義 : 外層迴圈是列舉所有可能的時間對齊（phase）情況，並為每一種情況建立一個對應的子濾波器。每一個 r 代表一種輸出樣本相對於上採樣後輸入訊號的對齊方式。內層迴圈則負責從原始 FIR 濾波器中，每隔 L 個係數取一次值，收集屬於同一個 phase 的係數。這樣的取法可確保僅保留會與非零輸入樣本相乘的濾波器係數，避免與零值進行不必要的乘加運算。


- #### 1.初始化變數：

  - `max_len = 0`，用來記錄所有相位中最長的濾波器長度。

- #### 2.外層迴圈 `for(r = 0; r < L; r++)`：

  - 處理每個相位 r。

  - 初始化索引 `idx = 0`。

  - 內層迴圈 `for(n = r; n < P; n+=L)`：

    - 從原始 FIR 濾波器係數 h 中，每隔 L 個取一個值，對應到相位 r。

    - 將取出的係數存入 `h_poly[r][idx]`。

    - 索引值加一 `idx++`。

  - 將該相位的係數數量存入 `phase_len[r]`。

  - 更新最大長度 `max_len`（若此相位更長）。



## 7. 多相取樣率轉換函數（src_polyphase）

```c
void src_polyphase(const int16_t *x, int N_in, int16_t *y, int *N_out, double h_poly[L][(P+L-1)/L], const int *phase_len)
{
	int k, r, k0, x_idx;
	int n = 0;
	double acc;
	while(1)
	{
		r = (n * M) % L;            // Compute phase index (fractional part of n*M/L)
		k0 = (n * M - r) / L;       // Compute integer input sample index
		
		if(k0 >= N_in)
		{
			break;
		}
		
		acc = 0.0;  // Reset accumulator for current output sample
		
		
		for(k = 0; k < phase_len[r]; k++)  //convolution
		{
			x_idx = k0 - k;
			if(x_idx >= 0 && x_idx < N_in)
			{
				acc += x[x_idx] * h_poly[r][k];
			}
		}
		
		acc *=  (double)M / (double)L;   
		
		if(acc > 32767)       // Saturation to int16 range
		{
			acc = 32767;
		}
		if(acc < -32768)
		{
			acc = -32768;
		}
		
		y[n++] = (int16_t)acc;   // Store output sample and advance output index
	} 
	
	*N_out = n;
} 
```

說明:

### 多相 SRC 中 phase $r$ 與輸入索引 $k_0$ 的數學推導

考慮取樣率轉換比例為 $L/M$ 的多相 SRC，理想輸出可寫為：

$$y[n] = \sum_m h[m] x_{\uparrow}[nM - m]$$

其中 $x_{\uparrow}[n]$ 是上採樣 $L$ 倍後的訊號，滿足：

$$x_{\uparrow}[\ell] \neq 0 \iff \ell \equiv 0 \pmod{L}$$

也就是上採樣後訊號在非 $L$ 的倍數位置都是 0。

---

#### Step 1：找出對輸出有貢獻的濾波器係數 $m$

由於卷積中只有 $x_{\uparrow}[nM - m] \neq 0$ 的項才會對輸出有貢獻，因此必須滿足：

$$nM - m = Lk \implies m = nM - Lk, \quad k \in \mathbb{Z}$$

這一步是利用上採樣後訊號的零值，排除不必要的計算。

#### Step 2：對 $m$ 做 $L$ 分解（多相分解）

任何整數 $m$ 可唯一寫為：

$$m = k'L + r, \quad r \in \{0, 1, \dots, L-1\}$$

代入 Step 1 的結果 $m = nM - Lk$ 得：

$$k'L + r = nM - Lk \implies nM - r = (k + k') L$$

這一步把濾波器分解成 $L$ 個相位，每個 phase 有自己的一組濾波器係數。

#### Step 3：推出 Phase 指數 $r$

上式表示 $nM - r$ 必須能被 $L$ 整除，因此：

$$\boxed{r = (nM) \bmod L}$$

 $r$ 表示第 $n$ 個輸出樣本應該使用哪一個子濾波器分支。

#### Step 4：推出輸入索引基準 $k_0$

我們定義 $k_0 = k + k'$ 為對應原始輸入序列 $x[k]$ 的索引。由 $nM - r = Lk_0$ 可得：

$$\boxed{k_0 = \frac{nM - r}{L} = \left\lfloor \frac{nM}{L} \right\rfloor}$$

 $k_0$ 是計算第 $n$ 個輸出時，在輸入訊號中對齊的基準位置。


 ### 程式碼說明
- #### 1.初始化變數：

  - `n = 0`，輸出樣本索引。

  - `acc`，暫存累加值。

- #### 2.迴圈 `while(1)`：

  - 計算相位索引 `r = (n * M) % L`，對應於多相濾波器的哪一相。

  - 計算輸入樣本索引 `k0 = (n * M - r) / L`。

  - 若 `k0 >= N_in`，跳出迴圈，表示輸入已用完。

  - 初始化累加器 `acc = 0.0`。

  - 對當前相位的濾波器做卷積：

    - `for(k = 0; k < phase_len[r]; k++)`

       - 計算輸入索引 `x_idx = k0 - k`。

       - 若索引合法 `(0 <= x_idx < N_in)`，將輸入乘上濾波器係數累加到 acc。

  - 將累加值乘上比例 `(double)M / (double)L`，補償取樣率改變造成的增益。

  - 限幅到 `int16` 範圍 [-32768, 32767]。

  - 將累加結果存入輸出陣列 `y[n++]`。

- #### 3.將輸出樣本數存入 `*N_out`。

## 8. 主程式（Main Function）

```c
int main(void)
{
	int16_t *L_in, *R_in;
	int16_t *L_out, *R_out;
	int N_in, N_out_L, N_out_R;
	int fs_in, fs_out;
	const char *input_wav = "C:\\Users\\user\\Downloads\\blue_giant_fragment_44.4kHz_16bits_stereo.wav";
	const char *output_wav = "C:\\Users\\user\\Downloads\\output_src.wav";
	
	// Read input stereo WAV
	if(read_wav_stereo(input_wav, &L_in, &R_in, &N_in, &fs_in) != 0)
	{
		printf("Failed to read WAV file\n");
        return -1;
	}
	printf("input samples = %d, fs = %d HZ\n", N_in, fs_in);
	
	// Design FIR filter and decompose into polyphase components
	double h[P];
	double h_poly[L][(P+L-1)/L];
	int phase_len[L];
	fir_design(h);
	polyphase_decompose(h, h_poly, phase_len);
	
	// Allocate memory for output
	int max_out = (int)(N_in * ((double)L / M)) + 10;
	L_out = (int16_t*)malloc(max_out * sizeof(int16_t));
    R_out = (int16_t*)malloc(max_out * sizeof(int16_t));
    
    // Apply polyphase SRC to left and right channels
    src_polyphase(L_in, N_in, L_out, &N_out_L, h_poly, phase_len);
    src_polyphase(R_in, N_in, R_out, &N_out_R, h_poly, phase_len);

    int N_out = (N_out_L < N_out_R) ? N_out_L : N_out_R;
    fs_out = fs_in * L / M;
    
    // Write output WAV
    write_wav_stereo(output_wav, L_out, R_out, N_out, fs_out);

    free(L_in); free(R_in);
    free(L_out); free(R_out);

    printf("SRC done. Output Fs = %d Hz\n", fs_out);
    return 0;
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


## 9. FIR low-pass filter作圖
![](figure3/FIR_LPF_Magnitude.png)


FIR 低通濾波器之頻率響應分析

本實驗中所設計之 FIR 低通濾波器，其脈衝響應由理想 sinc 函數乘上 Hamming window 所構成。理想情況下，時域中的 sinc 函數在經過傅立葉轉換後，於頻域中對應為一個矩形函數，也就是一個方波，因此可實現理想的低通濾波特性。

在程式中，截止角頻率設定為:

$$ Wc = \frac{\pi}{M} $$

其中 M=441，因此截止頻率相對於 Nyquist 頻率而言非常小。此設定使得濾波器在頻域中的通帶寬度極窄，故在以 FFT 繪製頻率響應時，可觀察到一個頻寬很小的低通通帶，外觀近似於一個非常窄的矩形頻譜。
然而，實際設計中並未使用無限長的 sinc，而是採用有限長度(P=1025）的 FIR 濾波器，並搭配 Hamming window 以降低頻域旁瓣效應。由於有限長截斷的影響，理想矩形頻譜在頻域中會產生漣波（Gibbs phenomenon），而 Hamming window 的使用則能有效抑制旁瓣能量，使頻率響應更加平滑，但同時也會導致通帶邊緣變得不再完全銳利，主瓣略為展寬。由此可得FIR低通濾波器的頻率響應呈現出「通帶狹窄、旁瓣抑制良好、邊緣平滑」的特性，符合在取樣率轉換（Sample Rate Conversion）中作為抗混疊與成像抑制濾波器的設計需求。

## 10. 總結
本次作業利用 FIR 低通濾波器結合 Polyphase 分解方法，完成對立體聲 WAV 檔案的取樣率轉換 (Sample Rate Conversion, SRC)。FIR 濾波器設計採用窗函數法，時域採用 sinc function 再乘上 Hamming window，確保濾波器具有良好的頻率響應特性。由於 sinc 函數在時域對應於矩形函數在頻域，因此濾波器呈現理想低通特性，有效抑制超過目標奈奎斯特頻率的高頻成分，避免aliasing。
為了將原始 44.1 kHz 立體聲 WAV 檔轉換為 80/441 倍的取樣率，透過 Polyphase 分解方法，將 FIR 濾波器拆分為 L 個子濾波器，每個子濾波器只作用於對應相位的輸入樣本。藉由這樣的方式可以使濾波與插值同步進行，大幅減少了運算量。結果顯示，輸出音訊維持良好的音質，無明顯混疊或失真，高頻部分經 FIR 低通濾波器有效抑制，避免超過奈奎斯特頻率的頻譜成分造成aliasing。
