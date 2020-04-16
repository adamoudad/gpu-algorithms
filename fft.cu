#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "calculation.h"

unsigned int get_msb(unsigned int x)
{
  /* Returns Most significant bit of x */
  unsigned int v = x;
  unsigned int r = 0;
  while (v >>= 1) {
    r++;
  }
  return r;
}

unsigned int reverseBits(unsigned int num, unsigned int msb)
{
  /* Bit-reverse operation for a Most Significant bit in place msb */
  unsigned int  NO_OF_BITS = msb;
  unsigned int reverse_num = 0;
  int i;
  for (i = 0; i < NO_OF_BITS; i++)
    {
      if((num & (1 << i)))
	reverse_num |= 1 << ((NO_OF_BITS - 1) - i);  
    }
  return reverse_num;
}

__global__ void device_compute_w(double *w_re, double *w_im, int num)
{
  /* Computes num unity roots w_i */
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  w_re[thread_id] = cos(2*PI*thread_id/num);
  w_im[thread_id] = sin(2*PI*thread_id/num);
}

__global__ void device_fft_ifft(double *d_re, double *d_im, double *d_temp_re, double *d_temp_im, double *w_re, double *w_im, int num, int N, bool gate, int flag)
{
  /* Computes dft/idft transform  */
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int Nh = N/2;			      /* half of the size */
  int i = thread_id%N;		      /* id in the restricted sample */
  int half = (i < Nh) ? 1: - 1; /* First(even) or second(odd) half of the restricted sample*/
  int step = (gate) ? 2: 1;
  if(half==1)
    {
      /* If first half of the N sized part of the sample */
      for(int j=0;j<N/2;j++)
	{
	  d_re[thread_id] += d_temp_re[thread_id-i+j];
	  d_im[thread_id] += d_temp_im[thread_id-i+j];
	}
      d_re[thread_id] += d_temp_re[thread_id]
	+ w_re[(step*thread_id)%num]*d_temp_re[thread_id+Nh]
	+ flag*w_im[(step*thread_id)%num]*d_temp_im[thread_id+Nh];
      d_im[thread_id] += d_temp_im[thread_id]
	- flag*w_im[(step*thread_id)%num]*d_temp_re[thread_id+Nh]
	+ w_re[(step*thread_id)%num]*d_temp_im[thread_id+Nh];
	
      }
  else
    {
      for(int j=0;j<N/2;j++)
	{
	  d_re[thread_id] += d_temp_re[thread_id-i+j];
	  d_im[thread_id] += d_temp_im[thread_id-i+j];
	}


      d_re[thread_id] = d_temp_re[thread_id-Nh]
	- w_re[(step*(thread_id-Nh))%num]*d_temp_re[thread_id+Nh]
	- flag*w_im[(step*(thread_id-Nh))%num]*d_temp_im[thread_id+Nh];
      d_im[thread_id] = d_temp_im[thread_id-Nh]
	+ flag*w_im[(step*(thread_id-Nh))%num]*d_temp_re[thread_id+Nh]
	- w_re[(step*(thread_id-Nh))%num]*d_temp_im[thread_id+Nh];
    }

}

__global__ void device_ifft_divide(double *d_re, double *d_im, int num)
{
  /* Divide each element of tables re et im by num */
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  d_re[thread_id] /= num;
  d_im[thread_id] /= num;
}

void launch_kernel(double *h_re, double *h_im, int num, int flag)
{
  
  /* Step 1: Define variables */
  double *d_re, *d_im, *d_temp_re, *d_temp_im; /* device side data */
  double *w_re, *w_im;		/* Unity roots */
  int block_size = 8;
  dim3 dim_grid(num/block_size, 1, 1), dim_block(block_size, 1, 1);
  int msb = get_msb(num);
  int N;
  bool gate = false;

  /* Step 1.1: Compute bit-reverse permutation over host data */
  for(int i = 0; i < num/2 ; i++)
    {
      swap(&h_re[i], &h_re[reverseBits(i,msb)]);
      swap(&h_im[i], &h_im[reverseBits(i,msb)]);
    }

  /* Step 2: Allocate memory on device */
  cudaMalloc((void **)&d_re, sizeof(double) * num);
  cudaMalloc((void **)&d_im, sizeof(double) * num);
  cudaMalloc((void **)&d_temp_re, sizeof(double) * num);
  cudaMalloc((void **)&d_temp_im, sizeof(double) * num);
  cudaMalloc((void **)&w_re, sizeof(double) * num);
  cudaMalloc((void **)&w_im, sizeof(double) * num);
  
  /* Step 3: Copy data to device, and initialize values */
  cudaMemcpy(d_re, h_re, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemcpy(d_im, h_im, sizeof(double) * num, cudaMemcpyHostToDevice);
  cudaMemset(d_temp_re, 0, sizeof(double) * num);
  cudaMemset(d_temp_im, 0, sizeof(double) * num);
  cudaMemset(w_re, 0, sizeof(double) * num);
  cudaMemset(w_im, 0, sizeof(double) * num);
  
  /* Step 4: Kernel calls */
  device_compute_w<<<dim_grid,dim_block>>>(w_re,w_im,num);
  for(N = 2 ; N <= num ; N *= 2)
    {
      if(gate)
	{
	  device_fft_ifft<<<dim_grid,dim_block>>>(d_re,d_im,d_temp_re,d_temp_im,w_re,w_im,num,N,gate,flag);
	}
      else
	{
	  device_fft_ifft<<<dim_grid,dim_block>>>(d_temp_re,d_temp_im,d_re,d_im,w_re,w_im,num,N,gate,flag);
	}
      /* Update compuation flow : d to d_temp or d_temp to d */
      gate = !(gate);
    }
  
  /* Step 4.1: Divide by num if IDFT */
  if(flag == IDFT)
    {
      if(gate)
	{
	  device_ifft_divide<<<dim_grid,dim_block>>>(d_temp_re,d_temp_im,num);
	}
      else
	{
	  device_ifft_divide<<<dim_grid,dim_block>>>(d_re,d_im,num);
	}
    }

  /* Step 5: Copy back to host the results */
  if(gate)
    {
      cudaMemcpy(h_re, d_temp_re, sizeof(double)*num, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_im, d_temp_im, sizeof(double)*num, cudaMemcpyDeviceToHost);
    }
  else
    {
      cudaMemcpy(h_re, d_re, sizeof(double)*num, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_im, d_im, sizeof(double)*num, cudaMemcpyDeviceToHost);
    }
  
  /* Step 6: Free memory */
  cudaFree(d_re);
  cudaFree(d_im);
  cudaFree(d_temp_re);
  cudaFree(d_temp_im);
  
}
