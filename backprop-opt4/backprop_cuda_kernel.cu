

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
  //  __shared__ float input_node[HEIGHT];
  //  __shared__ float weight_matrix[HEIGHT][WIDTH];
  __shared__ float weight_matrix[WIDTH];
  //  weight_matrix[ty][tx] = input_hidden_cuda[index] * input_cuda[index_in];

  float r1 = input_hidden_cuda[index] * input_cuda[index_in];
  float r2, r3, r4, r5, r6, r7, r8, r9, rfinal;
  //  __syncthreads();   
   
  //  for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
  //    unsigned int power_two = i - 1;

  //    if( (ty & power_two) == 0 ) {
  //      weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
  //    }

  //    __syncthreads();
  //  }


  //  float r1 = weight_matrix[ty][tx];
  //  float r2 = weight_matrix[ty+1][tx];

  

  if((ty & 1) ==0){
    r2 = __shfl_down_sync(0xffffffff, r1, 1);
    r3 = r1+r2;
    // weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 1][tx];
    if((ty & 3) ==0){
      r4 = __shfl_down_sync(0xffffffff, r3, 3);
      // weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 2][tx];
      r5 = r3+r4;
      if((ty & 7) ==0){
        r6 = __shfl_down_sync(0xffffffff, r5, 7);
        r7 = r5+r6;
        // weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 4][tx];
        if((ty & 15) ==0){
          r8 = __shfl_down_sync(0xffffffff, r7, 15);
          
          rfinal = r7+r8;
          // weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + 8][tx];
        }else{
          rfinal = r7;
        }
      }else{
        rfinal = r5;
      }
    }else{
      rfinal = r3;
    }
  }else{
    rfinal = r1;
  }
  input_hidden_cuda[index] = rfinal;
  //  input_hidden_cuda[index] = weight_matrix[ty][tx];
  if(ty==0){
    weight_matrix[tx] = rfinal;
  __syncthreads();
  }
   if (tx == 0) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[ty];
   }

}



__global__ void bpnn_adjust_weights_cuda(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));


   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }


}
#endif 
