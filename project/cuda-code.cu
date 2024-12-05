#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

#define ACTIVATION_MAX(a, b) ((a) > (b) ? (a) : (b))

// Define a structure to hold tensor data
typedef struct {
    int rows;
    int cols;
    int channels;
    float *elements;
} Matrix3D;

// Function to allocate memory for tensor on the host
Matrix3D allocate_host_matrix(int rows, int cols, int channels) {
    Matrix3D mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.channels = channels;
    size_t total_size = rows * cols * channels * sizeof(float);
    mat.elements = (float *)malloc(total_size);
    return mat;
}

// Function to allocate memory for tensor on the device
Matrix3D allocate_device_matrix(int rows, int cols, int channels) {
    Matrix3D mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.channels = channels;
    size_t total_size = rows * cols * channels * sizeof(float);
    cudaMalloc(&mat.elements, total_size);
    return mat;
}

// Free host memory
void free_host_matrix(Matrix3D *mat) {
    free(mat->elements);
}

// Free device memory
void free_device_matrix(Matrix3D *mat) {
    cudaFree(mat->elements);
}

// Initialize tensor with random floating point values
double random_double() {
    return ((double)rand() / (double)RAND_MAX) * 2 - 1;
}

void initialize_matrix(Matrix3D *mat) {
    int total_size = mat->rows * mat->cols * mat->channels;
    for (int i = 0; i < total_size; i++) {
        mat->elements[i] = random_double(); // Values between -1 and 1
    }
}

// Kernel for ReLU activation function
__global__ void relu_activation(float *elements, int total_elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_elements) {
        elements[index] = ACTIVATION_MAX(0.0, elements[index]);
    }
}

// Kernel for softmax activation function
__global__ void softmax_activation(float *elements, int num_classes) {
    int tid = threadIdx.x;
    
    __shared__ float total_sum;
    if (tid == 0) total_sum = 0.0f;
    __syncthreads();

    // Compute exponentials
    float value = exp(elements[tid]);
    atomicAdd(&total_sum, value);
    __syncthreads();

    // Normalize values
    elements[tid] = value / total_sum;
}

// Kernel for 2D convolution
__global__ void convolution_2d(float *input, float *filter, float *output, int in_width, int in_height, int in_channels,
                               int filter_width, int filter_height, int filter_channels, int out_width, int out_height, int out_channels) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_channel = blockIdx.z;

    if (out_x < out_width && out_y < out_height && out_channel < out_channels) {
        float accum = 0.0f;
        for (int fy = 0; fy < filter_height; fy++) {
            for (int fx = 0; fx < filter_width; fx++) {
                for (int fc = 0; fc < in_channels; fc++) {
                    int in_x = out_x + fx;
                    int in_y = out_y + fy;
                    int input_idx = (in_y * in_width + in_x) * in_channels + fc;
                    int filter_idx = (fy * filter_width + fx) * filter_channels + fc;
                    accum += input[input_idx] * filter[filter_idx];
                }
            }
        }
        int out_idx = (out_y * out_width + out_x) * out_channels + out_channel;
        output[out_idx] = accum;
    }
}

int main() {
    srand(time(NULL));

    // Create input tensor on host
    Matrix3D host_input = allocate_host_matrix(28, 28, 3); // Input size is now 28x28 with 3 channels
    initialize_matrix(&host_input);

    // Transfer input tensor to device
    Matrix3D device_input = allocate_device_matrix(28, 28, 3);
    cudaMemcpy(device_input.elements, host_input.elements, 28 * 28 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Create convolution filter on host
    Matrix3D filter = allocate_host_matrix(3, 3, 3); // 3x3 filter with 3 channels
    initialize_matrix(&filter);

    // Transfer filter to device
    Matrix3D device_filter = allocate_device_matrix(3, 3, 3);
    cudaMemcpy(device_filter.elements, filter.elements, 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Create output tensor on device
    Matrix3D device_output = allocate_device_matrix(26, 26, 8); // Output size calculated accordingly

    // Timers
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Launch convolution kernel
    dim3 thread_block(16, 16);
    dim3 grid_blocks((26 + thread_block.x - 1) / thread_block.x, (26 + thread_block.y - 1) / thread_block.y, 8);

    cudaEventRecord(start_event);
    convolution_2d<<<grid_blocks, thread_block>>>(device_input.elements, device_filter.elements, device_output.elements,
                                                  28, 28, 3, 3, 3, 3, 26, 26, 8);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float convolution_time;
    cudaEventElapsedTime(&convolution_time, start_event, stop_event);
    printf("[Execution Report] Convolution Layer: %.2f ms\n", convolution_time);

    // Launch ReLU activation
    int output_elements = 26 * 26 * 8;
    cudaEventRecord(start_event);
    relu_activation<<<(output_elements + 255) / 256, 256>>>(device_output.elements, output_elements);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float relu_execution_time;
    cudaEventElapsedTime(&relu_execution_time, start_event, stop_event);
    printf("[Execution Report] ReLU Activation: %.2f ms\n", relu_execution_time);

    // Allocate space for softmax output
    Matrix3D device_softmax_output = allocate_device_matrix(1, 1, 12); // Classifying into 12 classes
    initialize_matrix(&device_softmax_output);

    // Launch Softmax activation
    cudaEventRecord(start_event);
    softmax_activation<<<1, 12>>>(device_softmax_output.elements, 12);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float softmax_execution_time;
    cudaEventElapsedTime(&softmax_execution_time, start_event, stop_event);
    printf("[Execution Report] Softmax Activation: %.2f ms\n", softmax_execution_time);

    // Transfer output back to host for inspection
    Matrix3D host_output = allocate_host_matrix(1, 1, 12);
    cudaMemcpy(host_output.elements, device_softmax_output.elements, 12 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output values
    printf("[Output Probabilities Summary]:\n");
    for (int i = 0; i < 12; i++) {
        printf("-> Class %02d: Probability = %.6f\n", i, host_output.elements[i]);
    }

    // Clean up resources
    free_host_matrix(&host_input);
    free_host_matrix(&filter);
    free_device_matrix(&device_input);
    free_device_matrix(&device_filter);
    free_device_matrix(&device_output);
    free_device_matrix(&device_softmax_output);
    free_host_matrix(&host_output);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
