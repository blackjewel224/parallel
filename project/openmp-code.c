#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define RELU_THRESHOLD(a, b) ((a) > (b) ? (a) : (b))

// Structure for a 3D tensor
typedef struct {
    int width;
    int height;
    int channels;
    float *values;
} Tensor3D;

// Allocate memory for a tensor
Tensor3D allocate_tensor(int width, int height, int channels) {
    Tensor3D tensor;
    tensor.width = width;
    tensor.height = height;
    tensor.channels = channels;
    tensor.values = (float *)calloc(width * height * channels, sizeof(float));
    return tensor;
}

// Release tensor memory
void free_tensor(Tensor3D *tensor) {
    free(tensor->values);
}

// Initialize tensor with random values
void initialize_tensor_random(Tensor3D *tensor) {
    int size = tensor->width * tensor->height * tensor->channels;
    for (int i = 0; i < size; i++) {
        tensor->values[i] = ((float)rand() / RAND_MAX - 0.5f) * 2; // Random values between -1 and 1
    }
}

// ReLU activation function
void relu_activation(Tensor3D *tensor) {
    int size = tensor->width * tensor->height * tensor->channels;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        tensor->values[i] = RELU_THRESHOLD(0, tensor->values[i]);
    }
}

// Softmax activation function
void softmax_activation(Tensor3D *tensor) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < tensor->channels; i++) {
        tensor->values[i] = exp(tensor->values[i]);
        sum += tensor->values[i];
    }
    #pragma omp parallel for
    for (int i = 0; i < tensor->channels; i++) {
        tensor->values[i] /= sum;
    }
}

// Convolution operation
void perform_convolution(Tensor3D *input, Tensor3D *kernel, Tensor3D *output) {
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < kernel->channels; c++) {
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                float total = 0.0;
                for (int ky = 0; ky < kernel->height; ky++) {
                    for (int kx = 0; kx < kernel->width; kx++) {
                        for (int kc = 0; kc < input->channels; kc++) {
                            int in_x = x + kx;
                            int in_y = y + ky;
                            int input_idx = (in_y * input->width + in_x) * input->channels + kc;
                            int kernel_idx = (ky * kernel->width + kx) * kernel->channels + kc;
                            total += input->values[input_idx] * kernel->values[kernel_idx];
                        }
                    }
                }
                int output_idx = (y * output->width + x) * output->channels + c;
                output->values[output_idx] = total;
            }
        }
    }
}

// Max pooling operation
void perform_max_pooling(Tensor3D *input, Tensor3D *output, int pool_size) {
    #pragma omp parallel for collapse(3)
    for (int ch = 0; ch < input->channels; ch++) {
        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                float max_val = -INFINITY;
                for (int py = 0; py < pool_size; py++) {
                    for (int px = 0; px < pool_size; px++) {
                        int in_x = x * pool_size + px;
                        int in_y = y * pool_size + py;
                        int input_idx = (in_y * input->width + in_x) * input->channels + ch;
                        max_val = RELU_THRESHOLD(max_val, input->values[input_idx]);
                    }
                }
                int output_idx = (y * output->width + x) * output->channels + ch;
                output->values[output_idx] = max_val;
            }
        }
    }
}

// Fully connected layer operation
void fully_connected_layer(Tensor3D *input, Tensor3D *weights, Tensor3D *bias, Tensor3D *output) {
    #pragma omp parallel for
    for (int i = 0; i < output->channels; i++) {
        float sum = bias->values[i];
        for (int j = 0; j < input->width * input->height * input->channels; j++) {
            sum += input->values[j] * weights->values[i * (input->width * input->height * input->channels) + j];
        }
        output->values[i] = RELU_THRESHOLD(0, sum);
    }
}

// LeNet-5 forward pass
void lenet5_forward(Tensor3D *input) {
    double start, end;

    // Layer 1: Convolution
    start = omp_get_wtime();
    Tensor3D conv1_kernel = allocate_tensor(5, 5, 1);
    Tensor3D conv1_output = allocate_tensor(28, 28, 6);
    initialize_tensor_random(&conv1_kernel);
    perform_convolution(input, &conv1_kernel, &conv1_output);
    relu_activation(&conv1_output);
    end = omp_get_wtime();
    printf("[Layer 1] Convolution Time: %.2f ms\n", (end - start) * 1000);

    // Layer 1: Max Pooling
    start = omp_get_wtime();
    Tensor3D pool1_output = allocate_tensor(14, 14, 6);
    perform_max_pooling(&conv1_output, &pool1_output, 2);
    end = omp_get_wtime();
    printf("[Layer 1] Max Pooling Time: %.2f ms\n", (end - start) * 1000);

    // Layer 2: Convolution
    start = omp_get_wtime();
    Tensor3D conv2_kernel = allocate_tensor(5, 5, 6);
    Tensor3D conv2_output = allocate_tensor(10, 10, 16);
    initialize_tensor_random(&conv2_kernel);
    perform_convolution(&pool1_output, &conv2_kernel, &conv2_output);
    relu_activation(&conv2_output);
    end = omp_get_wtime();
    printf("[Layer 2] Convolution Time: %.2f ms\n", (end - start) * 1000);

    // Layer 2: Max Pooling
    start = omp_get_wtime();
    Tensor3D pool2_output = allocate_tensor(5, 5, 16);
    perform_max_pooling(&conv2_output, &pool2_output, 2);
    end = omp_get_wtime();
    printf("[Layer 2] Max Pooling Time: %.2f ms\n", (end - start) * 1000);

    // Fully Connected Layers
    start = omp_get_wtime();
    Tensor3D fc1_weights = allocate_tensor(1, 1, 120);
    Tensor3D fc1_bias = allocate_tensor(1, 1, 120);
    Tensor3D fc1_output = allocate_tensor(1, 1, 120);
    initialize_tensor_random(&fc1_weights);
    initialize_tensor_random(&fc1_bias);
    fully_connected_layer(&pool2_output, &fc1_weights, &fc1_bias, &fc1_output);
    end = omp_get_wtime();
    printf("[Fully Connected 1] Layer Time: %.2f ms\n", (end - start) * 1000);

    start = omp_get_wtime();
    Tensor3D fc2_weights = allocate_tensor(1, 1, 84);
    Tensor3D fc2_bias = allocate_tensor(1, 1, 84);
    Tensor3D fc2_output = allocate_tensor(1, 1, 84);
    initialize_tensor_random(&fc2_weights);
    initialize_tensor_random(&fc2_bias);
    fully_connected_layer(&fc1_output, &fc2_weights, &fc2_bias, &fc2_output);
    end = omp_get_wtime();
    printf("[Fully Connected 2] Layer Time: %.2f ms\n", (end - start) * 1000);

    start = omp_get_wtime();
    Tensor3D fc3_weights = allocate_tensor(1, 1, 10);
    Tensor3D fc3_bias = allocate_tensor(1, 1, 10);
    Tensor3D fc3_output = allocate_tensor(1, 1, 10);
    initialize_tensor_random(&fc3_weights);
    initialize_tensor_random(&fc3_bias);
    fully_connected_layer(&fc2_output, &fc3_weights, &fc3_bias, &fc3_output);
    softmax_activation(&fc3_output);
    end = omp_get_wtime();
    printf("[Output Layer] Softmax Time: %.2f ms\n", (end - start) * 1000);

    // Print classification results
    printf("[Predicted Classification Probabilities]:\n");
    for (int i = 0; i < 10; i++) {
        printf("Class %02d: Probability = %.6f\n", i, fc3_output.values[i]);
    }

    // Release memory
    free_tensor(&conv1_kernel);
    free_tensor(&conv1_output);
    free_tensor(&pool1_output);
    free_tensor(&conv2_kernel);
    free_tensor(&conv2_output);
    free_tensor(&pool2_output);
    free_tensor(&fc1_weights);
    free_tensor(&fc1_bias);
    free_tensor(&fc1_output);
    free_tensor(&fc2_weights);
    free_tensor(&fc2_bias);
    free_tensor(&fc2_output);
    free_tensor(&fc3_weights);
    free_tensor(&fc3_bias);
    free_tensor(&fc3_output);
}

int main() {
    // Set number of threads for OpenMP
    omp_set_num_threads(omp_get_max_threads());

    // Seed random number generator
    srand(time(NULL));

    // Create input tensor (32x32 grayscale image)
    Tensor3D input_tensor = allocate_tensor(32, 32, 1);
    for (int i = 0; i < 32 * 32; i++) {
        input_tensor.values[i] = rand() % 256 / 255.0; // Normalized random values
    }

    // Execute LeNet-5 forward pass
    lenet5_forward(&input_tensor);

    // Release memory for input tensor
    free_tensor(&input_tensor);

    return 0;
}
