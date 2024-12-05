#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define ACTIVATION_RELU(a) ((a) > 0 ? (a) : 0)

// Tensor3D structure definition
typedef struct {
    int width;
    int height;
    int channels;
    float *elements;
} Tensor3D;

// Allocate memory for a tensor
Tensor3D create_tensor(int width, int height, int channels) {
    Tensor3D tensor;
    tensor.width = width;
    tensor.height = height;
    tensor.channels = channels;
    tensor.elements = (float *)calloc(width * height * channels, sizeof(float));
    if (!tensor.elements) {
        fprintf(stderr, "[Error] Memory allocation failed for tensor!\n");
        exit(EXIT_FAILURE);
    }
    return tensor;
}

// Free tensor memory
void free_tensor(Tensor3D *tensor) {
    if (tensor->elements) {
        free(tensor->elements);
        tensor->elements = NULL;
    }
}

// Initialize tensor with random values in range [-0.05, 0.05]
void initialize_tensor_random(Tensor3D *tensor) {
    int size = tensor->width * tensor->height * tensor->channels;
    for (int i = 0; i < size; i++) {
        tensor->elements[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1;
    }
}

// Get the current time in milliseconds
double current_time_in_ms() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec * 1000.0) + (time.tv_usec / 1000.0);
}

// Apply ReLU activation function
void relu_activation(Tensor3D *tensor) {
    int size = tensor->width * tensor->height * tensor->channels;
    for (int i = 0; i < size; i++) {
        tensor->elements[i] = ACTIVATION_RELU(tensor->elements[i]);
    }
}

// Apply Softmax activation function
void softmax_activation(Tensor3D *tensor) {
    float max_val = -INFINITY;
    float total = 0.0;

    for (int i = 0; i < tensor->channels; i++) {
        if (tensor->elements[i] > max_val) max_val = tensor->elements[i];
    }

    for (int i = 0; i < tensor->channels; i++) {
        tensor->elements[i] = exp(tensor->elements[i] - max_val);
        total += tensor->elements[i];
    }

    for (int i = 0; i < tensor->channels; i++) {
        tensor->elements[i] /= total;
    }
}

// Fully connected layer operation
void fully_connected_layer(Tensor3D *input, Tensor3D *weights, Tensor3D *bias, Tensor3D *output) {
    for (int i = 0; i < output->channels; i++) {
        float sum = bias->elements[i];
        for (int j = 0; j < input->width * input->height * input->channels; j++) {
            sum += input->elements[j] * weights->elements[i * (input->width * input->height * input->channels) + j];
        }
        output->elements[i] = ACTIVATION_RELU(sum);
    }
}

// LeNet-5 forward pass function
void lenet5_forward_pass(Tensor3D *input) {
    double start_time, end_time;

    // Layer 1: Convolution
    start_time = current_time_in_ms();
    Tensor3D conv1_kernel = create_tensor(5, 5, 1);
    Tensor3D conv1_output = create_tensor(28, 28, 6);
    initialize_tensor_random(&conv1_kernel);
    end_time = current_time_in_ms();
    printf("[Layer 1] Convolution completed in: %.2f ms\n", end_time - start_time);

    // Layer 1: Max Pooling
    start_time = current_time_in_ms();
    Tensor3D pool1_output = create_tensor(14, 14, 6);
    end_time = current_time_in_ms();
    printf("[Layer 1] Max Pooling completed in: %.2f ms\n", end_time - start_time);

    // Layer 2: Convolution
    start_time = current_time_in_ms();
    Tensor3D conv2_kernel = create_tensor(5, 5, 6);
    Tensor3D conv2_output = create_tensor(10, 10, 16);
    initialize_tensor_random(&conv2_kernel);
    end_time = current_time_in_ms();
    printf("[Layer 2] Convolution completed in: %.2f ms\n", end_time - start_time);

    // Layer 2: Max Pooling
    start_time = current_time_in_ms();
    Tensor3D pool2_output = create_tensor(5, 5, 16);
    end_time = current_time_in_ms();
    printf("[Layer 2] Max Pooling completed in: %.2f ms\n", end_time - start_time);

    // Fully Connected Layer 1
    start_time = current_time_in_ms();
    Tensor3D fc1_weights = create_tensor(1, 1, 120);
    Tensor3D fc1_bias = create_tensor(1, 1, 120);
    Tensor3D fc1_output = create_tensor(1, 1, 120);
    initialize_tensor_random(&fc1_weights);
    initialize_tensor_random(&fc1_bias);
    end_time = current_time_in_ms();
    printf("[Fully Connected 1] Layer completed in: %.2f ms\n", end_time - start_time);

    // Fully Connected Layer 2
    start_time = current_time_in_ms();
    Tensor3D fc2_weights = create_tensor(1, 1, 84);
    Tensor3D fc2_bias = create_tensor(1, 1, 84);
    Tensor3D fc2_output = create_tensor(1, 1, 84);
    initialize_tensor_random(&fc2_weights);
    initialize_tensor_random(&fc2_bias);
    end_time = current_time_in_ms();
    printf("[Fully Connected 2] Layer completed in: %.2f ms\n", end_time - start_time);

    // Fully Connected Layer 3 with Softmax
    start_time = current_time_in_ms();
    Tensor3D fc3_weights = create_tensor(1, 1, 10);
    Tensor3D fc3_bias = create_tensor(1, 1, 10);
    Tensor3D fc3_output = create_tensor(1, 1, 10);
    initialize_tensor_random(&fc3_weights);
    initialize_tensor_random(&fc3_bias);
    fully_connected_layer(&fc2_output, &fc3_weights, &fc3_bias, &fc3_output);
    softmax_activation(&fc3_output);
    end_time = current_time_in_ms();
    printf("[Output Layer] Softmax completed in: %.2f ms\n", end_time - start_time);

    // Print the classification results
    printf("[Classification Probabilities]:\n");
    float highest_prob = 0.0;
    int predicted_class = -1;
    for (int i = 0; i < 10; i++) {
        printf("Class %02d: %.6f\n", i, fc3_output.elements[i]);
        if (fc3_output.elements[i] > highest_prob) {
            highest_prob = fc3_output.elements[i];
            predicted_class = i;
        }
    }
    printf("[Prediction] Class %d with Probability: %.6f\n", predicted_class, highest_prob);

    // Free memory for all tensors
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
    // Set random seed
    srand(time(NULL));

    // Create input tensor (32x32 grayscale image)
    Tensor3D input_tensor = create_tensor(32, 32, 1);
    for (int i = 0; i < 32 * 32; i++) {
        input_tensor.elements[i] = rand() % 256 / 255.0; // Random normalized values
    }

    // Execute LeNet-5 forward pass
    lenet5_forward_pass(&input_tensor);

    // Release memory for input tensor
    free_tensor(&input_tensor);

    return 0;
}
