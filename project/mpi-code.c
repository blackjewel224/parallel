#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define RELU_FUNC(a, b) ((a) > (b) ? (a) : (b))

// Define a structure for a 3D tensor
typedef struct {
    int cols;
    int rows;
    int channels;
    float *data;
} Matrix3D;

// Function to allocate memory for a tensor
Matrix3D create_tensor(int cols, int rows, int channels) {
    Matrix3D tensor;
    tensor.cols = cols;
    tensor.rows = rows;
    tensor.channels = channels;
    tensor.data = (float *)calloc(cols * rows * channels, sizeof(float));
    return tensor;
}

// Function to free tensor memory
void release_tensor(Matrix3D *tensor) {
    free(tensor->data);
}

// Initialize tensor with random values
void initialize_tensor(Matrix3D *tensor) {
    int size = tensor->cols * tensor->rows * tensor->channels;
    for (int i = 0; i < size; i++) {
        tensor->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2; // Values between -1 and 1
    }
}

// ReLU activation function
void relu_activation(Matrix3D *tensor) {
    int size = tensor->cols * tensor->rows * tensor->channels;
    for (int i = 0; i < size; i++) {
        tensor->data[i] = RELU_FUNC(0, tensor->data[i]);
    }
}

// Softmax activation function
void softmax_activation(Matrix3D *tensor) {
    float sum = 0.0;
    for (int i = 0; i < tensor->channels; i++) {
        tensor->data[i] = exp(tensor->data[i]);
        sum += tensor->data[i];
    }
    for (int i = 0; i < tensor->channels; i++) {
        tensor->data[i] /= sum;
    }
}

// Distributed convolution operation
void parallel_convolution(Matrix3D *input, Matrix3D *filter, Matrix3D *output, int rank, int num_procs) {
    int start_channel = rank * (filter->channels / num_procs);
    int end_channel = (rank == num_procs - 1) ? filter->channels : start_channel + (filter->channels / num_procs);

    for (int c = start_channel; c < end_channel; c++) {
        for (int y = 0; y < output->rows; y++) {
            for (int x = 0; x < output->cols; x++) {
                float acc = 0.0;
                for (int fy = 0; fy < filter->rows; fy++) {
                    for (int fx = 0; fx < filter->cols; fx++) {
                        for (int fc = 0; fc < input->channels; fc++) {
                            int in_x = x + fx;
                            int in_y = y + fy;
                            int input_idx = (in_y * input->cols + in_x) * input->channels + fc;
                            int filter_idx = (fy * filter->cols + fx) * filter->channels + fc;
                            acc += input->data[input_idx] * filter->data[filter_idx];
                        }
                    }
                }
                int output_idx = (y * output->cols + x) * output->channels + c;
                output->data[output_idx] = acc;
            }
        }
    }
}

// Distributed fully connected layer
void parallel_fully_connected(Matrix3D *input, Matrix3D *weights, Matrix3D *bias, Matrix3D *output, int rank, int num_procs) {
    int start_channel = rank * (output->channels / num_procs);
    int end_channel = (rank == num_procs - 1) ? output->channels : start_channel + (output->channels / num_procs);

    for (int c = start_channel; c < end_channel; c++) {
        float sum = bias->data[c];
        for (int j = 0; j < input->cols * input->rows * input->channels; j++) {
            sum += input->data[j] * weights->data[c * (input->cols * input->rows * input->channels) + j];
        }
        output->data[c] = RELU_FUNC(0, sum); // Apply ReLU activation
    }
}

int main(int argc, char **argv) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    srand(time(NULL) + rank); // Initialize random seed differently for each process

    Matrix3D input_tensor, conv_kernel, conv_output;
    if (rank == 0) {
        // Create input tensor (grayscale 30x30 image)
        input_tensor = create_tensor(30, 30, 1);
        for (int i = 0; i < 30 * 30; i++) {
            input_tensor.data[i] = rand() % 256 / 255.0; // Normalized random values
        }

        // Create convolution kernel
        conv_kernel = create_tensor(3, 3, 1);
        initialize_tensor(&conv_kernel);

        // Create output tensor
        conv_output = create_tensor(28, 28, 8);
    }

    // Broadcast tensors to all processes
    MPI_Bcast(input_tensor.data, 30 * 30 * 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv_kernel.data, 3 * 3 * 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Perform convolution in parallel
    parallel_convolution(&input_tensor, &conv_kernel, &conv_output, rank, num_procs);

    // Gather results from all processes
    MPI_Gather(conv_output.data, conv_output.cols * conv_output.rows * (8 / num_procs), MPI_FLOAT,
               conv_output.data, conv_output.cols * conv_output.rows * (8 / num_procs), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Apply ReLU and Softmax activations
        relu_activation(&conv_output);
        softmax_activation(&conv_output);

        // Print output probabilities
        printf("=== Predicted Class Probabilities ===\n");
        for (int i = 0; i < conv_output.channels; i++) {
            printf("Class [%02d]: %.6f\n", i, conv_output.data[i]);
        }

        // Free memory
        release_tensor(&input_tensor);
        release_tensor(&conv_kernel);
        release_tensor(&conv_output);
    }

    MPI_Finalize();
    return 0;
}
