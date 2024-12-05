#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to run a program and measure its execution time
void run_and_time(const char *program_name) {
    clock_t start = clock();
    
    int ret = system(program_name);

    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (ret == 0) {
        printf("[Report] %s completed successfully in: %.3f seconds\n", program_name, elapsed_time);
    } else {
        printf("[Error] %s failed with return code: %d\n", program_name, ret);
    }
}

int main() {
    printf("=== Program Execution Times Report ===\n");
    printf("----------------------------------------\n");

    // Updated to reflect the correct executable names
    run_and_time("./sequn-code");
    run_and_time("./mpi-code");
    run_and_time("./openmp-code");
    run_and_time("./code-cuda");
    
    printf("----------------------------------------\n");
    printf("Execution Report Complete\n");
    
    return 0;
}
