#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
float func(float x) {
    return x * x;
}

float cta(float start, float end, float step) {
    float area = 0;
    for (float p = start; p < end; p += step) {
        area += func(p) + func(p + step);
    }
    return (area * step) / 2.0f;
}
float trapezoid_area(float start, float end, int intervals) {
    float width = (end - start) / intervals;
    float total = 0.0f;

    for (int i = 0; i < intervals; i++) {
        float x1 = start + i * width;
        float x2 = x1 + width;
        total += (x1 * x1 + x2 * x2) / 2.0f * width; // Assuming y = x^2
    }

    return total;
}

int main() {
    float start = 0.0f, end = 1.0f, total;
    int intervals;
    printf("Enter number of intervals: ");
    scanf("%d", &intervals);

    clock_t t_start = clock();
    total = trapezoid_area(start, end, intervals);
    clock_t t_end = clock();

    double time_taken = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Total area: %f\n", total);
    printf("Execution time: %f seconds\n", time_taken);

    return 0;
}

