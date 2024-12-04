                                                                                                   asn1.c                                                                                                                
#include <mpi.h>
#include <stdio.h>
#include <math.h>

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

int main(int argc, char** argv) {
    int id, p_count;
    float lb = 0.0f, ub = 1.0f, step, ls, le, la, total;
    int intervals;
    double t_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p_count);

    if (id == 0) {
        printf("Enter the number of intervals: ");
        scanf("%d", &intervals);
    }

    MPI_Bcast(&intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

    t_start = MPI_Wtime();

    step = (ub - lb) / intervals;
    float range = (ub - lb) / p_count;

    ls = lb + id * range;
    le = ls + range;

    la = cta(ls, le, step);

    MPI_Reduce(&la, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    t_end = MPI_Wtime();

    if (id == 0) {
        printf("Calculated total area: %f\n", total);
        printf("Execution time: %f seconds\n", t_end - t_start);
    }

    MPI_Finalize();
    return 0;
}

