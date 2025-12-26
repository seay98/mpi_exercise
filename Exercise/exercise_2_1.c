#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <mpi.h>

#include "array.h"
#include "multiply.h"

static unsigned int const seed = 1234;
static int const dimensions[] = {128*1, 128*2, 128*4, 128*8};
static int const n_dimensions = sizeof(dimensions)/sizeof(int);
static double const epsilon = 1e-10;

typedef void (*GEMM)(
    int const m, int const k, int const n,
    double const* const A, double const* const B, double* const C
);

static void populate_compatible_random_matrix_pairs(
    int const m, int const k, int const n,
    int const seed,
    double* const A, double* const B)
{
    set_initilize_rand_seed(seed);

    initialize_2d_double_blocked_rand(A, m, k);
    initialize_2d_double_blocked_rand(B, k, n);
}

static void initialize_problem_matrices(
    int const m, int const k, int const n,
    double** const A, double** const B, double** const C)
{
    *A = allocate_2d_double_blocked(m, k);
    *B = allocate_2d_double_blocked(k, n);
    *C = allocate_2d_double_blocked(m, n);
}

static void destroy_problem_matrices(double** const A, double** const B, double** const C)
{
    *A = free_2d_double_blocked(*A);
    *B = free_2d_double_blocked(*B);
    *C = free_2d_double_blocked(*C);
}

static bool test_muptiply(int const m, int const k, int const n, GEMM gemm, double const epsilon, unsigned int const seed)
{
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    initialize_problem_matrices(m, k, n, &A, &B, &C);
    populate_compatible_random_matrix_pairs(m, k, n, seed, A, B);

    // int my = 2, ny = 2, ky = 2;
    // A = (double[]){1.0, 1.0, 2.0, 2.0};
    // B = (double[]){3.0, 4.0, 5.0, 6.0};
    // C = (double[]){0.0, 0.0, 0.0, 0.0};

    gemm(m, k, n, A, B, C);
    bool result_is_correct = is_product(m, k, n, A, B, C, epsilon);

    destroy_problem_matrices(&A, &B, &C);

    return result_is_correct;
}

// Implement a function "parallel_gemm" of type GEMM, that implements the
// matrix multiplication operation.
//
void parallel_gemm(
  int const m, int const k, int const n,
  double const* const A, double const* const B, double* const C)
{
    int mpi_rank;
    int num_of_ranks;

    int base, rem;
    int* cols = NULL;
    int* col_disp = NULL;
    // MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);

    // broadcast A to all ranks
    MPI_Bcast((void*)A, m*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 computes the column distribution
    if (mpi_rank == 0) {
        base = n / num_of_ranks;
        rem = n % num_of_ranks;
        cols = (int*) calloc(num_of_ranks, sizeof(int));
        col_disp = (int*) calloc(num_of_ranks, sizeof(int));
        for (int i = 0; i < num_of_ranks; ++i) {
            cols[i] = base + (i < rem ? 1 : 0);
            col_disp[i] = (i == 0) ? 0 : col_disp[i - 1] + cols[i - 1];
        }
    }
    // Broadcast column distribution to all ranks
    if (cols == NULL) {
        cols = (int*) calloc(num_of_ranks, sizeof(int));
    }
    if (col_disp == NULL) {
        col_disp = (int*) calloc(num_of_ranks, sizeof(int));
    }
    MPI_Bcast(cols, num_of_ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_disp, num_of_ranks, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare counts and displs for Scatterv and Gatherv
    int* sendcounts = sendcounts = (int*) calloc(num_of_ranks, sizeof(int));
    int* displs = displs = (int*) calloc(num_of_ranks, sizeof(int));
    int* recvcounts = (int*) calloc(num_of_ranks, sizeof(int));
    int* recvdispls = (int*) calloc(num_of_ranks, sizeof(int));
    for (int i = 0; i < num_of_ranks; ++i) {
        sendcounts[i] = cols[i] * k;
        displs[i] = col_disp[i] * k;
        recvcounts[i] = cols[i] * m;
        recvdispls[i] = col_disp[i] * m;
    }

    int n_local_cols = cols[mpi_rank];
    int n_local_elements = n_local_cols * k;

    double* B_local = (double*) calloc(n_local_elements, sizeof(double));
    double* C_local = (double*) calloc(m * n_local_cols, sizeof(double));

    // Scatter B to all ranks
    MPI_Scatterv((void*)B, sendcounts, displs, MPI_DOUBLE, 
                B_local, n_local_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Each rank computes its local C
    multiply_matrices(m, k, n_local_cols, A, B_local, C_local);

    // Gather C from all ranks
    MPI_Gatherv(C_local, m * n_local_cols, MPI_DOUBLE,
                C, recvcounts, recvdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(cols);
    free(col_disp);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    free(B_local);
    free(C_local);
}

//
// Then set "tested_gemm" to the address of your funtion
// GEMM const tested_gemm = &multiply_matrices;
GEMM const tested_gemm = &parallel_gemm;

static bool generate_square_matrix_dimension(int* const m, int* const k, int* const n)
{
    int const max_dim = n_dimensions;
    static int dim = 0;

    if (dim >= max_dim) {
        return false;
    }

    *m = dimensions[dim];
    *k = dimensions[dim];
    *n = dimensions[dim];
    
    dim++;

    return true;
}

int main(int argc, char* argv[])
{
    bool all_test_pass = true;

    int m = 0;
    int k = 0;
    int n = 0;

    int rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    while (generate_square_matrix_dimension(&m, &k, &n)) {
        bool const test_pass = test_muptiply(m, k, n, tested_gemm, epsilon, seed);
        if (rank == 0 && !test_pass) {
                printf("Multiplication failed for: m=%d, k=%d, n=%d\n", m, k, n);
        }
        if (!test_pass) {
            all_test_pass = false;
        }
    }

    MPI_Bcast(&all_test_pass, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    if (!all_test_pass) {
        return EXIT_FAILURE;
    }

    // printf("All tests passed\n");
    return EXIT_SUCCESS;
}
