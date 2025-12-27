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
    int mpi_rank, num_of_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);

    int rows_per_rank = m / num_of_ranks;
    int remainder_rows = m % num_of_ranks;
    int cols_per_rank = n / num_of_ranks;
    int remainder_cols = n % num_of_ranks;

    int* cols = (int*) malloc(num_of_ranks * sizeof(int));
    int* col_disp = (int*) malloc(num_of_ranks * sizeof(int));
    for (int i = 0; i < num_of_ranks; i++) {
        cols[i] = cols_per_rank + (i < remainder_cols ? 1 : 0);
        col_disp[i] = (i == 0) ? 0 : col_disp[i-1] + cols[i-1];
    }

    int* sendcounts = (int*) malloc(num_of_ranks * sizeof(int));
    int* displs = (int*) malloc(num_of_ranks * sizeof(int));
    int* recvcounts = (int*) calloc(num_of_ranks, sizeof(int));
    int* recvdispls = (int*) calloc(num_of_ranks, sizeof(int));
    for (int i = 0; i < num_of_ranks; i++) {
        sendcounts[i] = cols[i] * k;
        displs[i] = col_disp[i] * k;
        recvcounts[i] = cols[i] * m;
        recvdispls[i] = col_disp[i] * m;
    }

    int n_local_cols = cols[mpi_rank];
    int n_local_col_elements = n_local_cols * k;

    double* B_local = (double*) calloc(n_local_col_elements, sizeof(double));
    double* C_local = (double*) calloc(m * n_local_cols, sizeof(double));

    // Scatter B to all ranks
    MPI_Scatterv((void*)B, sendcounts, displs, MPI_DOUBLE, 
                B_local, n_local_col_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 broadcast A to all ranks by rows
    // Each rank receives its rows of A and computes its part of C
    for (int i = 0; i < num_of_ranks; ++i) {
        int rows_to_send = rows_per_rank + (i < remainder_rows ? 1 : 0);
        int start_row = i * rows_per_rank + (i < remainder_rows ? i : remainder_rows);
        // Get rows in A which is stored in column-major order
        double* A_rows = (double*) calloc(rows_to_send * k, sizeof(double));
        if (mpi_rank == 0) {
            for (int c = 0; c < k; ++c) {
                for (int r = 0; r < rows_to_send; ++r) {
                    A_rows[r + c * rows_to_send] = A[r + c * m + start_row];
                }
            }
        }
        MPI_Bcast(A_rows, rows_to_send * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Each rank computes its local C for the received rows
        double* C_block = (double*) calloc(rows_to_send * n_local_cols, sizeof(double));
        multiply_matrices(rows_to_send, k, n_local_cols, A_rows, B_local, C_block);

        // Place the computed block into the correct position in C_local
        for (int c = 0; c < n_local_cols; ++c) {
            for (int r = 0; r < rows_to_send; ++r) {
                C_local[r + c * m + start_row] = C_block[r + c * rows_to_send];
            }
        }
        free(C_block);
        free(A_rows);
    }
    
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

    return EXIT_SUCCESS;
}
