/* 
 * Sequential implementation of the Block-Lanczos algorithm.
 *
 * This is based on the paper: 
 *     "A modified block Lanczos algorithm with fewer vectors" 
 *
 *  by Emmanuel Thomé, available online at 
 *      https://hal.inria.fr/hal-01293351/document
 *
 * Authors : Charles Bouillaguet
 *
 * v1.00 (2022-01-18)
 * v1.01 (2022-03-13) bugfix with (non-transposed) matrices that have more columns than rows
 *
 * USAGE: 
 *      $ ./lanczos_modp --prime 65537 --n 4 --matrix random_small.mtx
 *
 */
#define _POSIX_C_SOURCE  1  // ctime
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>

#include "mmio.h"
#include <mpi.h>

typedef uint64_t u64;
typedef uint32_t u32;

/******************* global variables ********************/

long n = 1;
u64 prime;
char *matrix_filename;
char *kernel_filename;
bool right_kernel = false;
int stop_after = -1;

int n_iterations;      /* variables of the "verbosity engine" */
double start;
double last_print;
bool ETA_flag;
int expected_iterations;

/******************* sparse matrix data structure **************/

struct sparsematrix_t {
        int nrows;        // dimensions
        int ncols;
        long int nnz;     // number of non-zero coefficients
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

/******************* pseudo-random generator (xoshiro256+) ********************/

/* fixed seed --- this is bad */
u64 rng_state[4] = {0x1415926535, 0x8979323846, 0x2643383279, 0x5028841971};

u64 rotl(u64 x, int k)
{
        u64 foo = x << k;
        u64 bar = x >> (64 - k);
        return foo ^ bar;
}

u64 random64()
{
        u64 result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
        u64 t = rng_state[1] << 17;
        rng_state[2] ^= rng_state[0];
        rng_state[3] ^= rng_state[1];
        rng_state[1] ^= rng_state[2];
        rng_state[0] ^= rng_state[3];
        rng_state[2] ^= t;
        rng_state[3] = rotl(rng_state[3], 45);
        return result;
}

/******************* utility functions ********************/

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

/* represent n in <= 6 char  */
void human_format(char * target, long n) {
        if (n < 1000) {
                sprintf(target, "%" PRId64, n);
                return;
        }
        if (n < 1000000) {
                sprintf(target, "%.1fK", n / 1e3);
                return;
        }
        if (n < 1000000000) {
                sprintf(target, "%.1fM", n / 1e6);
                return;
        }
        if (n < 1000000000000ll) {
                sprintf(target, "%.1fG", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll) {
                sprintf(target, "%.1fT", n / 1e12);
                return;
        }
}

/************************** command-line options ****************************/

void usage(char ** argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--matrix FILENAME           MatrixMarket file containing the spasre matrix\n");
        printf("--prime P                   compute modulo P\n");
        printf("--n N                       blocking factor [default 1]\n");
        printf("--output-file FILENAME      store the block of kernel vectors\n");
        printf("--right                     compute right kernel vectors\n");
        printf("--left                      compute left kernel vectors [default]\n");
        printf("--stop-after N              stop the algorithm after N iterations\n");
        printf("\n");
        printf("The --matrix and --prime arguments are required\n");
        printf("The --stop-after and --output-file arguments mutually exclusive\n");
        exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
        struct option longopts[8] = {
                {"matrix", required_argument, NULL, 'm'},
                {"prime", required_argument, NULL, 'p'},
                {"n", required_argument, NULL, 'n'},
                {"output-file", required_argument, NULL, 'o'},
                {"right", no_argument, NULL, 'r'},
                {"left", no_argument, NULL, 'l'},
                {"stop-after", required_argument, NULL, 's'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'n':
                        n = atoi(optarg);
                        break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'o':
                        kernel_filename = optarg;
                        break;
                case 'r':
                        right_kernel = true;
                        break;
                case 'l':
                        right_kernel = false;
                        break;
                case 's':
                        stop_after = atoll(optarg);
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }

        /* missing required args? */
        if (matrix_filename == NULL || prime == 0)
                usage(argv);
        /* exclusive arguments? */
        if (kernel_filename != NULL && stop_after > 0)
                usage(argv);
        /* range checking */
        if (prime > 0x3fffffdd) {
                errx(1, "p is capped at 2**30 - 35.  Slighly larger values could work, with the\n");
                printf("suitable code modifications.\n");
                exit(1);
        }
}

/****************** sparse matrix operations ******************/

/* Load a matrix from a file in "list of triplet" representation */
void sparsematrix_mm_load(struct sparsematrix_t * M, char const * filename)
{
        int nrows = 0;
        int ncols = 0;
        long nnz = 0;

        printf("Loading matrix from %s\n", filename);
        fflush(stdout);

        FILE *f = fopen(filename, "r");
        if (f == NULL)
                err(1, "impossible d'ouvrir %s", filename);

        /* read the header, check format */
        MM_typecode matcode;
        if (mm_read_banner(f, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", 
                        mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", 
                        mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
        fprintf(stderr, "  - Allocating %.1f MByte\n", 1e-6 * (12.0 * nnz));

        /* Allocate memory for the matrix */
        int *Mi = malloc(nnz * sizeof(*Mi));
        int *Mj = malloc(nnz * sizeof(*Mj));
        u32 *Mx = malloc(nnz * sizeof(*Mx));
        if (Mi == NULL || Mj == NULL || Mx == NULL)
                err(1, "Cannot allocate sparse matrix");

        /* Parse and load actual entries */
        double start = wtime();
        for (long u = 0; u < nnz; u++) {
                int i, j;
                u32 x;
                if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                        errx(1, "parse error entry %ld\n", u);
                Mi[u] = i - 1;  /* MatrixMarket is 1-based */
                Mj[u] = j - 1;
                Mx[u] = x % prime;
                
                // verbosity
                if ((u & 0xffff) == 0xffff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / nnz;
                        double rate = ftell(f) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }

        /* finalization */
        fclose(f);
        printf("\n");
        M->nrows = nrows;
        M->ncols = ncols;
        M->nnz = nnz;
        M->i = Mi;
        M->j = Mj;
        M->x = Mx;
}

/* y += M*x or y += transpose(M)*x, according to the transpose flag */ 
void sparse_matrix_vector_product(u32 * y, struct sparsematrix_t const * M, u32 const * x, bool transpose)
{
	long nnz = M->nnz;
	int nrows = transpose ? M->ncols : M->nrows;
	int const * Mi = M->i;
	int const * Mj = M->j;
	u32 const * Mx = M->x;
	
	for (long i = 0; i < nrows * n; i++)
	{
		y[i] = 0;
	}
			
	for (long k = 0; k < nnz; k++)
	{
		int i = transpose ? Mj[k] : Mi[k];
		int j = transpose ? Mi[k] : Mj[k];
		u64 v = Mx[k];

		for (int l = 0; l < n; l++)
		{
			u64 a = y[i * n + l];
			u64 b = x[j * n + l];
			y[i * n + l] = (a + v * b) % prime;
		}
	}
}

/****************** dense linear algebra modulo p *************************/ 

/* C += A*B   for n x n matrices */
void matmul_CpAB(u32 * C, u32 const * A, u32 const * B)
{
	u32 Bt[n * n];

	for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            Bt[j + i*n] = B[i + j*n];
        }
    }
    
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			u64 x = C[i * n + j];
			
			for (int k = 0; k < n; k++)
			{	
				u64 y = A[i * n + k];
				u64 z = Bt[j * n + k];
				x += y * z;
			}

			C[i * n + j] = x % prime;
		}
	}
}

/* C += transpose(A)*B   for n x n matrices */
void matmul_CpAtB(u32 * C, u32 const * A, u32 const * B)
{
	u32 At[n * n];
	u32 Bt[n * n];

	for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            At[j + i*n] = A[i + j*n];
			Bt[j + i*n] = B[i + j*n];
        }
    }

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			u64 x = C[i * n + j];

			for (int k = 0; k < n; k++)
			{
				u64 y = At[i * n + k];
				u64 z = Bt[j * n + k];
				x += y * z;
			}

			C[i * n + j] = x % prime;
		}
	}	
}

/* return a^(-1) mod b */
u32 invmod(u32 a, u32 b)
{
        long int t = 0;  
        long int nt = 1;  
        long int r = b;  
        long int nr = a % b;
        while (nr != 0) {
                long int q = r / nr;
                long int tmp = nt;  
                nt = t - q*nt;  
                t = tmp;
                tmp = nr;  
                nr = r - q*nr;  
                r = tmp;
        }
        if (t < 0)
                t += b;
        return (u32) t;
}

/* 
 * Given an n x n matrix U, compute a "partial-inverse" V and a diagonal matrix
 * d such that d*V == V*d == V and d == V*U*d. Return the number of pivots.
 */ 
int semi_inverse(u32 const * M_, u32 * winv, u32 * d)
{
        u32 M[n * n];
        int npiv = 0;
        for (int i = 0; i < n * n; i++)   /* copy M <--- M_ */
                M[i] = M_[i];
        /* phase 1: compute d */
        for (int i = 0; i < n; i++)       /* setup d */
                d[i] = 0;
        for (int j = 0; j < n; j++) {     /* search a pivot on column j */
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;         /* no pivot found */
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);  /* multiply pivot row to make pivot == 1 */
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {   /* swap pivot row with row j */
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {  /* eliminate everything else on column j */
                        if (i == j)
                                continue;
                        u64 multiplier = M[i*n+j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;  
                        }
                }
        }
        /* phase 2: compute d and winv */
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        M[i*n + j] = (d[i] && d[j]) ? M_[i*n + j] : 0;
                        winv[i*n + j] = ((i == j) && d[i]) ? 1 : 0;
                }
        npiv = 0;
        for (int i = 0; i < n; i++)
                d[i] = 0;
        /* same dance */
        for (int j = 0; j < n; j++) { 
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i*n + j] != 0) {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot*n + j], prime);
                for (int k = 0; k < n; k++) {
                        u64 x = M[pivot*n + k];
                        M[pivot*n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = M[j*n + k];
                        M[j*n + k] = M[pivot*n + k];
                        M[pivot*n + k] = tmp;
                }
                for (int k = 0; k < n; k++) {
                        u64 x = winv[pivot * n + k];
                        winv[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++) {
                        u32 tmp = winv[j * n + k];
                        winv[j * n + k] = winv[pivot * n + k];
                        winv[pivot * n + k] = tmp;
                }
                for (int i = 0; i < n; i++) {
                        if (i == j)
                                continue;
                        u64 multiplier = M[i * n + j];
                        for (int k = 0; k < n; k++) {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;
                                u64 w = winv[i * n + k];
                                u64 z = winv[j * n + k];
                                winv[i * n + k] = (w + (prime - multiplier) * z) % prime;  
                        }
                }
        }
        return npiv;
}

/****************** global variables for MPI implementation ******************/

int p; //total number of processes 
int myRank; //rank in MPI_COMM_WORLD 
int myGridRank; //rank in gridComm
int myRowRank; //rank in rowComm
int myColRank; //rank in colComm
int myGridCoord[2]; //(x,y) coordinates in the grid topology
int dims[2]; //dimensions of the virtual grid
int notWrapping[2] = {0,0}; //flags to turn off wrapping in grid
MPI_Comm gridComm; //grid communicator
MPI_Comm rowComm; //row subset communicator
MPI_Comm colComm; //column subset communicator


/****************** MPI functions ******************/

/* Initialize MPI, the virtual grid topology and the matrices */
void mpi_init(struct sparsematrix_t *M, struct sparsematrix_t *m)
{
    //init mpi env
    MPI_Init(NULL,NULL);

    //get total number of processes 
	MPI_Comm_size(MPI_COMM_WORLD,&p);

    //get the rank of each process 
	MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

    //get the optimal dimensions based on the number of processes for the virtual grid 
    MPI_Dims_create(p, 2, dims);

    //generate grid communicator using mpi suggested dimensions 
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, notWrapping, 1, &gridComm);

    //get the process rank in the grid
    MPI_Comm_rank(gridComm,&myGridRank);

    //get the process coordinates in the grid
    MPI_Cart_coords(gridComm,myGridRank,2,myGridCoord);

    //generate row communicator based on the grid coordinates
    MPI_Comm_split(gridComm,myGridCoord[0],myGridCoord[1],&rowComm);
    MPI_Comm_rank(rowComm,&myRowRank);

    //generate column communicator based on the grid coordinates
    MPI_Comm_split(gridComm,myGridCoord[1],myGridCoord[0],&colComm);
    MPI_Comm_rank(colComm,&myColRank);

    //set matrices
    M->i = NULL;
    M->j = NULL;
    M->x = NULL;
    M->nnz = 0;
    M->ncols = 0;
    M->nrows = 0;

    m->i = NULL;
    m->j = NULL;
    m->x = NULL;
    m->nnz = 0;
    m->ncols = 0;
    m->nrows = 0;
}

/* Load the matrix from file and broadcast its size to all processes */
void mpi_get_matrix_size(struct sparsematrix_t *M,char const * filename)
{
    //the process with the rank 0 in the grid load the matrix
    if(myGridRank == 0)
    {
	    sparsematrix_mm_load(M,filename);
    }

    //wait until the process 0 loads the matrix
    MPI_Barrier(gridComm);

    //broadcast the size of the loaded matrix 
    MPI_Bcast(&(M->nrows),1,MPI_INT,0,gridComm);
    MPI_Bcast(&(M->ncols),1,MPI_INT,0,gridComm);

    //check if the size of the matrix is enough for all processes
    if(M->nrows < dims[0] || M->ncols < dims[1])
    {
        printf("Matrix size to small : why using MPI ?\n");
        MPI_Abort(MPI_COMM_WORLD,MPI_ERR_IO);
    }
}

/* Compute the size of the sub block m based on the size of M and the grid */
void mpi_get_block_size(const struct sparsematrix_t *M, struct sparsematrix_t *m)
{
    //to store
    int div = M->nrows / dims[0];
    int mod = M->nrows % dims[0];

    /** row **/
    if(mod == 0) //even
    {
        m->nrows = div;
    }

    else //not even
    {
        m->nrows = (myGridCoord[0] == 0) ? div + mod : div; //row 0 takes the excess
    }

    /** column **/
    div = M->ncols / dims[1];
    mod = M->ncols % dims[1];

    if(mod == 0) //even
    {
        m->ncols = div;
    }

    else //not even
    {
        m->ncols = (myGridCoord[1] == 0) ? div + mod : div; //column 0 takes the excess
    } 
}

/* Alloc memory for the sub block m and get its value */
void mpi_create_matrix_block(const struct sparsematrix_t *M, struct sparsematrix_t *m)
{
    mpi_get_block_size(M,m); //get sub block size

    int coord[2] = {0,0}; //to store the grid coord of the current process
    int dest_rank; //destination rank in the grid
    long int nnz; //number of non zero
    MPI_Status status; //result of MPI_Recv call
 
    //get the slicing for each coord
    int div_x = M->nrows / dims[0];
    int mod_x = M->nrows % dims[0];
    int div_y = M->ncols / dims[1];
    int mod_y = M->ncols % dims[1];

    //max matrix column based on the current column of the grid
    int max_col;

    //keep in track the pointers distance
    int low_p = 0;
    int high_p = 0;

    /* per grid column */ 
    for(int j = 0; j < dims[1]; j++)
    {
        coord[1] = j; //save y coord 
        max_col = (div_y + mod_y) + div_y * j - 1; //get max size of the matrix column based on the grid topology
        
        //process 0 gets values from matrix M based on the current grid column
        if(myGridRank == 0)
        {
            low_p = (high_p != 0) ? high_p + 1 : high_p; //set the low distance pointer
            
            //case 1 : next iteration is not the end of the matrix M
            if(j + 1 != dims[1])
            {
                //set the high distance pointer
                for(int k = high_p; k < M->nnz; k++)
                { 
                    if(M->j[k] > max_col)
                    {
                        high_p = k - 1;
                        break;
                    }                
                } 
            }

            //case 2 : the next iteration is the end of the matrix M
            else
            {
                high_p = M->nnz - 1;
            }
        }

        //interval for matrix row value corresponding to the grid row 
        int low;
        int high = 0;

        /* per grid row */ 
        for(int i = 0; i < dims[0]; i++)
        {
            coord[0] = i; //save x coord 
            MPI_Cart_rank(gridComm,coord,&dest_rank); //get the rank of the corresponding process 

            //process 0 computes the correct value and sends it
            if(myGridRank == 0)
            {
                //get the correct interval for the current grid row
                low = (high != 0) ? high + 1 : high;
                high = (div_x + mod_x) + div_x * i - 1;

                //compute the size of the corresponding sub block
                int nrows = (i == 0) ? div_x + mod_x : div_x;
                int ncols = (j == 0) ? div_y + mod_y : div_y;
                int size = nrows * ncols;
                
                //set the buffers
                int buffer_i[size];
                int buffer_j[size];
                int buffer_x[size]; 
                nnz = 0;

                //select correct value
                for(int k = low_p; k <= high_p; k++)
                {
                    if(M->i[k] >= low && M->i[k] <= high)
                    {   
                        //shift the coord of each point x
                        buffer_i[nnz] = (i == 0) ? M->i[k] : M->i[k] - low;
                        buffer_j[nnz] = (j == 0) ? M->j[k] : M->j[k] - (max_col - (div_y * j - 1));
                        buffer_x[nnz] = M->x[k];
                        nnz++; 
                    }
                }

                //case 1 : current process is 0
                if(dest_rank == myGridRank && nnz != 0)
                {
                    //malloc
                    int *mi = malloc(nnz * sizeof(*mi));
                    int *mj = malloc(nnz * sizeof(*mj));
                    u32 *mx = malloc(nnz * sizeof(*mx));

                    //check the allocation
                    if(mi == NULL || mj == NULL || mx == NULL)
                    {
                        printf("Cannot allocate memory for sub block m\n");
                        MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
                    }

                    //get values from buffers
                    for(long int k = 0; k <= nnz; k++)
                    {
                        mi[k] = buffer_i[k];
                        mj[k] = buffer_j[k]; 
                        mx[k] = buffer_x[k];
                    }

                    //link to sub block m
                    m->nnz = nnz;
                    m->i = mi;
                    m->j = mj;
                    m->x = mx;
                }

                //case 2 : send all the values to dest_rank process
                else
                {
                    MPI_Send(&nnz,1,MPI_LONG,dest_rank,0,gridComm);
                    MPI_Send(buffer_i,nnz,MPI_INT,dest_rank,1,gridComm);
                    MPI_Send(buffer_j,nnz,MPI_INT,dest_rank,2,gridComm);
                    MPI_Send(buffer_x,nnz,MPI_INT,dest_rank,3,gridComm);
                }
            }

            //process dest_rank receiving values from process 0
            else if(dest_rank == myGridRank)
            {
                //get first nnz 
                MPI_Recv(&nnz,1,MPI_LONG,0,0,gridComm,&status);

                //malloc
                int *mi = malloc(nnz * sizeof(*mi));
                int *mj = malloc(nnz * sizeof(*mj));
                u32 *mx = malloc(nnz * sizeof(*mx));

                //check the allocation
                if(mi == NULL || mj == NULL || mx == NULL)
                {
                    printf("Cannot allocate memory for sub block m\n");
                    MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
                }

                //get the rest 
                MPI_Recv(mi,nnz,MPI_INT,0,1,gridComm,&status);
                MPI_Recv(mj,nnz,MPI_INT,0,2,gridComm,&status);
                MPI_Recv(mx,nnz,MPI_INT,0,3,gridComm,&status);

                //link to sub block m
                m->nnz = nnz;
                m->i = mi;
                m->j = mj;
                m->x = mx;
            }
        }
    }
}

/* Alloc memory for the sub vector v and get its value */ 
void mpi_create_vector_block(const struct sparsematrix_t *M, const u32* V, u32** v, int n, bool transpose)
{
    MPI_Status status; //result of MPI_Recv call
    int coord[2] = {0,0}; //to store the grid coord of the current process
    int dest_rank; //destination rank in the grid

    //get the slicing for the current coord
    int dim = (transpose) ? M->nrows : M->ncols;
    int gridDim = (transpose) ? dims[0] : dims[1];
    int div = dim / gridDim;
    int mod = dim % gridDim;

    //interval for vector value corresponding to the position of the current process 
    int low;
    int high = 0;

    /* per grid column or row if transpose == true */ 
    for(int j = 0; j < gridDim; j++)
    {
        //set according to transpose the correct coord 
        if(transpose)
        {
            coord[0] = j;
        }

        else
        {
            coord[1] = j; 
        }
        
        //get the rank of the corresponding process
        MPI_Cart_rank(gridComm,coord,&dest_rank); 

        //compute the size of the corresponding sub block
        int N = (j == 0) ? div + mod : div;
        int size = n * N;

        //process 0 computes the correct value and sends it
        if(myGridRank == 0)
        {
            //set the correct interval corresponding to the current process 
            low = (j == 0) ? 0 : high;
            high = (j == gridDim - 1) ? dim * n : ((div + mod) + div * j) * n;

            //set the buffer
            int buffer[size];
           
            //select correct value
            for(int k = low; k < high; k++)
            {
                buffer[k - low] = V[k];
            }

            //case 1 : current process is 0
            if(dest_rank == myGridRank)
            {
                //malloc
                *v = malloc(size * sizeof(*v));

                //check the allocation
                if(*v == NULL)
                {
                    printf("Cannot allocate memory for sub block v\n");
                    MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
                }

                //get values from buffer
                for(int k = 0; k < size; k++)
                {
                    (*v)[k] = buffer[k];
                }
            }

            //case 2 : send all the value to dest_rank process
            else
            {
                MPI_Send(buffer,size,MPI_INT,dest_rank,0,gridComm);
            }
        }

        //process dest_rank receiving values from process 0
        else if(dest_rank == myGridRank)
        {
            //malloc
            *v = malloc(size * sizeof(*v));
            
            //check the allocation
            if(*v == NULL)
            {
                printf("Cannot allocate memory for sub block v\n");
                MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
            }
            
            //get the value
            MPI_Recv(*v,size,MPI_INT,0,0,gridComm,&status);
        }

        //set the condition to broadcast according to transpose
        bool condition = (transpose) ? myGridCoord[0] == j : myGridCoord[1] == j; 

        /* broadcast to all process in the same row if transpose == true or column otherwise */
        if(condition)
        {
            //first the receiving processes alloc memory for the sub block v
            if(myGridRank != dest_rank)
            {
                //malloc
                *v = malloc(size * sizeof(*v));
                
                //check the allocation
                if(*v == NULL)
                {
                    printf("Cannot allocate memory for sub block v\n");
                    MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
                }
            }

            //and then, broadcast v 
            (transpose) ? MPI_Bcast(*v,size,MPI_INT,0,rowComm) : MPI_Bcast(*v,size,MPI_INT,0,colComm);
        }
    }
}

/* Y += m*v or Y += transpose(m)*v, according to the transpose flag */ 
void mpi_matrix_vector_product(u32 * Y, const struct sparsematrix_t *m, const u32* v, int n, bool transpose)
{
	//set some variables
    long nnz = m->nnz;
    int nrows = transpose ? m->ncols : m->nrows;
    int size = nrows * n;
    
    int const * mi = m->i;
    int const * mj = m->j;
    u32 const * mx = m->x;

    u32 tmp[size];
    u32 y[size];
    
	//prepare the arrays before computing
    for(long i = 0; i < size; i++)
    {
        tmp[i] = 0;
        y[i] = 0;
    }

	//compute the partial product with m and v 
    for(long k = 0; k < nnz; k++)
    {
        int i = transpose ? mj[k] : mi[k];
        int j = transpose ? mi[k] : mj[k];
        u64 x = mx[k];

        for(int l = 0; l < n; l++)
        {
            u64 a = tmp[i * n + l];
            u64 b = v[j * n + l];
            tmp[i * n + l] = (a + x * b) % prime;
        }
    }

	//sum by row/column (each process at the first column/row of the grid will get the sum)
    (transpose) ? MPI_Reduce(tmp,y,size,MPI_INT,MPI_SUM,0,colComm) : MPI_Reduce(tmp,y,size,MPI_INT,MPI_SUM,0,rowComm);

    //set the condition to gather according to transpose
    bool condition = (transpose) ? myGridCoord[0] == 0 : myGridCoord[1] == 0;

	//gather all the values by the first column/row
    if(condition)
    {
        int gridDim = (transpose) ? dims[0] : dims[1]; //get the correct dimension
		int counts[gridDim]; //number of points for each process
		int disps[gridDim]; //the displacement of these points in the array Y

		//get the counts from the other process in the same column/row
		(transpose) ? MPI_Gather(&size,1,MPI_INT,counts,1,MPI_INT,0,rowComm) : MPI_Gather(&size,1,MPI_INT,counts,1,MPI_INT,0,colComm);

		//compute the displacement based on the counts
		for(int i = 0; i < gridDim; i++)
		{
			disps[i] = (i > 0) ? (disps[i-1] + counts[i-1]) : 0;
		}

		//gather all the data in Y 
		(transpose) ? MPI_Gatherv(y,size,MPI_INT,Y,counts,disps,MPI_INT,0,rowComm) : MPI_Gatherv(y,size,MPI_INT,Y,counts,disps,MPI_INT,0,colComm);
    }
}

/* Free used memory */
void mpi_free_matrices(struct sparsematrix_t *M, struct sparsematrix_t *m)
{
    //process 0 has the matrix M
    if(myGridRank == 0)
    {
        free(M->i);
        free(M->j);
        free(M->x);
    }

    free(m->i);
    free(m->j);
    free(m->x);
}

/*************************** block-Lanczos algorithm ************************/

/* Computes vtAv <-- transpose(v) * Av, vtAAv <-- transpose(Av) * Av */
void block_dot_products(u32 * vtAv, u32 * vtAAv, int N, u32 const * Av, u32 const * v)
{
        for (int i = 0; i < n * n; i++)
                vtAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAv, &v[i*n], &Av[i*n]);
        
        for (int i = 0; i < n * n; i++)
                vtAAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAAv, &Av[i*n], &Av[i*n]);
}

/* Compute the next values of v (in tmp) and p */
void orthogonalize(u32 * v, u32 * tmp, u32 * p, u32 * d, u32 const * vtAv, const u32 *vtAAv, 
        u32 const * winv, int N, u32 const * Av)
{
        /* compute the n x n matrix c */
        u32 c[n * n];
        u32 spliced[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                        spliced[i*n + j] = d[j] ? vtAAv[i * n + j] : vtAv[i * n + j];
                        c[i * n + j] = 0;
                }
        matmul_CpAB(c, winv, spliced);
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        c[i * n + j] = prime - c[i * n + j];

        u32 vtAvd[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        vtAvd[i*n + j] = d[j] ? prime - vtAv[i * n + j] : 0;

        /* compute the next value of v ; store it in tmp */        
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        tmp[i*n + j] = d[j] ? Av[i*n + j] : v[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &v[i*n], c);
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i*n], &p[i*n], vtAvd);
        
        /* compute the next value of p */
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        p[i * n + j] = d[j] ? 0 : p[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&p[i*n], &v[i*n], winv);
}

void verbosity()
{
        n_iterations += 1;
        double elapsed = wtime() - start;
        if (elapsed - last_print < 1) 
                return;

        last_print = elapsed;
        double per_iteration = elapsed / n_iterations;
        double estimated_length = expected_iterations * per_iteration;
        time_t end = start + estimated_length;
        if (!ETA_flag) {
                int d = estimated_length / 86400;
                estimated_length -= d * 86400;
                int h = estimated_length / 3600;
                estimated_length -= h * 3600;
                int m = estimated_length / 60;
                estimated_length -= m * 60;
                int s = estimated_length;
                printf("    - Expected duration : ");
                if (d > 0)
                        printf("%d j ", d);
                if (h > 0)
                        printf("%d h ", h);
                if (m > 0)
                        printf("%d min ", m);
                printf("%d s\n", s);
                ETA_flag = true;
        }
        char ETA[30];
        ctime_r(&end, ETA);
        ETA[strlen(ETA) - 1] = 0;  // élimine le \n final
        printf("\r    - iteration %d / %d. %.3fs per iteration. ETA: %s", 
                n_iterations, expected_iterations, per_iteration, ETA);
        fflush(stdout);
}

/* optional tests */
void correctness_tests(u32 const * vtAv, u32 const * vtAAv, u32 const * winv, u32 const * d)
{
        /* vtAv, vtAAv, winv are actually symmetric + winv and d match */
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        assert(vtAv[i*n + j] == vtAv[j*n + i]);
                        assert(vtAAv[i*n + j] == vtAAv[j*n + i]);
                        assert(winv[i*n + j] == winv[j*n + i]);
                        assert((winv[i*n + j] == 0) || d[i] || d[j]);
                }
        /* winv satisfies d == winv * vtAv*d */
        u32 vtAvd[n * n];
        u32 check[n * n];
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++) {
                        vtAvd[i*n + j] = d[j] ? vtAv[i*n + j] : 0;
                        check[i*n + j] = 0;
                }
        matmul_CpAB(check, winv, vtAvd);
        for (int i = 0; i < n; i++) 
                for (int j = 0; j < n; j++)
                        if (i == j)
                                assert(check[j*n + j] == d[i]);
                        else
                                assert(check[i*n + j] == 0);
}

/* check that we actually computed a kernel vector */
void final_check(int nrows, int ncols, u32 const * v, u32 const * vtM)
{
        printf("Final check:\n");
        /* Check if v != 0 */
        bool good = false;
        for (long i = 0; i < nrows; i++)
                for (long j = 0; j < n; j++)
                        good |= (v[i*n + j] != 0);
        if (good)
                printf("  - OK:    v != 0\n");
        else
                printf("  - KO:    v == 0\n");
                
        /* tmp == Mt * v. Check if tmp == 0 */
        good = true;
        for (long i = 0; i < ncols; i++)
                for (long j = 0; j < n; j++)
                        good &= (vtM[i*n + j] == 0);
        if (good)
                printf("  - OK: vt*M == 0\n");
        else
                printf("  - KO: vt*M != 0\n");                
}

/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 * block_lanczos(struct sparsematrix_t const * M, struct sparsematrix_t const * m, int n, bool transpose)
{
	//global variables
	int nrows = 0;
	int ncols = 0;
	long block_size = 0;
	u32 *v = NULL;
	u32 *tmp = NULL;
	u32 *Av = NULL;
	u32 *p = NULL;

	//process 0 initiates the algorithm  	
	if(myGridRank == 0)
	{
		printf("Block Lanczos\n");
		nrows = transpose ? M->ncols : M->nrows;
		ncols = transpose ? M->nrows : M->ncols;
		block_size = nrows * n;
		long Npad = ((nrows + n - 1) / n) * n;
		long Mpad = ((ncols + n - 1) / n) * n;
		long block_size_pad = (Npad > Mpad ? Npad : Mpad) * n;

		char human_size[8];
		human_format(human_size, 4 * sizeof(int) * block_size_pad);
		printf("  - Extra storage needed: %sB\n", human_size);

		v = malloc(sizeof(*v) * block_size_pad);
		tmp = malloc(sizeof(*tmp) * block_size_pad);
		Av = malloc(sizeof(*Av) * block_size_pad);
		p = malloc(sizeof(*p) * block_size_pad);

		if (v == NULL || tmp == NULL || Av == NULL || p == NULL)
		{
		    printf("impossible d'allouer les blocs de vecteur\n");
            MPI_Abort(MPI_COMM_WORLD,MPI_ERR_NO_MEM);
		}
		
		/* warn the user */
		expected_iterations = 1 + ncols / n;
		char human_its[8];
		human_format(human_its, expected_iterations);
		printf("  - Expecting %s iterations\n", human_its);
		
		/* prepare initial values */
		for (long i = 0; i < block_size_pad; i++)
		{
			Av[i] = 0;
			v[i] = 0;
			p[i] = 0;
			tmp[i] = 0;
		}

		for (long i = 0; i < block_size; i++)
			v[i] = random64() % prime;
	}

	/************* main loop *************/
	if(myGridRank == 0)
	{
		printf("  - Main loop\n");
		start = wtime();
	}
	
	bool stop = false;
	u32* sub_v = NULL; //sub block of the vector 

	while(true)
	{
		//only the process 0 handles the execution and when it stops
		if(myGridRank == 0 && stop_after > 0 && n_iterations == stop_after)
			stop = true;

		//warn all the processes to if we have finished
		MPI_Bcast(&stop,1,MPI_INT,0,gridComm);
    				
		if (stop)
			break;

		//parallel matrix-vector multiplication (u <- Mt * v)
		mpi_create_vector_block(M,v,&sub_v,n,!transpose);
    	mpi_matrix_vector_product(tmp,m,sub_v,n,!transpose);

		//free sub_v for the next multiplication
		if(sub_v != NULL)
		{
			free(sub_v);
			sub_v = NULL;
		}

		//parallel matrix-vector multiplication (v <- M * u)
		mpi_create_vector_block(M,tmp,&sub_v,n,transpose);
    	mpi_matrix_vector_product(Av,m,sub_v,n,transpose);

		//free sub_v for the next iteration
		if(sub_v != NULL)
		{
			free(sub_v);
			sub_v = NULL;
		}

		//n * n products
		u32 vtAv[n * n];
		u32 vtAAv[n * n];
		u32 winv[n * n];
		u32 d[n];

		//process 0 computes n * n products 
		if(myGridRank == 0)
		{
			block_dot_products(vtAv, vtAAv, nrows, Av, v);
			stop = (semi_inverse(vtAv, winv, d) == 0);

			/* check that everything is working ; disable in production */
			correctness_tests(vtAv, vtAAv, winv, d);
		}

		//warn all the processes to if we have finished
		MPI_Bcast(&stop,1,MPI_INT,0,gridComm);
    				
		if (stop)
			break;

		//process 0 prepares the next iteration
		if(myGridRank == 0)
		{
			orthogonalize(v, tmp, p, d, vtAv, vtAAv, winv, nrows, Av);

			/* the next value of v is in tmp ; copy */
			for (long i = 0; i < block_size; i++)
				v[i] = tmp[i];

			verbosity();
		}
	}

	//process 0 prints out the execution time and result
	if(myGridRank == 0)
	{
		printf("\n");
		if (stop_after < 0)
			final_check(nrows, ncols, v, tmp);
		printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);
		free(tmp);
		free(Av);
		free(p);
	}
	
	return v;
}

/**************************** dense vector block IO ************************/

void save_vector_block(char const * filename, int nrows, int ncols, u32 const * v)
{
        printf("Saving result in %s\n", filename);
        FILE * f = fopen(filename, "w");
        if (f == NULL)
                err(1, "cannot open %s", filename);
        fprintf(f, "%%%%MatrixMarket matrix array integer general\n");
        fprintf(f, "%%block of left-kernel vector computed by lanczos_modp\n");
        fprintf(f, "%d %d\n", nrows, ncols);
        for (long j = 0; j < ncols; j++)
                for (long i = 0; i < nrows; i++)
                        fprintf(f, "%d\n", v[i*n + j]);
        fclose(f);
}

/*************************** main function *********************************/

int main(int argc, char ** argv)
{
	struct sparsematrix_t M; //matrix loaded from the file
	struct sparsematrix_t m; //sub block of the matrix M

	//init MPI 
	mpi_init(&M,&m); 

	//get arg from command line
	process_command_line_options(argc, argv); 

	//prepare sub blocks m
	mpi_get_matrix_size(&M,matrix_filename);
	mpi_create_matrix_block(&M,&m);

	//all processes solve the problem
	u32 *kernel = block_lanczos(&M, &m, n, right_kernel);

	//process 0 clean up
	if(myGridRank == 0)
	{
		if (kernel_filename)
				save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
		else
				printf("Not saving result (no --output given)\n");
		free(kernel);
	}

	//processes clean up
	mpi_free_matrices(&M,&m);
	MPI_Finalize();

	exit(EXIT_SUCCESS);
}