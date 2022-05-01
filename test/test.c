#include <mpi.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <err.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mmio.h"

typedef uint64_t u64;
typedef uint32_t u32;

//modulo p
u64 prime = 293;

/******************* sparse matrix data structure **************/

struct sparsematrix_t {
        int nrows;        // dimensions
        int ncols;
        long int nnz;     // number of non-zero coefficients
        int *i;           // row indices (for COO matrices)
        int *j;           // column indices
        u32 *x;           // coefficients
};

/******************* utility functions ********************/

/*get current time */
double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

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
                    printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", filename, percent, rate);
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

int main()
{
    struct sparsematrix_t M; //matrix loaded from the file
    struct sparsematrix_t m; //sub block of the matrix M

    mpi_init(&M,&m);
    mpi_get_matrix_size(&M,"../TF/Trec5.mtx");
    mpi_create_matrix_block(&M,&m);
    
    bool transpose = false;
    int dim = (transpose) ? M.nrows : M.ncols; 
    int n = 3;
    u32 V[n * dim]; //vector
    u32 temp[n * dim];
    u32* v = NULL; //sub block of the vector V
    
    for(int i = 0; i < n * dim; i++)
    {
        V[i] = i + 1;
        temp[i] = 0;
    }

    mpi_create_vector_block(&M,V,&v,n,transpose);
    mpi_matrix_vector_product(temp,&m,v,n,transpose);

    if(myGridRank == 0)
    {
        int size = n * ((transpose) ? M.ncols : M.nrows);

        for (int i = 0; i < size; i++)
            printf("Y[%d] : %d\n",i,temp[i]);
    }

    mpi_free_matrices(&M,&m);
    MPI_Finalize();

    return 0;
}