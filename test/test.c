#include <omp.h>
#include "stdio.h"

int main()
{
    int arr[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    #pragma omp parallel for 
    for(int j = 0; j < 8; j++)
    {
        #pragma omp parallel for
        for(int i = 0; i<16; i++)
        {
            printf("%d ", arr[i]);
        }

        printf("\n");
    }
    
    
    printf("\n");

    return 0;
}